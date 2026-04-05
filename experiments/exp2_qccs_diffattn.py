"""
Experiment 2b: QCCS-Gated Differential Transformer (Eq. 3)
===========================================================
Implements the per-token inhibitory gating described in Eq. 3 of the paper:

  A_QCCS = softmax(Q1 K1^T / sqrt(d)) - (λ · g) ⊙ softmax(Q2 K2^T / sqrt(d))

where g ∈ [0,1]^n are per-token relevance scores from a task-conditioned gate.

In the EHRSHOT setting the "gate" is a lightweight binary classifier:
  - query  = task description text
  - sentence = decoded event code text (code string)
  - label = 1.0 if this code matches the task anchor LOINC, else 0.0
  - Architecture: same CharNgram MLP as QCCS gate (Section 3.2)

This directly evaluates Eq. 3 on all four completed EHRSHOT tasks.

Run: python experiments/exp2_qccs_diffattn.py
Output: figures/exp2_qccs_diffattn_results.csv
"""

import math, sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from exp3_qccs_gate import CharNgramTokenizer

MEDS_BASE   = Path("/Users/sanjaybasu/waymark-local/data/ehrshot/EHRSHOT_files/EHRSHOT_MEDS")
ASSETS_BASE = Path("/Users/sanjaybasu/waymark-local/data/ehrshot/EHRSHOT_files/EHRSHOT_ASSETS")
OUT_DIR     = Path(__file__).parent.parent / "figures"

EMBED_DIM = 64
N_HEADS   = 4
N_LAYERS  = 2
MAX_SEQ   = 256

# Task definitions: anchor LOINC codes and plain-text descriptions for the gate
LAB_TASKS = {
    "lab_anemia":          {
        "loinc": "LOINC/718-7",  "op": "<", "thresh": 11.0,
        "query": "hemoglobin anemia low red blood cell count",
    },
    "lab_hyperkalemia":    {
        "loinc": "LOINC/2823-3", "op": ">", "thresh": 5.5,
        "query": "potassium hyperkalemia elevated potassium",
    },
    "lab_hyponatremia":    {
        "loinc": "LOINC/2951-2", "op": "<", "thresh": 135.0,
        "query": "sodium hyponatremia low sodium electrolyte",
    },
    "lab_thrombocytopenia":{
        "loinc": "LOINC/777-3",  "op": "<", "thresh": 100.0,
        "query": "platelet count thrombocytopenia low platelets",
    },
}


# ── Task-conditioned gate (same arch as QCCS gate) ───────────────────────────

class TaskGate(nn.Module):
    """Lightweight gate that scores each event code given the task query."""
    def __init__(self, vocab_size: int = 5000, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim, mode="mean")
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 32),            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, q_ids: torch.Tensor, s_ids: torch.Tensor) -> torch.Tensor:
        """q_ids: (B, Lq)  s_ids: (B, Ls) → (B,) logits"""
        q = self.embed(q_ids)
        s = self.embed(s_ids)
        return self.mlp(torch.cat([q, s], dim=-1)).squeeze(-1)


def _pad_toks(tensors: list) -> torch.Tensor:
    """Pad a list of 1-D token tensors to the same length and stack."""
    max_len = max(t.shape[0] for t in tensors)
    return torch.stack([F.pad(t, (0, max_len - t.shape[0])) for t in tensors])


def train_task_gate(task_def: dict, meds: pd.DataFrame,
                    tok: CharNgramTokenizer) -> TaskGate:
    """Train gate to score each event: relevance = 1 if anchor LOINC, else 0."""
    anchor = task_def["loinc"]
    query_text = task_def["query"]

    # Build training pairs: (query, code_text) → 1 if anchor, else 0
    codes = meds["code"].dropna().unique()
    pos_codes = [c for c in codes if c == anchor]
    neg_codes = [c for c in codes if c != anchor]
    np.random.seed(42)
    n_neg = min(len(neg_codes), max(len(pos_codes) * 10, 500))
    neg_sample = np.random.choice(neg_codes, n_neg, replace=False).tolist()
    all_codes = pos_codes + neg_sample
    labels = [1.0] * len(pos_codes) + [0.0] * len(neg_sample)

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    gate = TaskGate().to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    q_tok = tok.tokenize(query_text)  # (Lq,)

    gate.train()
    for epoch in range(30):
        idx = np.random.permutation(len(all_codes))
        total_loss = 0.0
        for start in range(0, len(idx), 64):
            batch_idx = idx[start:start + 64]
            s_toks = [tok.tokenize(all_codes[j]) for j in batch_idx]
            s_ids = _pad_toks(s_toks).to(device)            # (B, Ls)
            q_ids = _pad_toks([q_tok] * len(batch_idx)).to(device)  # (B, Lq)
            logits = gate(q_ids, s_ids)
            y = torch.tensor([labels[j] for j in batch_idx],
                              dtype=torch.float, device=device)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
    gate.eval()
    return gate


def compute_gate_scores(seqs: np.ndarray, gate: TaskGate,
                        tok: CharNgramTokenizer, code_vocab_inv: dict,
                        query_text: str) -> np.ndarray:
    """
    seqs: (N, T) integer-coded sequences
    Returns: (N, T) float gate scores in [0, 1]
    """
    q_tok = tok.tokenize(query_text)  # (Lq,)
    N, T = seqs.shape
    scores = np.zeros((N, T), dtype=np.float32)
    unique_codes = np.unique(seqs[seqs > 0])

    # Cache scores per unique code
    code_score_cache: dict[int, float] = {0: 0.0}  # PAD = 0
    code_batch = list(unique_codes)
    if code_batch:
        dev = next(gate.parameters()).device
        with torch.no_grad():
            s_toks = [tok.tokenize(code_vocab_inv.get(c, f"CODE{c}"))
                      for c in code_batch]
            s_ids = _pad_toks(s_toks).to(dev)
            q_b   = _pad_toks([q_tok] * len(code_batch)).to(dev)
            logits = gate(q_b, s_ids)
            probs = torch.sigmoid(logits).cpu().numpy()
        for c, p in zip(code_batch, probs):
            code_score_cache[int(c)] = float(p)

    for n in range(N):
        for t in range(T):
            scores[n, t] = code_score_cache.get(int(seqs[n, t]), 0.0)
    return scores


# ── QCCS-gated Differential Transformer ─────────────────────────────────────

class QCCSDiffAttnLayer(nn.Module):
    """
    Implements Eq. 3:
      A_QCCS = softmax(Q1 K1^T/√d) − (λ · g) ⊙ softmax(Q2 K2^T/√d)
    where g ∈ [0,1]^T are per-token gate scores (broadcast over queries).
    """
    def __init__(self, embed_dim: int, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.hd    = embed_dim // nhead
        self.hhd   = self.hd // 2

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

        H, hhd = nhead, self.hhd
        self.lq1 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lk1 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lq2 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lk2 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lambda_init = 0.8
        self.scale = math.sqrt(self.hhd)

    def _lambda(self):
        l = (torch.exp((self.lq1 * self.lk1).sum(-1))
             - torch.exp((self.lq2 * self.lk2).sum(-1))
             + self.lambda_init)
        return l.view(1, self.nhead, 1, 1)  # B,H,1,1

    def forward(self, x: torch.Tensor,
                gate_scores: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B, T, D)
        gate_scores: (B, T) relevance weights in [0, 1]; None → Eq. 1 (standard DiffAttn)
        """
        B, T, D = x.shape
        H, hd, hhd = self.nhead, self.hd, self.hhd

        def _split(w):
            return w.view(B, T, H, hd).permute(0, 2, 1, 3)

        Q = _split(self.Wq(x)); K = _split(self.Wk(x)); V = _split(self.Wv(x))
        Q1, Q2 = Q[..., :hhd], Q[..., hhd:]
        K1, K2 = K[..., :hhd], K[..., hhd:]

        def _attn(q, k):
            scores = q @ k.transpose(-2, -1) / self.scale
            if key_padding_mask is not None:
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            return torch.softmax(scores, dim=-1)

        A1 = _attn(Q1, K1)   # B,H,T,T
        A2 = _attn(Q2, K2)

        if gate_scores is not None:
            # g: (B, T) → (B, 1, 1, T) broadcast: scale inhibition per key position
            g = gate_scores.unsqueeze(1).unsqueeze(2)   # B,1,1,T
            lam = self._lambda() * g                    # B,H,1,T
        else:
            lam = self._lambda()                        # B,H,1,1

        A = A1 - lam * A2
        out = (A @ V).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.norm(self.out(out) + x)


class QCCSDiffTransformerHead(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 nhead: int = N_HEADS, num_layers: int = N_LAYERS,
                 max_seq_len: int = MAX_SEQ):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.cls       = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers    = nn.ModuleList([
            nn.ModuleDict({
                "attn": QCCSDiffAttnLayer(embed_dim, nhead),
                "ffn":  nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(256, embed_dim)),
                "ln1":  nn.LayerNorm(embed_dim),
                "ln2":  nn.LayerNorm(embed_dim),
            }) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor,
                gate_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.embed(x) + self.pos_embed(pos)
        cls  = self.cls.expand(B, -1, -1)
        h    = torch.cat([cls, h], dim=1)

        pad_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool),
                               x == 0], dim=1)

        # Prepend a zero gate score for the CLS token
        if gate_scores is not None:
            g = torch.cat([torch.zeros(B, 1, device=gate_scores.device),
                           gate_scores], dim=1)  # B, T+1
        else:
            g = None

        for layer in self.layers:
            h = layer["attn"](layer["ln1"](h), gate_scores=g,
                              key_padding_mask=pad_mask) + h
            h = layer["ffn"](layer["ln2"](h)) + h

        return self.head(h[:, 0]).squeeze(-1)


# ── Training loop ────────────────────────────────────────────────────────────

def run_task(task: str, task_def: dict) -> dict:
    from exp2_ehrshot_diffattn import (load_meds, build_code_vocab,
                                       build_patient_sequences)
    print(f"\n{'='*60}\nTask: {task}")
    meds, labels, splits = load_meds(task)
    code_vocab = build_code_vocab(meds)
    code_vocab_inv = {v: k for k, v in code_vocab.items()}
    seqs, ys, positions = build_patient_sequences(meds, labels, task_def,
                                                  code_vocab, MAX_SEQ)
    if len(seqs) == 0:
        return {}

    # Train/test split (use subject_splits from EHRSHOT)
    sub_ids = labels["subject_id"].values
    try:
        train_mask = splits.set_index("subject_id")["split"].reindex(sub_ids) == "train"
        train_idx  = np.where(train_mask.values)[0]
        test_idx   = np.where(~train_mask.values)[0]
    except Exception:
        train_idx, test_idx = train_test_split(
            np.arange(len(seqs)), test_size=0.3, random_state=42, stratify=ys)

    train_seqs, test_seqs = seqs[train_idx], seqs[test_idx]
    train_ys,   test_ys   = ys[train_idx],   ys[test_idx]

    # Train task gate
    print("  Training task-conditioned gate...")
    tok = CharNgramTokenizer()
    gate_model = train_task_gate(task_def, meds, tok)

    # Compute gate scores
    print("  Computing per-token gate scores...")
    train_scores = compute_gate_scores(train_seqs, gate_model, tok,
                                       code_vocab_inv, task_def["query"])
    test_scores  = compute_gate_scores(test_seqs,  gate_model, tok,
                                       code_vocab_inv, task_def["query"])

    device   = (torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu"))
    vocab_sz = len(code_vocab)
    model    = QCCSDiffTransformerHead(vocab_sz).to(device)
    opt      = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(train_ys == 0).sum() / max((train_ys == 1).sum(), 1)],
                               dtype=torch.float).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_seqs_t  = torch.tensor(train_seqs,  dtype=torch.long)
    train_ys_t    = torch.tensor(train_ys,    dtype=torch.float)
    train_scores_t= torch.tensor(train_scores,dtype=torch.float)

    model.train()
    for epoch in range(40):
        perm = torch.randperm(len(train_seqs_t))
        epoch_loss = 0.0
        for start in range(0, len(perm), 64):
            idx  = perm[start:start + 64]
            xb   = train_seqs_t[idx].to(device)
            yb   = train_ys_t[idx].to(device)
            gb   = train_scores_t[idx].to(device)
            logits = model(xb, gate_scores=gb)
            loss   = criterion(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch+1:3d}  loss={epoch_loss:.4f}")

    model.eval()
    test_seqs_t   = torch.tensor(test_seqs,   dtype=torch.long)
    test_scores_t = torch.tensor(test_scores, dtype=torch.float)
    logits_list = []
    with torch.no_grad():
        for i in range(0, len(test_seqs_t), 64):
            xb = test_seqs_t[i:i+64].to(device)
            gb = test_scores_t[i:i+64].to(device)
            logits_list.append(model(xb, gate_scores=gb).cpu())
    probs = torch.sigmoid(torch.cat(logits_list)).numpy()

    if len(np.unique(test_ys)) < 2:
        print(f"  {task}: only one class in test — skipping")
        return {}

    auroc = roc_auc_score(test_ys, probs)
    auprc = average_precision_score(test_ys, probs)
    print(f"  {task}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}")
    return {"task": task, "model": "QCCS-DiffTransformer (Eq.3)",
            "test_auroc": auroc, "test_auprc": auprc}


def main():
    results = []
    out = OUT_DIR / "exp2_qccs_diffattn_results.csv"
    for task, task_def in LAB_TASKS.items():
        try:
            r = run_task(task, task_def)
            if r:
                results.append(r)
                # Save after each task so partial results survive interruption
                pd.DataFrame(results).to_csv(out, index=False)
                print(f"  [saved {len(results)} result(s) → {out}]")
        except Exception as e:
            print(f"  {task} FAILED: {e}")

    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
