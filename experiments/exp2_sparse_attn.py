"""
Experiment 2c: α-Entmax Sparse Attention on EHRSHOT
====================================================
Implements a thresholded/sparse attention variant as an alternative to
Differential Transformer (Exp 2b). Addresses reviewer Q7: "thresholded/signed
attention variants (e.g., TDA or Cog Attention)."

α-Entmax (Correia et al., EMNLP 2019) replaces softmax with α-entmax (α=1.5),
producing genuinely sparse attention weights — tokens with low relevance receive
exact-zero attention. This is a principled inhibitory mechanism: unlike DiffAttn's
subtraction of a second attention head, α-entmax achieves inhibition through
learned sparsification of a single head.

Three variants evaluated:
  - Standard:   softmax attention (baseline replication)
  - Entmax15:   α-entmax (α=1.5) — sparse, inhibitory
  - Sparsemax:  α-entmax (α=2.0) — maximally sparse (sparsemax = entmax at α=2)

Run: python experiments/exp2_sparse_attn.py
Output: figures/exp2_sparse_attn_results.csv

Runtime: ~15–20 min on CPU (3 variants × 4 tasks × 40 epochs).
"""

import math, sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from entmax import entmax15, sparsemax
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
from exp3_qccs_gate import CharNgramTokenizer
from exp2_qccs_diffattn import (train_task_gate, compute_gate_scores,
                                  LAB_TASKS, EMBED_DIM, N_HEADS, N_LAYERS, MAX_SEQ)

OUT_DIR = Path(__file__).parent.parent / "figures"

# ── Attention variants ────────────────────────────────────────────────────────

ATTN_FNS = {
    "softmax":   lambda s, dim: torch.softmax(s, dim=dim),
    "entmax15":  lambda s, dim: entmax15(s, dim=dim),
    "sparsemax": lambda s, dim: sparsemax(s, dim=dim),
}


class SparseAttnLayer(nn.Module):
    """Single-head scaled dot-product attention with pluggable attn function."""
    def __init__(self, embed_dim: int, nhead: int, attn_fn_name: str = "softmax"):
        super().__init__()
        self.nhead = nhead
        self.hd = embed_dim // nhead
        self.scale = math.sqrt(self.hd)
        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.attn_fn = ATTN_FNS[attn_fn_name]

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.nhead, self.hd

        def _split(w):
            return w.view(B, T, H, hd).permute(0, 2, 1, 3)

        Q = _split(self.Wq(x))
        K = _split(self.Wk(x))
        V = _split(self.Wv(x))

        scores = Q @ K.transpose(-2, -1) / self.scale  # B,H,T,T
        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        A = self.attn_fn(scores, dim=-1)
        out = (A @ V).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.norm(self.out(out) + x)


class SparseTransformerHead(nn.Module):
    def __init__(self, vocab_size: int, attn_fn_name: str = "softmax",
                 embed_dim: int = EMBED_DIM, nhead: int = N_HEADS,
                 num_layers: int = N_LAYERS, max_seq_len: int = MAX_SEQ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": SparseAttnLayer(embed_dim, nhead, attn_fn_name),
                "ffn": nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(),
                                     nn.Dropout(0.1), nn.Linear(256, embed_dim)),
                "ln1": nn.LayerNorm(embed_dim),
                "ln2": nn.LayerNorm(embed_dim),
            }) for _ in range(num_layers)
        ])
        self.head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        pad_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool),
                               x == 0], dim=1)
        for layer in self.layers:
            h = layer["attn"](layer["ln1"](h), key_padding_mask=pad_mask) + h
            h = layer["ffn"](layer["ln2"](h)) + h
        return self.head(h[:, 0]).squeeze(-1)


# ── Per-task runner ───────────────────────────────────────────────────────────

def run_task_variant(task: str, task_def: dict, attn_name: str) -> dict:
    from exp2_ehrshot_diffattn import (load_meds, build_code_vocab,
                                       build_patient_sequences)

    meds, labels, splits = load_meds(task)
    code_vocab = build_code_vocab(meds)
    seqs, ys, _ = build_patient_sequences(meds, labels, task_def, code_vocab, MAX_SEQ)
    if len(seqs) == 0:
        return {}

    sub_ids = labels["subject_id"].values
    try:
        train_mask = splits.set_index("subject_id")["split"].reindex(sub_ids) == "train"
        train_idx = np.where(train_mask.values)[0]
        test_idx  = np.where(~train_mask.values)[0]
    except Exception:
        train_idx, test_idx = train_test_split(
            np.arange(len(seqs)), test_size=0.3, random_state=42, stratify=ys)

    train_seqs, test_seqs = seqs[train_idx], seqs[test_idx]
    train_ys,   test_ys   = ys[train_idx],   ys[test_idx]

    device = (torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    vocab_sz = len(code_vocab)

    model = SparseTransformerHead(vocab_sz, attn_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor(
        [(train_ys == 0).sum() / max((train_ys == 1).sum(), 1)],
        dtype=torch.float).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_seqs_t = torch.tensor(train_seqs, dtype=torch.long)
    train_ys_t   = torch.tensor(train_ys,   dtype=torch.float)

    model.train()
    for epoch in range(40):
        perm = torch.randperm(len(train_seqs_t))
        for start in range(0, len(perm), 64):
            idx = perm[start:start + 64]
            xb = train_seqs_t[idx].to(device)
            yb = train_ys_t[idx].to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        if (epoch + 1) % 10 == 0:
            print(f"    [{attn_name}] epoch {epoch+1}/40")

    model.eval()
    test_seqs_t = torch.tensor(test_seqs, dtype=torch.long)
    all_logits = []
    with torch.no_grad():
        for start in range(0, len(test_seqs_t), 64):
            xb = test_seqs_t[start:start + 64].to(device)
            all_logits.append(model(xb).cpu())
    logits = torch.cat(all_logits).numpy()
    probs = 1.0 / (1.0 + np.exp(-logits))

    try:
        auroc = roc_auc_score(test_ys, probs)
        auprc = average_precision_score(test_ys, probs)
    except Exception:
        auroc = auprc = float("nan")

    # Measure attention sparsity: fraction of exact-zero attention weights in last layer
    # (entmax/sparsemax can produce exact zeros; softmax never does)
    sparsity = float("nan")
    try:
        with torch.no_grad():
            sample = test_seqs_t[:min(32, len(test_seqs_t))].to(device)
            B, T = sample.shape
            last_attn = model.layers[-1]["attn"]
            h_in = model.embed(sample) + model.pos_embed(
                torch.arange(T, device=device).unsqueeze(0))
            cls = model.cls.expand(B, -1, -1)
            h_in = torch.cat([cls, h_in], dim=1)
            H, hd, scale = last_attn.nhead, last_attn.hd, last_attn.scale

            def _sp(w):
                return w.view(B, T+1, H, hd).permute(0, 2, 1, 3)

            Q = _sp(last_attn.Wq(h_in))
            K = _sp(last_attn.Wk(h_in))
            s = Q @ K.transpose(-2, -1) / scale
            A = last_attn.attn_fn(s, dim=-1)
            sparsity = float((A == 0).float().mean().item())
    except Exception:
        pass

    return {"task": task, "attn": attn_name,
            "auroc": round(auroc, 4), "auprc": round(auprc, 4),
            "attn_sparsity": round(sparsity, 4) if not np.isnan(sparsity) else float("nan")}


def main():
    results = []
    for attn_name in ["softmax", "entmax15", "sparsemax"]:
        print(f"\n{'='*60}\nAttention: {attn_name}")
        for task, task_def in LAB_TASKS.items():
            print(f"  Task: {task}")
            r = run_task_variant(task, task_def, attn_name)
            if r:
                results.append(r)
                print(f"    AUROC={r['auroc']:.4f}  AUPRC={r['auprc']:.4f}  "
                      f"sparsity={r['attn_sparsity']:.3f}")

    df = pd.DataFrame(results)
    out = OUT_DIR / "exp2_sparse_attn_results.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Pivot summary
    print("\n=== AUROC by task and attention variant ===")
    pivot = df.pivot(index="task", columns="attn", values="auroc")
    print(pivot.to_string())
    print("\n=== Attention sparsity (fraction exact-zero weights) ===")
    sp = df.pivot(index="task", columns="attn", values="attn_sparsity")
    print(sp.to_string())


if __name__ == "__main__":
    main()
