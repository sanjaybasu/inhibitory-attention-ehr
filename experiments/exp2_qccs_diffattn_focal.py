"""
Experiment 2b variant: QCCS-Gated Differential Transformer with focal loss.

Addresses reviewer concern: the per-token gate in exp2_qccs_diffattn.py
was trained with plain BCE, potentially suffering from gradient starvation
under extreme class imbalance (e.g., ~2.4% positive prevalence for hyperkalemia).

This script is identical to exp2_qccs_diffattn.py except that train_task_gate
uses focal BCE loss (gamma=2) with positive-class reweighting for the gate,
attempting to rescue per-token inhibitory gating from gradient starvation.

If AUROC remains near the standard DiffAttn baseline (as in the main paper),
this confirms that gradient starvation is fundamental to per-token gating at
this imbalance ratio, not an artifact of the loss function choice.

Usage:
  python experiments/exp2_qccs_diffattn_focal.py
Output: figures/exp2_qccs_diffattn_focal_results.csv
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
# Import architecture from main module (reuse QCCSDiffTransformerHead, etc.)
from exp2_qccs_diffattn import (
    LAB_TASKS, TaskGate, QCCSDiffTransformerHead,
    _pad_toks, compute_gate_scores,
    EMBED_DIM, N_HEADS, N_LAYERS, MAX_SEQ,
    MEDS_BASE, ASSETS_BASE,
)

OUT_DIR = Path(__file__).parent.parent / "figures"

FOCAL_GAMMA = 2.0   # focal loss concentration parameter


def focal_bce_loss(logits: torch.Tensor, y: torch.Tensor,
                   gamma: float = FOCAL_GAMMA,
                   pos_weight_val: float = 1.0) -> torch.Tensor:
    """
    Focal binary cross-entropy with optional positive-class weighting.

    focal_loss = -alpha_t * (1 - p_t)^gamma * log(p_t)

    where p_t = sigmoid(logits) for y=1, else 1 - sigmoid(logits),
    and alpha_t = pos_weight_val for y=1, else 1.0.

    This down-weights easy negatives so the gate gradient is not dominated
    by the overwhelmingly prevalent negative class.
    """
    p = torch.sigmoid(logits)
    pt = torch.where(y == 1, p, 1.0 - p)
    alpha_t = torch.where(y == 1,
                          torch.full_like(y, pos_weight_val),
                          torch.ones_like(y))
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    focal_weight = (1.0 - pt) ** gamma
    loss = alpha_t * focal_weight * ce
    return loss.mean()


def train_task_gate_focal(task_def: dict, meds: pd.DataFrame,
                          tok: CharNgramTokenizer) -> TaskGate:
    """
    Train gate with focal BCE + positive-class reweighting.
    Identical to train_task_gate in exp2_qccs_diffattn.py except for the loss.
    """
    anchor = task_def["loinc"]
    query_text = task_def["query"]

    codes = meds["code"].dropna().unique()
    pos_codes = [c for c in codes if c == anchor]
    neg_codes = [c for c in codes if c != anchor]
    np.random.seed(42)
    n_neg = min(len(neg_codes), max(len(pos_codes) * 10, 500))
    neg_sample = np.random.choice(neg_codes, n_neg, replace=False).tolist()
    all_codes = pos_codes + neg_sample
    labels = [1.0] * len(pos_codes) + [0.0] * len(neg_sample)

    # pos_weight = ratio of negatives to positives (class imbalance correction)
    n_pos = max(len(pos_codes), 1)
    n_total_neg = len(neg_sample)
    pw = n_total_neg / n_pos
    print(f"    focal gate: {len(pos_codes)} pos, {n_neg} neg, pos_weight={pw:.1f}")

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    gate = TaskGate().to(device)
    opt = torch.optim.AdamW(gate.parameters(), lr=1e-3, weight_decay=1e-4)
    q_tok = tok.tokenize(query_text)

    gate.train()
    for epoch in range(30):
        idx = np.random.permutation(len(all_codes))
        total_loss = 0.0
        for start in range(0, len(idx), 64):
            batch_idx = idx[start:start + 64]
            s_toks = [tok.tokenize(all_codes[j]) for j in batch_idx]
            s_ids = _pad_toks(s_toks).to(device)
            q_ids = _pad_toks([q_tok] * len(batch_idx)).to(device)
            logits = gate(q_ids, s_ids)
            y = torch.tensor([labels[j] for j in batch_idx],
                              dtype=torch.float, device=device)
            # Focal BCE with pos_weight
            loss = focal_bce_loss(logits, y, gamma=FOCAL_GAMMA,
                                  pos_weight_val=min(pw, 20.0))  # cap at 20
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item()
    gate.eval()
    return gate


def run_task_focal(task: str, task_def: dict) -> dict:
    from exp2_ehrshot_diffattn import (load_meds, build_code_vocab,
                                       build_patient_sequences)
    print(f"\n{'='*60}\nTask: {task} [focal gate]")
    meds, labels, splits = load_meds(task)
    code_vocab = build_code_vocab(meds)
    code_vocab_inv = {v: k for k, v in code_vocab.items()}
    seqs, ys, positions = build_patient_sequences(meds, labels, task_def,
                                                  code_vocab, MAX_SEQ)
    if len(seqs) == 0:
        return {}

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

    print("  Training task-conditioned gate (focal BCE)...")
    tok = CharNgramTokenizer()
    gate_model = train_task_gate_focal(task_def, meds, tok)

    print("  Computing per-token gate scores...")
    train_scores = compute_gate_scores(train_seqs, gate_model, tok,
                                       code_vocab_inv, task_def["query"])
    test_scores  = compute_gate_scores(test_seqs,  gate_model, tok,
                                       code_vocab_inv, task_def["query"])

    # Gate score diagnostics (check for collapse)
    all_scores_flat = test_scores[test_seqs > 0].ravel()
    print(f"    Gate scores on test non-pad tokens: "
          f"mean={all_scores_flat.mean():.4f}  std={all_scores_flat.std():.4f}  "
          f"min={all_scores_flat.min():.4f}  max={all_scores_flat.max():.4f}")

    device   = (torch.device("cuda") if torch.cuda.is_available()
                else torch.device("mps") if torch.backends.mps.is_available()
                else torch.device("cpu"))
    vocab_sz = len(code_vocab)
    model    = QCCSDiffTransformerHead(vocab_sz).to(device)
    opt      = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pos_weight = torch.tensor([(train_ys == 0).sum() / max((train_ys == 1).sum(), 1)],
                               dtype=torch.float).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_seqs_t   = torch.tensor(train_seqs,   dtype=torch.long)
    train_ys_t     = torch.tensor(train_ys,     dtype=torch.float)
    train_scores_t = torch.tensor(train_scores, dtype=torch.float)

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
    gate_mean = float(all_scores_flat.mean())
    gate_std  = float(all_scores_flat.std())
    collapsed = gate_mean > 0.45 and gate_mean < 0.55 and gate_std < 0.05

    print(f"  {task}: AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
          f"gate_mean={gate_mean:.3f}  gate_std={gate_std:.3f}  "
          f"{'COLLAPSED' if collapsed else 'ok'}")
    return {
        "task": task,
        "model": "QCCS-DiffTransformer-Focal (gamma=2)",
        "test_auroc": auroc,
        "test_auprc": auprc,
        "gate_mean": gate_mean,
        "gate_std": gate_std,
        "gate_collapsed": collapsed,
    }


def main():
    results = []
    out = OUT_DIR / "exp2_qccs_diffattn_focal_results.csv"
    for task, task_def in LAB_TASKS.items():
        try:
            r = run_task_focal(task, task_def)
            if r:
                results.append(r)
                pd.DataFrame(results).to_csv(out, index=False)
                print(f"  [saved {len(results)} result(s) → {out}]")
        except Exception as e:
            print(f"  {task} FAILED: {e}")

    df = pd.DataFrame(results)
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    print(df.to_string(index=False))

    # Load original results for comparison
    orig_path = OUT_DIR / "exp2_qccs_diffattn_results.csv"
    if orig_path.exists():
        orig = pd.read_csv(orig_path)
        print("\n=== Focal vs. Plain BCE Gate ===")
        for task in LAB_TASKS:
            orig_row = orig[orig["task"] == task]
            focal_row = df[df["task"] == task]
            if len(orig_row) > 0 and len(focal_row) > 0:
                print(f"  {task}:  plain={orig_row['test_auroc'].values[0]:.3f}  "
                      f"focal={focal_row['test_auroc'].values[0]:.3f}")


if __name__ == "__main__":
    main()
