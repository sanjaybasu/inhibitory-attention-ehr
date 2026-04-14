"""
Experiment 3: QCCS Gate Ablations — n-gram × embed_dim × MLP depth
===================================================================
Trains 12 gate variants and evaluates Stage-1 recall + calibration gap.

Ablation grid:
  n-gram:    {2, 3, 4}
  embed_dim: {32, 64, 128}
  mlp_depth: {"shallow", "standard", "deep"}

The "standard" variant (ngram=3, embed_dim=64) is the existing baseline.

Output: figures/exp3_gate_ablations.csv
Columns: ngram, embed_dim, mlp_depth, stage1_recall_pct, pos_score_mean,
         neg_score_mean, calibration_gap

Run: python experiments/exp3_gate_ablations.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from exp3_qccs_gate import (
    parse_ehr_sentences,
    build_gate_training_data,
    GateDataset,
    train_gate,
    QCCSGate,
    CharNgramTokenizer,
    qccs_compress,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
TSV_CLIN = BASE / "medalign_instructions_v1_3/clinician-instruction-responses.tsv"
TSV_REV  = BASE / "medalign_instructions_v1_3/clinician-reviewed-model-responses.tsv"
EHR_DIR  = BASE / "medalign_instructions_v1_3/ehrs"
FIGURES  = Path("/Users/sanjaybasu/waymark-local/notebooks/inhibitory-attention-ehr/figures")
FIGURES.mkdir(parents=True, exist_ok=True)


# ── Variant gate constructor ───────────────────────────────────────────────────

def build_gate(vocab_size: int, embed_dim: int, mlp_depth: str) -> QCCSGate:
    """
    Return a QCCSGate with the specified MLP depth.
    MLP input size = embed_dim * 2.
    """
    inp = embed_dim * 2
    if mlp_depth == "shallow":
        mlp = nn.Sequential(nn.Linear(inp, 1))
    elif mlp_depth == "standard":
        mlp = nn.Sequential(
            nn.Linear(inp, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    else:  # deep
        mlp = nn.Sequential(
            nn.Linear(inp, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
    gate = QCCSGate(vocab_size=vocab_size, embed_dim=embed_dim)
    gate.mlp = mlp
    return gate


# ── Stage-1 recall evaluation ─────────────────────────────────────────────────

def evaluate_stage1_recall(
    df_test: pd.DataFrame,
    xml_cache: dict,
    gate: QCCSGate,
    tokenizer: CharNgramTokenizer,
    keep_top_k: int = 20,
    device: str = "cpu",
) -> tuple[float, float, float]:
    """
    For each test instruction, check whether the gold-evidence sentence is
    retained in QCCS top-k=20 selections using this gate.

    Gold-evidence sentence: any EHR sentence that contains at least one word
    (≥4 chars) from the evidence field.

    Returns: (stage1_recall_pct, pos_score_mean, neg_score_mean)
    """
    gate.eval()
    hits = 0
    total = 0
    all_pos_scores = []
    all_neg_scores = []

    for _, row in df_test.iterrows():
        query    = str(row.get("question", ""))
        evidence = str(row.get("evidence", ""))
        fname    = str(row.get("filename", ""))
        if not query or not evidence or not fname:
            continue
        events = xml_cache.get(fname, [])
        if not events:
            continue

        ev_words = set(
            tok for tok in re.findall(r"\w+", evidence.lower()) if len(tok) >= 4
        )
        if not ev_words:
            continue

        # Score each event
        scores = []
        is_positive = []
        with torch.no_grad():
            for ev in events:
                import torch.nn.functional as F
                q_tok = tokenizer.tokenize(query)
                s_tok = tokenizer.tokenize(ev["text"])
                max_len = 128
                q_pad = F.pad(q_tok, (0, max(0, max_len - len(q_tok))))[:max_len]
                s_pad = F.pad(s_tok, (0, max(0, max_len - len(s_tok))))[:max_len]
                score = torch.sigmoid(
                    gate(q_pad.unsqueeze(0).to(device),
                         s_pad.unsqueeze(0).to(device))
                ).item()
                scores.append(score)
                text_words = set(re.findall(r"\w+", ev["text"].lower()))
                is_positive.append(bool(ev_words & text_words))

        # Collect calibration scores
        for sc, pos in zip(scores, is_positive):
            if pos:
                all_pos_scores.append(sc)
            else:
                all_neg_scores.append(sc)

        # Top-k keep set
        scored_idx = sorted(range(len(scores)), key=lambda i: -scores[i])
        top_idx = set(scored_idx[:keep_top_k])
        # Always keep last 5 (recency buffer, consistent with qccs_compress)
        top_idx |= set(range(max(0, len(events) - 5), len(events)))

        # Does the gold sentence appear in the kept set?
        evidence_in_top = any(
            is_positive[i] and i in top_idx for i in range(len(events))
        )
        hits += int(evidence_in_top)
        total += 1

    recall_pct = 100.0 * hits / total if total > 0 else 0.0
    pos_mean = float(np.mean(all_pos_scores)) if all_pos_scores else 0.0
    neg_mean = float(np.mean(all_neg_scores)) if all_neg_scores else 0.0
    return recall_pct, pos_mean, neg_mean


# ── Main ──────────────────────────────────────────────────────────────────────

def run_ablations(device_str: str = "cpu"):
    device = device_str
    print("\n=== Experiment 3: Gate Ablations ===")

    # Load training data (clinician-instruction-responses)
    df_clin = pd.read_csv(TSV_CLIN, sep="\t", low_memory=False)

    # Load test data (clinician-reviewed-model-responses, deduplicated)
    df_rev = pd.read_csv(TSV_REV, sep="\t", low_memory=False)
    df_rev = df_rev.dropna(subset=["question", "evidence", "filename"])
    df_rev = df_rev.drop_duplicates(subset=["filename", "question"])

    # Build XML cache over all unique files
    all_files = list(
        set(df_clin["filename"].dropna().unique()) |
        set(df_rev["filename"].dropna().unique())
    )
    print(f"  Loading XML cache ({len(all_files)} files)...")
    xml_cache = {}
    for fname in all_files:
        p = EHR_DIR / str(fname)
        if p.exists():
            xml_cache[str(fname)] = parse_ehr_sentences(p)
    print(f"  Cached {len(xml_cache)} patient XMLs")

    # Same train/test split as exp3_qccs_gate.py (random_state=42, test_size=0.3)
    cached_files = list(xml_cache.keys())
    train_files, test_files = train_test_split(
        cached_files, test_size=0.30, random_state=42
    )
    train_set = set(train_files)
    test_set  = set(test_files)

    df_train = df_clin[df_clin["filename"].astype(str).isin(train_set)]
    df_test  = df_rev[df_rev["filename"].astype(str).isin(test_set)]
    print(f"  Train: {len(df_train)} rows | Test: {len(df_test)} instructions")

    # Ablation grid
    ngram_vals    = [2, 3, 4]
    embed_vals    = [32, 64, 128]
    depth_vals    = ["shallow", "standard", "deep"]

    rows = []
    total_variants = len(ngram_vals) * len(embed_vals) * len(depth_vals)
    done = 0

    for ngram in ngram_vals:
        for embed_dim in embed_vals:
            for mlp_depth in depth_vals:
                done += 1
                print(f"\n  [{done}/{total_variants}] ngram={ngram}, "
                      f"embed_dim={embed_dim}, mlp_depth={mlp_depth}")

                tokenizer = CharNgramTokenizer(vocab_size=5000, ngram=ngram)

                # Build training pairs for this tokenizer
                q_ids, s_ids, labels = build_gate_training_data(
                    df_train, xml_cache, tokenizer
                )
                dataset = GateDataset(q_ids, s_ids, labels)

                # Build and train gate
                gate = build_gate(vocab_size=5000, embed_dim=embed_dim,
                                  mlp_depth=mlp_depth)
                gate = train_gate(gate, dataset, epochs=15, lr=1e-3, device=device)

                # Evaluate
                recall, pos_mean, neg_mean = evaluate_stage1_recall(
                    df_test, xml_cache, gate, tokenizer,
                    keep_top_k=20, device=device
                )
                gap = pos_mean - neg_mean
                print(f"    recall={recall:.1f}%  pos={pos_mean:.3f}  "
                      f"neg={neg_mean:.3f}  gap={gap:.3f}")

                rows.append({
                    "ngram":              ngram,
                    "embed_dim":          embed_dim,
                    "mlp_depth":          mlp_depth,
                    "stage1_recall_pct":  round(recall, 2),
                    "pos_score_mean":     round(pos_mean, 4),
                    "neg_score_mean":     round(neg_mean, 4),
                    "calibration_gap":    round(gap, 4),
                })

    out_path = FIGURES / "exp3_gate_ablations.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  Ablation results saved → {out_path}")

    # Print summary table
    df_out = pd.DataFrame(rows)
    print("\n" + df_out.to_string(index=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_ablations(args.device)
