"""
Judge extras: BM25 k-sweep and map-reduce Stage-2 accuracy.

After downloading results from Modal:
  modal volume get clinical-litm-results exp3_bm25_k_sweep.csv figures/
  modal volume get clinical-litm-results exp3_mapreduce_results.csv figures/

Usage:
  python experiments/exp3_judge_extras.py --ksweep
  python experiments/exp3_judge_extras.py --mapreduce
  python experiments/exp3_judge_extras.py --ksweep --mapreduce
"""

import re
import time
import argparse
from pathlib import Path
import pandas as pd
import anthropic

FIGURES = Path(__file__).parent.parent / "figures"

JUDGE_PROMPT = """\
You are a medical expert evaluating whether a clinical AI response correctly \
answers a question given the gold-standard evidence extracted from the EHR.

Question: {question}
Gold evidence from EHR: {evidence}
AI response: {response}

Does the AI response CORRECTLY answer the question, given the gold evidence?
Consider the response correct if it conveys the same factual answer as the evidence, \
even if phrased differently. Consider it incorrect if it says "no information" \
when the evidence provides a specific answer, or if it contradicts the evidence.

Answer with exactly one word: YES or NO"""


def judge_one(client: anthropic.Anthropic, question: str,
              evidence: str, response: str) -> float:
    if not response or len(response.strip()) < 3:
        return 0.0
    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content":
                        JUDGE_PROMPT.format(question=question,
                                            evidence=evidence,
                                            response=response)}]
        )
        text = msg.content[0].text.strip().upper()
        return 1.0 if "YES" in text else 0.0
    except Exception as e:
        print(f"  API error: {e}")
        time.sleep(2)
        return float("nan")


def judge_ksweep():
    """Judge BM25 k=1,3,5 responses; save judged CSV and print accuracy table."""
    in_csv = FIGURES / "exp3_bm25_k_sweep.csv"
    if not in_csv.exists():
        print(f"ERROR: {in_csv} not found. Download from Modal first.")
        return

    df = pd.read_csv(in_csv)
    client = anthropic.Anthropic()
    K_VALUES = [1, 3, 5]

    # Judge each k-value
    for k in K_VALUES:
        col_r = f"bm25_k{k}_response"
        col_j = f"bm25_k{k}_judge"
        if col_r not in df.columns:
            print(f"WARNING: column {col_r} not found; skipping k={k}")
            continue
        df[col_j] = float("nan")
        print(f"Judging k={k} ({len(df)} rows)...")
        for i, row in df.iterrows():
            df.at[i, col_j] = judge_one(
                client,
                str(row["question"]),
                str(row.get("evidence", "")),
                str(row.get(col_r, "")),
            )

    out = FIGURES / "exp3_bm25_k_sweep_judged.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Print accuracy table
    print("\n--- BM25 k-sweep accuracy ---")
    print(f"{'k':>3}  {'Overall (%)':>12}  {'Middle 30-70% (%)':>18}")
    for k in K_VALUES:
        col_j = f"bm25_k{k}_judge"
        if col_j not in df.columns:
            continue
        acc_all = (df[col_j] == 1.0).mean() * 100
        mid = df[(df["position"] >= 0.30) & (df["position"] <= 0.70)]
        acc_mid = (mid[col_j] == 1.0).mean() * 100 if len(mid) > 0 else float("nan")
        n_mid = len(mid)
        print(f"{k:>3}  {acc_all:>12.1f}  {acc_mid:>18.1f}  (n_mid={n_mid})")

    # k=20 baseline for comparison
    print(f"{'20':>3}  {'2.4':>12}  {'3.3':>18}  (main paper)")


def bootstrap_ci(values, n_boot=5000, seed=42, ci=95):
    """Compute bootstrap CI for mean of binary values."""
    import numpy as np
    rng = np.random.default_rng(seed)
    arr = pd.array(values, dtype=float)
    arr = arr[~pd.isna(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    means = [rng.choice(arr, size=len(arr), replace=True).mean() * 100
             for _ in range(n_boot)]
    lo = (100 - ci) / 2
    return (float(np.mean(arr)) * 100,
            float(np.percentile(means, lo)),
            float(np.percentile(means, 100 - lo)))


def judge_mapreduce():
    """Judge map-reduce responses; save judged CSV and print position-band table."""
    in_csv = FIGURES / "exp3_mapreduce_results.csv"
    if not in_csv.exists():
        print(f"ERROR: {in_csv} not found. Download from Modal first.")
        return

    df = pd.read_csv(in_csv)
    client = anthropic.Anthropic()

    print(f"Judging {len(df)} map-reduce responses...")
    df["mapreduce_judge"] = float("nan")
    for i, row in df.iterrows():
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(df)}")
        df.at[i, "mapreduce_judge"] = judge_one(
            client,
            str(row["question"]),
            str(row.get("evidence", "")),
            str(row.get("mapreduce_response", "")),
        )

    out = FIGURES / "exp3_mapreduce_judged.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Print position-band accuracy table (matches Tab. llm bands)
    bands = [
        ("0--10\\%",   0.00, 0.10),
        ("10--30\\%",  0.10, 0.30),
        ("30--50\\%",  0.30, 0.50),
        ("50--70\\%",  0.50, 0.70),
        ("70--90\\%",  0.70, 0.90),
        ("90--100\\%", 0.90, 1.01),
    ]

    print("\n--- Map-reduce accuracy by position band ---")
    print(f"{'Band':<12}  {'n':>4}  {'Acc (%)':>8}")
    for label, lo, hi in bands:
        sub = df[(df["position"] >= lo) & (df["position"] < hi)]
        if len(sub) == 0:
            print(f"{label:<12}  {0:>4}  {'  N/A':>8}")
            continue
        acc = (sub["mapreduce_judge"] == 1.0).mean() * 100
        print(f"{label:<12}  {len(sub):>4}  {acc:>8.1f}")

    # Middle/edge with bootstrap CIs
    mid = df[(df["position"] >= 0.30) & (df["position"] <= 0.70)]
    edge = df[(df["position"] < 0.30) | (df["position"] > 0.70)]
    overall = df

    acc_mid, ci_mid_lo, ci_mid_hi = bootstrap_ci(mid["mapreduce_judge"].tolist())
    acc_edge, ci_edge_lo, ci_edge_hi = bootstrap_ci(edge["mapreduce_judge"].tolist())
    acc_all, ci_all_lo, ci_all_hi = bootstrap_ci(overall["mapreduce_judge"].tolist())

    print(f"\nMiddle (30--70\\%) n={len(mid)}: {acc_mid:.1f}% [{ci_mid_lo:.0f},{ci_mid_hi:.0f}]")
    print(f"Edge              n={len(edge)}: {acc_edge:.1f}% [{ci_edge_lo:.0f},{ci_edge_hi:.0f}]")
    print(f"Overall           n={len(overall)}: {acc_all:.1f}% [{ci_all_lo:.0f},{ci_all_hi:.0f}]")

    print("\nFor Tab. llm (replace \\textit{p} in MR column):")
    for label, lo, hi in bands:
        sub = df[(df["position"] >= lo) & (df["position"] < hi)]
        acc = (sub["mapreduce_judge"] == 1.0).mean() * 100 if len(sub) > 0 else 0.0
        print(f"  {label}: {acc:.1f}")
    print(f"  Middle: {acc_mid:.1f}\\,[{ci_mid_lo:.0f},{ci_mid_hi:.0f}]")
    print(f"  Edge: {acc_edge:.1f}\\,[{ci_edge_lo:.0f},{ci_edge_hi:.0f}]")
    print(f"  Overall: {acc_all:.1f}\\,[{ci_all_lo:.0f},{ci_all_hi:.0f}]")


def main():
    parser = argparse.ArgumentParser(description="Judge BM25 k-sweep and map-reduce results")
    parser.add_argument("--ksweep", action="store_true", help="Judge BM25 k-sweep")
    parser.add_argument("--mapreduce", action="store_true", help="Judge map-reduce")
    args = parser.parse_args()

    if args.ksweep:
        judge_ksweep()
    if args.mapreduce:
        judge_mapreduce()
    if not args.ksweep and not args.mapreduce:
        print("Usage: python exp3_judge_extras.py [--ksweep] [--mapreduce]")


if __name__ == "__main__":
    main()
