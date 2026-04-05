"""
Second LLM judge for Experiment 3 Stage 2 inter-rater validation.
Uses claude-sonnet-4-6 as a second judge on the same 83 instructions,
then computes Cohen's kappa against the primary claude-haiku-4-5-20251001 judgments.

Usage:
  python experiments/exp3_second_judge.py

Input:  figures/exp3_v4_llm_results_judged.csv  (primary Haiku judgments)
Output: figures/exp3_second_judge_results.csv   (Sonnet judgments + kappa)
"""

import re
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
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

ARMS = ["baseline", "bm25", "bm25_filtered", "dense", "ce", "qccs"]


def judge_response(client, question, evidence, response):
    if not response or not evidence or len(response.strip()) < 3:
        return 0.0
    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=10,
            messages=[{"role": "user", "content":
                        JUDGE_PROMPT.format(question=question,
                                            evidence=evidence,
                                            response=response)}]
        )
        text = msg.content[0].text.strip().upper()
        if "YES" in text:
            return 1.0
        if "NO" in text:
            return 0.0
        return 0.5
    except Exception as e:
        print(f"    API error: {e}")
        time.sleep(2)
        return float("nan")


def cohen_kappa(y1, y2):
    """Binary Cohen's kappa treating 1.0=yes, else=no. NaN rows dropped."""
    mask = ~(np.isnan(y1) | np.isnan(y2))
    a = (y1[mask] == 1.0)
    b = (y2[mask] == 1.0)
    n = mask.sum()
    if n == 0:
        return float("nan")
    p_o = (a == b).mean()
    p_e = (a.mean() * b.mean()) + ((~a).mean() * (~b).mean())
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def main():
    in_csv  = FIGURES / "exp3_v4_llm_results_judged.csv"
    out_csv = FIGURES / "exp3_second_judge_results.csv"

    df = pd.read_csv(in_csv)
    print(f"Loaded {len(df)} instructions with primary judge scores")

    client = anthropic.Anthropic()
    total = len(df) * len(ARMS)
    done  = 0

    for arm in ARMS:
        resp_col  = f"{arm}_response"
        judge2_col = f"{arm}_judge2"
        df[judge2_col] = float("nan")

        for i, row in df.iterrows():
            q  = str(row["question"])
            ev = str(row.get("evidence", ""))
            r  = str(row.get(resp_col, ""))
            score = judge_response(client, q, ev, r)
            df.at[i, judge2_col] = score
            done += 1
            if done % 25 == 0:
                print(f"  {done}/{total}  (arm={arm})")

    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Compute per-arm kappa
    print("\n=== Inter-rater Agreement (Cohen's kappa): Haiku vs. Sonnet ===")
    kappas = {}
    for arm in ARMS:
        j1 = df[f"{arm}_judge"].values.astype(float)
        j2 = df[f"{arm}_judge2"].values.astype(float)
        k  = cohen_kappa(j1, j2)
        acc1 = (j1 == 1.0).mean() * 100
        acc2 = (j2 == 1.0).mean() * 100
        kappas[arm] = k
        print(f"  {arm:15s}: Haiku={acc1:.1f}%  Sonnet={acc2:.1f}%  kappa={k:.3f}")

    mean_k = np.nanmean(list(kappas.values()))
    print(f"\n  Mean kappa across {len(ARMS)} arms: {mean_k:.3f}")

    # Summary comparison table for paper
    df["is_mid"] = df["position"].between(0.3, 0.7)
    print("\n=== Middle-band summary: Haiku vs. Sonnet ===")
    mid = df[df["is_mid"]]
    for arm in ARMS:
        j1_mid = (mid[f"{arm}_judge"] == 1.0).mean() * 100
        j2_mid = (mid[f"{arm}_judge2"] == 1.0).mean() * 100
        print(f"  {arm:15s}: Haiku={j1_mid:.1f}%  Sonnet={j2_mid:.1f}%")

    # Save kappa summary CSV
    kappa_rows = [{"arm": arm, "haiku_acc_pct": (df[f"{arm}_judge"]==1.0).mean()*100,
                   "sonnet_acc_pct": (df[f"{arm}_judge2"]==1.0).mean()*100,
                   "kappa": kappas[arm]} for arm in ARMS]
    kdf = pd.DataFrame(kappa_rows)
    kdf.to_csv(FIGURES / "exp3_kappa_summary.csv", index=False)
    print(f"\nKappa summary saved: {FIGURES}/exp3_kappa_summary.csv")


if __name__ == "__main__":
    main()
