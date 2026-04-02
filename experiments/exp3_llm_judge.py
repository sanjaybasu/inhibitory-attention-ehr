"""
LLM-as-Judge evaluation for Experiment 3 Stage 2.
Uses Claude claude-haiku-4-5-20251001 to semantically judge whether each response
correctly answers the clinical question given the gold evidence string.

Run:
  python exp3_llm_judge.py          # judges v2 (3 arms: baseline, bm25, qccs)
  python exp3_llm_judge.py --v3     # judges v3 (5 arms: + bm25_filtered, dense)

Inputs:
  figures/exp3_v2_llm_results.csv  (83 × 3 responses)
  figures/exp3_v3_llm_results.csv  (83 × 5 responses)
Outputs:
  figures/exp3_v2_llm_results_judged.csv
  figures/exp3_v3_llm_results_judged.csv
"""

import re
import sys
import time
from pathlib import Path
import pandas as pd
import anthropic

FIGURES = Path(__file__).parent.parent / "figures"
TSV = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
           "/medalign_instructions_v1_3/clinician-reviewed-model-responses.tsv")

V3_MODE = "--v3" in sys.argv  # judge v3 (5 arms) when flag is set

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


def judge_response(client: anthropic.Anthropic,
                   question: str, evidence: str, response: str) -> float:
    """Return 1.0 if judge says YES, 0.0 if NO, 0.5 if uncertain."""
    if not response or not evidence or len(response.strip()) < 3:
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
        if "YES" in text:
            return 1.0
        if "NO" in text:
            return 0.0
        return 0.5  # uncertain
    except Exception as e:
        print(f"    API error: {e}")
        time.sleep(2)
        return float("nan")


def main():
    import numpy as np

    if V3_MODE:
        in_csv  = FIGURES / "exp3_v3_llm_results.csv"
        out_csv = FIGURES / "exp3_v3_llm_results_judged.csv"
        arms = {
            "baseline":     "baseline_response",
            "bm25":         "bm25_response",
            "bm25_filtered":"bm25_filtered_response",
            "dense":        "dense_response",
            "qccs":         "qccs_response",
        }
        tok_cols = {arm: f"{arm}_correct" for arm in arms}
    else:
        in_csv  = FIGURES / "exp3_v2_llm_results.csv"
        out_csv = FIGURES / "exp3_v2_llm_results_judged.csv"
        arms = {
            "baseline": "baseline_response",
            "bm25":     "bm25_response",
            "qccs":     "qccs_response",
        }
        # v2 judged file uses *_tok columns for legacy reasons
        tok_cols = {arm: f"{arm}_tok" for arm in arms}

    df = pd.read_csv(in_csv)

    # Attach evidence strings from TSV
    tsv = pd.read_csv(TSV, sep="\t").dropna(subset=["question", "evidence", "filename"])
    tsv_dedup = tsv.drop_duplicates(subset=["filename", "question"])
    df = df.merge(tsv_dedup[["filename", "question", "evidence"]],
                  on=["filename", "question"], how="left")

    # Rename *_correct → *_tok for uniform token-overlap column naming
    for arm in arms:
        src = f"{arm}_correct"
        if src in df.columns and tok_cols[arm] not in df.columns:
            df[tok_cols[arm]] = df[src]

    client = anthropic.Anthropic()
    total = len(df) * len(arms)
    done  = 0
    for arm, resp_col in arms.items():
        judge_col = f"{arm}_judge"
        df[judge_col] = float("nan")
        for i, row in df.iterrows():
            q  = str(row["question"])
            ev = str(row.get("evidence", ""))
            r  = str(row.get(resp_col, ""))
            score = judge_response(client, q, ev, r)
            df.at[i, judge_col] = score
            done += 1
            if done % 10 == 0:
                print(f"  {done}/{total}  ({arm})")

    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # Summary
    df["is_mid"] = df["position"].between(0.3, 0.7)
    print("\n=== LLM-as-Judge Results (judge=1.0 → correct) ===")
    for label, mask in [("Overall", slice(None)),
                        ("Middle (30-70%)", df["is_mid"]),
                        ("Edge", ~df["is_mid"])]:
        sub = df[mask] if isinstance(mask, pd.Series) else df
        n = len(sub)
        print(f"\n{label} (N={n}):")
        for arm in arms:
            jcol = f"{arm}_judge"
            tcol = tok_cols[arm]
            j_pct = (sub[jcol] == 1.0).mean() * 100 if jcol in sub.columns else float("nan")
            t_pct = (sub[tcol] > 0).mean() * 100 if tcol in sub.columns else float("nan")
            print(f"  {arm}: judge={j_pct:.1f}%  tok={t_pct:.1f}%")

    # Bootstrap Wilson-score 95% CIs for judge middle
    np.random.seed(42)
    mid = df[df["is_mid"]]
    print("\nBootstrap 95% CIs (middle, N={}, 5000 resamples):".format(len(mid)))
    for arm in arms:
        jcol = f"{arm}_judge"
        if jcol in mid.columns:
            vals = (mid[jcol] == 1.0).astype(float).values
            boots = [np.mean(np.random.choice(vals, len(vals), replace=True))
                     for _ in range(5000)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            print(f"  {arm}: {vals.mean()*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]")


if __name__ == "__main__":
    main()
