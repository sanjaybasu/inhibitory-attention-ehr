"""
Oracle Control Experiment (Experiment 3, Stage 2 supplement).
For each instruction in the held-out test split:
  1. Identify the FIRST sentence in the EHR that contains a gold content word
     (the "oracle gold sentence").
  2. Build the oracle context: just that sentence + last-5 recent events.
  3. Run Qwen2.5-7B-Instruct on the oracle context via Modal A100.
  4. Judge with Claude Haiku.

Rationale: If Stage-2 accuracy with the oracle context is also ~2-3%,
it confirms the Qwen reader itself has limited capacity for this task.
If oracle accuracy is substantially higher (>20%), it confirms that
Stage-2 accuracy is sensitive to context quality, not just gold presence.

This script runs LOCAL CPU parts; Modal GPU call is handled in modal_app.py
via run_oracle_inference.

Usage:
  # Step 1: build oracle contexts (local, CPU)
  python experiments/exp3_oracle_control.py --build

  # Step 2: run inference on Modal (GPU)
  modal run experiments/modal_app.py::run_oracle_inference

  # Step 3: judge and report (local, CPU)
  python experiments/exp3_oracle_control.py --judge
"""

import re
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import anthropic

sys.path.insert(0, str(Path(__file__).parent))
from exp3_qccs_gate import parse_ehr_sentences

BASE    = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
TSV     = BASE / "medalign_instructions_v1_3" / "clinician-reviewed-model-responses.tsv"
EHR_DIR = BASE / "medalign_instructions_v1_3" / "ehrs"
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


def retrieval_hit(sentence_text: str, evidence: str) -> bool:
    ev_words = {w for w in re.findall(r'\w+', evidence.lower()) if len(w) >= 4}
    sent_words = set(re.findall(r'\w+', sentence_text.lower()))
    return bool(ev_words & sent_words)


def find_gold_sentence(events: list[dict], evidence: str) -> dict | None:
    """Return first event whose text matches the gold evidence, else None."""
    for ev in events:
        if retrieval_hit(ev["text"], evidence):
            return ev
    return None


def build_oracle_contexts():
    """
    For each test instruction, find the gold sentence and build oracle context.
    Saves: figures/exp3_oracle_contexts.csv
    """
    print("Loading MedAlign test split...")
    df = pd.read_csv(TSV, sep="\t").dropna(subset=["question","evidence","filename"])
    patients = df["filename"].unique()
    _, test_patients = train_test_split(patients, test_size=0.30, random_state=42)
    df_test = df[df["filename"].isin(set(test_patients))].drop_duplicates(
        subset=["filename","question"])
    print(f"Test: {len(test_patients)} patients, {len(df_test)} instructions")

    xml_cache = {}
    rows = []
    n_found = 0

    for _, row in df_test.iterrows():
        fname    = str(row["filename"])
        question = str(row["question"])
        evidence = str(row["evidence"]).strip()
        if not evidence:
            continue

        if fname not in xml_cache:
            p = EHR_DIR / fname
            xml_cache[fname] = parse_ehr_sentences(p) if p.exists() else []
        events = xml_cache[fname]
        if not events:
            continue

        gold_ev = find_gold_sentence(events, evidence)
        has_gold = gold_ev is not None

        # Oracle context: gold sentence (if found) + last-5 recent events
        recent = events[-5:]
        recent_texts = [e["text"] for e in recent]
        gold_text = gold_ev["text"] if gold_ev else ""

        oracle_parts = []
        if gold_ev:
            oracle_parts.append(f"[Gold evidence] {gold_text}")
        for rt in recent_texts:
            if rt != gold_text:  # avoid dup if gold is also recent
                oracle_parts.append(rt)
        oracle_context = "\n".join(oracle_parts)

        if has_gold:
            n_found += 1

        rows.append({
            "filename":       fname,
            "question":       question,
            "evidence":       evidence,
            "has_gold":       has_gold,
            "gold_text":      gold_text,
            "oracle_context": oracle_context,
        })

    result = pd.DataFrame(rows)
    out = FIGURES / "exp3_oracle_contexts.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved: {out}  ({n_found}/{len(result)} instructions have gold sentence)")
    print("Next: modal run experiments/modal_app.py::run_oracle_inference")
    return result


def judge_oracle_responses():
    """
    Judge Oracle + arm responses from exp3_oracle_inference.csv.
    Saves: figures/exp3_oracle_judged.csv
    """
    in_csv = FIGURES / "exp3_oracle_inference.csv"
    if not in_csv.exists():
        print(f"ERROR: {in_csv} not found. Run Modal inference first.")
        return

    df = pd.read_csv(in_csv)
    client = anthropic.Anthropic()
    judge_col = "oracle_judge"
    df[judge_col] = float("nan")

    print(f"Judging {len(df)} oracle responses...")
    for i, row in df.iterrows():
        q  = str(row["question"])
        ev = str(row["evidence"])
        r  = str(row.get("oracle_response", ""))
        if not r or len(r.strip()) < 3:
            df.at[i, judge_col] = 0.0
            continue
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content":
                            JUDGE_PROMPT.format(question=q, evidence=ev, response=r)}]
            )
            text = msg.content[0].text.strip().upper()
            df.at[i, judge_col] = 1.0 if "YES" in text else 0.0
        except Exception as e:
            print(f"  API error: {e}")
            time.sleep(2)

    out = FIGURES / "exp3_oracle_judged.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary
    has_gold = df[df["has_gold"] == True]
    acc_all = (df[judge_col] == 1.0).mean() * 100
    acc_gold = (has_gold[judge_col] == 1.0).mean() * 100 if len(has_gold) > 0 else float("nan")
    print(f"\nOracle accuracy (all N={len(df)}): {acc_all:.1f}%")
    print(f"Oracle accuracy (has gold, N={len(has_gold)}): {acc_gold:.1f}%")
    print("\nCompare to 6-arm Stage-2 results:")
    print("  QCCS: 25.3%  BM25: 2.4%  Dense: 2.4%  CE: 1.2%  Full: 3.6%")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true", help="Build oracle contexts CSV")
    parser.add_argument("--judge", action="store_true", help="Judge oracle inference results")
    args = parser.parse_args()

    if args.build:
        build_oracle_contexts()
    elif args.judge:
        judge_oracle_responses()
    else:
        # Default: build only
        build_oracle_contexts()


if __name__ == "__main__":
    main()
