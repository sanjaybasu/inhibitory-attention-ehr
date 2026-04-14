"""
fill_pending_tables.py — Download, judge, and fill the two pending LaTeX tables.

Usage (after Modal jobs complete):
  python fill_pending_tables.py          # just print numbers
  python fill_pending_tables.py --patch  # also patch paper.tex in place

Steps:
  1. Download exp3_v5_llm_results.csv and exp3_llmlingua2_results.csv from Modal volume
  2. Run LLM-as-judge (oracle) on both
  3. Print table values
  4. If --patch: replace \\textit{pending} cells in paper.tex
"""

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anthropic

FIGURES = Path(__file__).parent.parent / "figures"
TEX = Path(__file__).parent.parent / "neurips2026_submission" / "inhibitory_attention_clitm_ehr_neurips2026.tex"
TSV = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files"
           "/medalign_instructions_v1_3/clinician-reviewed-model-responses.tsv")
VOLUME = "clinical-litm-results"

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


def modal_download(remote_name: str, local_path: Path) -> bool:
    """Download file from Modal volume. Returns True on success."""
    r = subprocess.run(
        ["modal", "volume", "get", VOLUME, remote_name, str(local_path)],
        capture_output=True, text=True)
    if r.returncode == 0:
        print(f"  Downloaded {remote_name} → {local_path}")
        return True
    print(f"  [WARN] Could not download {remote_name}: {r.stderr.strip()}")
    return False


def judge_response(client: anthropic.Anthropic,
                   question: str, evidence: str, response: str) -> float:
    if not response or not evidence or len(str(response).strip()) < 3:
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
        if "YES" in text: return 1.0
        if "NO" in text: return 0.0
        return 0.5
    except Exception as e:
        print(f"    API error: {e}")
        time.sleep(2)
        return float("nan")


def judge_df(df: pd.DataFrame, resp_col: str, judge_col: str,
             client: anthropic.Anthropic) -> pd.DataFrame:
    """Add judge_col to df by judging resp_col. Skips already-judged rows."""
    if judge_col not in df.columns:
        df[judge_col] = float("nan")
    tsv = pd.read_csv(TSV, sep="\t").dropna(subset=["question", "evidence", "filename"])
    tsv_dedup = tsv.drop_duplicates(subset=["filename", "question"])
    if "evidence" not in df.columns:
        df = df.merge(tsv_dedup[["filename", "question", "evidence"]],
                      on=["filename", "question"], how="left")
    total = len(df)
    for i, row in df.iterrows():
        if df.at[i, judge_col] == 1.0 or df.at[i, judge_col] == 0.0:
            continue  # already judged
        df.at[i, judge_col] = judge_response(
            client, str(row["question"]), str(row.get("evidence", "")),
            str(row.get(resp_col, "")))
        if (i + 1) % 10 == 0:
            print(f"    {i+1}/{total}")
    return df


def accuracy(df: pd.DataFrame, judge_col: str, mask=None):
    sub = df[mask] if mask is not None else df
    return (sub[judge_col] == 1.0).mean() * 100


def summarise(df: pd.DataFrame, judge_col: str):
    df["is_mid"] = df["position"].between(0.3, 0.7)
    ov  = accuracy(df, judge_col)
    mid = accuracy(df, judge_col, df["is_mid"])
    edg = accuracy(df, judge_col, ~df["is_mid"])
    return ov, mid, edg


# ── v5 (14B) ─────────────────────────────────────────────────────────────────

def process_v5(client: anthropic.Anthropic, patch: bool):
    csv_path = FIGURES / "exp3_v5_llm_results.csv"
    if not csv_path.exists():
        print("Downloading exp3_v5_llm_results.csv...")
        if not modal_download("exp3_v5_llm_results.csv", csv_path):
            print("  v5 not ready yet.")
            return None

    df = pd.read_csv(csv_path)
    # Sanity check
    non_empty = df["baseline_response"].notna() & (df["baseline_response"].str.strip() != "")
    print(f"v5 CSV: {len(df)} rows, {non_empty.sum()} with non-empty baseline_response")
    if non_empty.sum() == 0:
        print("  All responses empty — v5 experiment failed or still running.")
        return None

    judged_path = FIGURES / "exp3_v5_llm_results_judged.csv"
    if judged_path.exists():
        df_j = pd.read_csv(judged_path)
    else:
        df_j = df.copy()

    arms = {
        "baseline": "baseline_response",
        "bm25":     "bm25_response",
        "bm25_filtered": "bm25_filtered_response",
        "dense":    "dense_response",
        "ce":       "ce_response",
        "qccs":     "qccs_response",
    }

    print("Running oracle judge on v5...")
    for arm, resp_col in arms.items():
        jcol = f"{arm}_judge"
        df_j = judge_df(df_j, resp_col, jcol, client)

    df_j.to_csv(judged_path, index=False)
    print(f"Saved: {judged_path}")

    print("\n=== v5 (14B) Results ===")
    print(f"{'Arm':20s}  {'Overall':>8}  {'Middle':>8}  {'Edge':>8}")
    v5_nums = {}
    for arm in arms:
        jcol = f"{arm}_judge"
        if jcol not in df_j.columns:
            continue
        ov, mid, edg = summarise(df_j, jcol)
        v5_nums[arm] = (ov, mid, edg)
        print(f"  {arm:18s}  {ov:8.1f}%  {mid:8.1f}%  {edg:8.1f}%")

    if patch:
        _patch_v5_table(v5_nums)
    return v5_nums


# ── LLMLingua-2 ───────────────────────────────────────────────────────────────

def process_llmlingua2(client: anthropic.Anthropic, patch: bool):
    csv_path = FIGURES / "exp3_llmlingua2_results.csv"
    if not csv_path.exists():
        print("Downloading exp3_llmlingua2_results.csv...")
        if not modal_download("exp3_llmlingua2_results.csv", csv_path):
            print("  LLMLingua-2 not ready yet.")
            return None

    df = pd.read_csv(csv_path)
    non_empty = df["llmlingua2_response"].notna() & (df["llmlingua2_response"].str.strip() != "")
    print(f"LLMLingua-2 CSV: {len(df)} rows, {non_empty.sum()} with non-empty response")
    if non_empty.sum() == 0:
        print("  All responses empty — LLMLingua-2 experiment failed or still running.")
        return None

    judged_path = FIGURES / "exp3_llmlingua2_results_judged.csv"
    if judged_path.exists():
        df_j = pd.read_csv(judged_path)
    else:
        df_j = df.copy()

    print("Running oracle judge on LLMLingua-2...")
    df_j = judge_df(df_j, "llmlingua2_response", "llmlingua2_judge", client)
    df_j.to_csv(judged_path, index=False)
    print(f"Saved: {judged_path}")

    ov, mid, edg = summarise(df_j, "llmlingua2_judge")
    print(f"\n=== LLMLingua-2 Results ===")
    print(f"  Overall: {ov:.1f}%   Middle: {mid:.1f}%   Edge: {edg:.1f}%")

    if patch:
        _patch_llmlingua2_table(ov, mid, edg)
    return (ov, mid, edg)


# ── LaTeX patching ────────────────────────────────────────────────────────────

def _patch_v5_table(v5_nums: dict):
    """Patch tab:largerreader in paper.tex with actual v5 numbers.
    Arm order in table: Full, BM25, B25f, Dense, CE, MR (7B), QCCS
    v5_nums[arm] = (ov, mid, edg) as returned by summarise().
    """
    import re as _re
    tex = TEX.read_text()

    def fmt(v): return f"{v:.1f}"

    # Row definitions: (row_label_prefix, arm_tuple_idx, mr_7b_value)
    # arm_tuple_idx: 0=overall, 1=middle, 2=edge
    rows = [
        ("Middle (30--70\\%)", 1, 20.0),
        ("Edge",               2, 11.3),
        ("Overall",            0, 14.5),
    ]

    # Anchor search to tab:largerreader to avoid matching other "Overall"/"Edge" rows
    anchor = "\\label{tab:largerreader}"
    anchor_pos = tex.find(anchor)
    if anchor_pos == -1:
        print("  [WARN] Could not find \\label{tab:largerreader}")
        return
    table_end = tex.find("\\end{tabular}", anchor_pos)
    if table_end == -1:
        table_end = anchor_pos + 600
    block = tex[anchor_pos:table_end]

    replaced = 0
    for label, arm_idx, mr_val in rows:
        arm_order = ["baseline", "bm25", "bm25_filtered", "dense", "ce"]
        cells = " & ".join(fmt(v5_nums[a][arm_idx]) for a in arm_order if a in v5_nums)
        qccs = fmt(v5_nums["qccs"][arm_idx]) if "qccs" in v5_nums else r"\textit{pending}"
        new_row_cells = f"& {cells} & {fmt(mr_val)} & {qccs} \\\\"
        # Replace everything after the label (with optional trailing spaces) up to \\
        # Use lambda to avoid regex backslash interpretation in replacement string
        pat = r'(' + _re.escape(label) + r'\s*)&[^\n]+\\\\'
        def _repl(m, _nc=new_row_cells): return m.group(1) + _nc
        new_block, n = _re.subn(pat, _repl, block, count=1)
        if n:
            block = new_block
            replaced += 1
        else:
            print(f"  [WARN] Could not find '{label}' row in tab:largerreader")

    tex = tex[:anchor_pos] + block + tex[table_end:]
    TEX.write_text(tex)
    print(f"Patched {replaced}/3 rows in tab:largerreader ({TEX})")


def _patch_llmlingua2_table(ov, mid, edg):
    """Patch tab:llmlingua2 in paper.tex with actual LLMLingua-2 numbers."""
    import re as _re
    tex = TEX.read_text()
    # Use regex to replace the LLMLingua-2 cell regardless of current value
    tex = _re.sub(
        r'(Middle \(30--70\\%\) & 3\.3 & 20\.0 & )[^\s\\]+( & 16\.7 \\\\)',
        rf'\g<1>{mid:.1f}\g<2>', tex)
    tex = _re.sub(
        r'(Edge\s+& 1\.9 & 11\.3 & )[^\s\\]+( & 30\.2 \\\\)',
        rf'\g<1>{edg:.1f}\g<2>', tex)
    tex = _re.sub(
        r'(Overall\s+& 2\.4 & 14\.5 & )[^\s\\]+( & 25\.3 \\\\)',
        rf'\g<1>{ov:.1f}\g<2>', tex)
    TEX.write_text(tex)
    print(f"Patched tab:llmlingua2 in {TEX}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    patch = "--patch" in sys.argv
    client = anthropic.Anthropic()

    v5 = process_v5(client, patch)
    ll2 = process_llmlingua2(client, patch)

    if v5 is None and ll2 is None:
        print("\nNeither experiment has results yet. Re-run after Modal jobs complete.")
    elif patch:
        print("\nAll available tables patched. Re-run 'latexmk -pdf paper.tex' to rebuild PDF.")


if __name__ == "__main__":
    main()
