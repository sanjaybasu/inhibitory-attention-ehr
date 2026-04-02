"""
Experiment 3 Extended Baselines (Stage 1 retrieval only, runs on CPU):
  1. Dense retrieval (sentence-transformers/all-MiniLM-L6-v2)
  2. BM25 + section-header filtering
  3. Dense + QCCS ensemble

Run: python exp3_extended_baselines.py

Outputs (appended to figures/):
  exp3_extended_stage1.csv  — per-instruction recall for all methods
  exp3_bm25_filtered.csv    — filtered BM25 results
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent))
from exp3_qccs_gate import parse_ehr_sentences, QCCSGate, CharNgramTokenizer, qccs_compress
import torch

FIGURES = Path(__file__).parent.parent / "figures"
BASE = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
TSV  = BASE / "medalign_instructions_v1_3" / "clinician-reviewed-model-responses.tsv"
EHR_DIR = BASE / "medalign_instructions_v1_3" / "ehrs"

# ── Section-header filter patterns ──────────────────────────────────────────
# Sentences that BM25 false-positively retrieves: structured Q/A headers,
# counseling templates, social-history headers, review-of-systems headers.
HEADER_PATTERNS = [
    re.compile(r'^\s*question\s*[:\d]', re.I),
    re.compile(r'^\s*answer\s*[:\d]', re.I),
    re.compile(r'^\s*counseling\s*[:\d]', re.I),
    re.compile(r'^\s*follow.?up\s*[:\d]', re.I),
    re.compile(r'^\s*review\s+of\s+systems?\s*[:\d]?', re.I),
    re.compile(r'^\s*chief\s+complaint\s*[:\d]?', re.I),
    re.compile(r'^\s*assessment\s*[:\d]', re.I),
    re.compile(r'^\s*plan\s*[:\d]', re.I),
    re.compile(r'^\s*subjective\s*[:\d]?', re.I),
    re.compile(r'^\s*objective\s*[:\d]?', re.I),
    re.compile(r'^\s*disposition\s*[:\d]?', re.I),
]

def is_header(text: str) -> bool:
    """True if the sentence looks like a section header or Q/A template."""
    t = text.strip()
    if len(t) < 5:
        return True
    for pat in HEADER_PATTERNS:
        if pat.match(t):
            return True
    # Also filter lines that are entirely capitalised section titles ≤ 40 chars
    if len(t) <= 40 and t.upper() == t and re.match(r'^[A-Z\s/:-]+$', t):
        return True
    return False


def retrieval_hit(sentence_text: str, evidence: str) -> bool:
    """Binary hit: does this sentence contain a content word from evidence?"""
    ev_words = {w for w in re.findall(r'\w+', evidence.lower()) if len(w) >= 4}
    sent_words = set(re.findall(r'\w+', sentence_text.lower()))
    return bool(ev_words & sent_words)


def bm25_recall(events: list[dict], query: str, evidence: str,
                k: int = 20, filter_headers: bool = False) -> float:
    """Return 1.0 if BM25 top-k contains a hit for evidence, else 0.0."""
    if filter_headers:
        events = [e for e in events if not is_header(e["text"])]
    if not events:
        return 0.0
    corpus = [re.findall(r'\w+', e["text"].lower()) for e in events]
    if not any(corpus):
        return 0.0
    bm25 = BM25Okapi(corpus)
    q_toks = re.findall(r'\w+', query.lower())
    scores = bm25.get_scores(q_toks) if q_toks else [0.0] * len(events)
    recent_idx = {e["idx"] for e in events[-5:]}
    scored = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep = set(scored[:k]) | {i for i, e in enumerate(events) if e["idx"] in recent_idx}
    kept_texts = [events[i]["text"] for i in range(len(events)) if i in keep]
    return float(any(retrieval_hit(t, evidence) for t in kept_texts))


def dense_recall(events: list[dict], query: str, evidence: str,
                 model: SentenceTransformer, k: int = 20,
                 filter_headers: bool = False) -> float:
    """Return 1.0 if dense-retrieval top-k contains a hit for evidence."""
    if filter_headers:
        events = [e for e in events if not is_header(e["text"])]
    if not events:
        return 0.0
    texts = [e["text"] for e in events]
    q_emb   = model.encode([query],  convert_to_numpy=True, normalize_embeddings=True)
    ev_embs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                           batch_size=128, show_progress_bar=False)
    sims = (ev_embs @ q_emb.T).ravel()
    recent_idx = {e["idx"] for e in events[-5:]}
    top_idxs = set(np.argsort(sims)[-k:])
    top_idxs |= {i for i, e in enumerate(events) if e["idx"] in recent_idx}
    kept_texts = [events[i]["text"] for i in range(len(events)) if i in top_idxs]
    return float(any(retrieval_hit(t, evidence) for t in kept_texts))


def qccs_recall_fn(events: list[dict], query: str, evidence: str,
                   gate: QCCSGate, tok: CharNgramTokenizer, k: int = 20) -> float:
    # scores is a list[float] aligned with events
    ctx, scores = qccs_compress(events, query, gate, tok, keep_top_k=k)
    recent_idx = {ev["idx"] for ev in events[-5:]}
    scored = sorted(range(len(scores)), key=lambda i: -scores[i])
    keep_positions = set(scored[:k])
    kept_texts = [events[i]["text"] for i in range(len(events))
                  if i in keep_positions or events[i]["idx"] in recent_idx]
    return float(any(retrieval_hit(t, evidence) for t in kept_texts))


def main():
    print("Loading MedAlign test split...")
    df = pd.read_csv(TSV, sep="\t").dropna(subset=["question", "evidence", "filename"])
    patients = df["filename"].unique()
    _, test_patients = train_test_split(patients, test_size=0.30, random_state=42)
    df_test = df[df["filename"].isin(set(test_patients))].drop_duplicates(
        subset=["filename", "question"])
    print(f"Test split: {len(test_patients)} patients, {len(df_test)} unique instructions")

    # Load position labels
    pos_path = FIGURES / "exp1_litm_results.csv"
    pos_df = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame()

    # Load QCCS gate
    gate_path = FIGURES / "qccs_gate.pt"
    gate = QCCSGate()
    state = torch.load(str(gate_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        gate.load_state_dict(state["state_dict"])
    else:
        gate.load_state_dict(state)
    gate.eval()
    tok = CharNgramTokenizer()

    # Load dense encoder
    print("Loading sentence encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    xml_cache = {}
    rows = []

    for i, (_, row) in enumerate(df_test.iterrows()):
        fname    = str(row["filename"])
        question = str(row["question"])
        evidence = str(row["evidence"]).strip()
        if not evidence:
            continue

        # Get EHR events
        if fname not in xml_cache:
            p = EHR_DIR / fname
            xml_cache[fname] = parse_ehr_sentences(p) if p.exists() else []
        events = xml_cache[fname]
        if not events:
            continue

        # Position
        pos_row = pos_df[(pos_df["filename"] == fname) &
                         (pos_df["question"] == question)] if not pos_df.empty else pd.DataFrame()
        pos = float(pos_row["position"].iloc[0]) if len(pos_row) > 0 else float("nan")

        r_bm25     = bm25_recall(events, question, evidence, k=20, filter_headers=False)
        r_bm25_f   = bm25_recall(events, question, evidence, k=20, filter_headers=True)
        r_dense    = dense_recall(events, question, evidence, encoder, k=20, filter_headers=False)
        r_dense_f  = dense_recall(events, question, evidence, encoder, k=20, filter_headers=True)
        r_qccs     = qccs_recall_fn(events, question, evidence, gate, tok, k=20)

        rows.append({"filename": fname, "question": question[:80],
                     "position": pos,
                     "bm25": r_bm25, "bm25_filtered": r_bm25_f,
                     "dense": r_dense, "dense_filtered": r_dense_f,
                     "qccs": r_qccs})

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(df_test)}")

    result = pd.DataFrame(rows)
    out = FIGURES / "exp3_extended_stage1.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary by position band
    result["is_mid"] = result["position"].between(0.3, 0.7)
    for label, mask in [("Overall", slice(None)),
                        ("Middle (30-70%)", result["is_mid"]),
                        ("Edge", ~result["is_mid"])]:
        sub = result[mask] if isinstance(mask, pd.Series) else result
        n = len(sub)
        print(f"\n{label} (N={n}):")
        for m in ["bm25", "bm25_filtered", "dense", "dense_filtered", "qccs"]:
            if m in sub.columns:
                print(f"  {m}: {sub[m].mean()*100:.1f}%")

    # Ablation at multiple k values
    k_rows = []
    for k in [5, 10, 15, 20]:
        for _, row in df_test.iterrows():
            fname    = str(row["filename"])
            question = str(row["question"])
            evidence = str(row["evidence"]).strip()
            if not evidence:
                continue
            events = xml_cache.get(fname, [])
            if not events:
                continue
            pos_row = pos_df[(pos_df["filename"] == fname) &
                             (pos_df["question"] == question)] if not pos_df.empty else pd.DataFrame()
            pos = float(pos_row["position"].iloc[0]) if len(pos_row) > 0 else float("nan")
            k_rows.append({"k": k, "position": pos,
                           "bm25": bm25_recall(events, question, evidence, k=k),
                           "bm25_filtered": bm25_recall(events, question, evidence, k=k, filter_headers=True),
                           "dense": dense_recall(events, question, evidence, encoder, k=k),
                           "qccs": qccs_recall_fn(events, question, evidence, gate, tok, k=k)})

    k_df = pd.DataFrame(k_rows)
    k_out = FIGURES / "exp3_extended_ablation_k.csv"
    k_df.to_csv(k_out, index=False)
    print(f"\nSaved k-ablation: {k_out}")
    print("\nRecall by k:")
    for k in [5, 10, 15, 20]:
        sub = k_df[k_df["k"] == k]
        mid_sub = sub[sub["position"].between(0.3, 0.7)]
        print(f"  k={k}: BM25={sub['bm25'].mean()*100:.1f}%  BM25-filt={sub['bm25_filtered'].mean()*100:.1f}%  "
              f"Dense={sub['dense'].mean()*100:.1f}%  QCCS={sub['qccs'].mean()*100:.1f}%  "
              f"[mid: BM25={mid_sub['bm25'].mean()*100:.1f}% BM25-f={mid_sub['bm25_filtered'].mean()*100:.1f}% "
              f"Dense={mid_sub['dense'].mean()*100:.1f}% QCCS={mid_sub['qccs'].mean()*100:.1f}%]")


if __name__ == "__main__":
    main()
