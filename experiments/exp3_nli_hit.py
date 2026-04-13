"""
Semantic NLI Hit Criterion for Experiment 3 Stage 1.

Addresses reviewer concern: the lexical-overlap Stage 1 "hit" metric may
mischaracterize true support (partial overlap in distractors; supporting
evidence with different phrasing). This script implements a semantic
entailment criterion using a cross-encoder NLI model (DeBERTa-v3-small)
as an independent validation of the conditional accuracy analysis.

For each of the 83 test instructions, for each retrieval arm:
  - Reconstruct the retained sentences using the same logic as each arm
  - Score entailment between evidence string and each retained sentence
    (bidirectional: evidence→sentence AND sentence→evidence)
  - "Semantic hit" = any retained sentence where max(p_ent_fw, p_ent_bw) > 0.5

Outputs:
  figures/exp3_nli_hits.csv  — per-instruction semantic hit flags per arm
  (printed summary: semantic recall by arm, comparison to lexical recall)

Usage:
  python experiments/exp3_nli_hit.py

Runtime: ~30-60 minutes on CPU (83 instructions × ~250 sentences × 2-way NLI).
"""

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(Path(__file__).parent))
from exp3_qccs_gate import (parse_ehr_sentences, qccs_compress,
                             QCCSGate, CharNgramTokenizer)
import torch

BASE    = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
TSV     = BASE / "medalign_instructions_v1_3" / "clinician-reviewed-model-responses.tsv"
EHR_DIR = BASE / "medalign_instructions_v1_3" / "ehrs"
FIGURES = Path(__file__).parent.parent / "figures"

# ── NLI model ────────────────────────────────────────────────────────────────
# Outputs 3 classes: [contradiction, entailment, neutral]
NLI_MODEL_ID = "cross-encoder/nli-deberta-v3-small"
ENTAIL_IDX   = 1   # class index for "entailment" in deberta-v3-small
ENTAIL_THRESH = 0.5


def lexical_hit(sentence_text: str, evidence: str) -> bool:
    """Existing lexical criterion: ≥1 content word (≥4 chars) shared."""
    ev_words = {w for w in re.findall(r'\w+', evidence.lower()) if len(w) >= 4}
    sent_words = set(re.findall(r'\w+', sentence_text.lower()))
    return bool(ev_words & sent_words)


def semantic_hit(nli_model: CrossEncoder,
                 sentence_text: str,
                 evidence: str) -> bool:
    """
    Semantic entailment criterion (bidirectional).
    Returns True if NLI entailment score > ENTAIL_THRESH in either direction.
    """
    if not sentence_text.strip() or not evidence.strip():
        return False
    # Truncate very long sentences to 512 chars (cross-encoder max_length=512 tokens)
    s = sentence_text[:512]
    e = evidence[:512]
    pairs = [(e, s), (s, e)]  # evidence→sentence, sentence→evidence
    scores = nli_model.predict(pairs, apply_softmax=True)  # shape: (2, 3)
    p_ent_fw = float(scores[0][ENTAIL_IDX])  # P(evidence entails sentence)
    p_ent_bw = float(scores[1][ENTAIL_IDX])  # P(sentence entails evidence)
    return max(p_ent_fw, p_ent_bw) > ENTAIL_THRESH


# ── Arm-specific context reconstructors ──────────────────────────────────────

def get_retained_bm25(events: list[dict], query: str, k: int = 20,
                      filter_headers: bool = False) -> list[str]:
    from rank_bm25 import BM25Okapi
    _HEADER_PATS = [
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
    def _is_header(t):
        t = t.strip()
        if len(t) < 5: return True
        for pat in _HEADER_PATS:
            if pat.match(t): return True
        if len(t) <= 40 and t.upper() == t and re.match(r'^[A-Z\s/:-]+$', t): return True
        return False

    ev = [e for e in events if not _is_header(e["text"])] if filter_headers else events
    if not ev:
        ev = events
    recent_set = {e["idx"] for e in events[-5:]}
    corpus = [re.findall(r'\w+', e["text"].lower()) for e in ev]
    if not any(corpus):
        return [e["text"] for e in events[-k:]]
    bm = BM25Okapi(corpus)
    q_toks = re.findall(r'\w+', query.lower())
    scores = bm.get_scores(q_toks) if q_toks else [0.0] * len(ev)
    top_pos = set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])
    kept = [ev[i]["text"] for i in range(len(ev)) if i in top_pos]
    for e in events:
        if e["idx"] in recent_set and e["text"] not in kept:
            kept.append(e["text"])
    return kept


def get_retained_dense(events: list[dict], query: str, encoder,
                        k: int = 20) -> list[str]:
    texts = [e["text"] for e in events]
    q_emb   = encoder.encode([query], normalize_embeddings=True)
    ev_embs = encoder.encode(texts, normalize_embeddings=True, batch_size=64,
                              show_progress_bar=False)
    sims = (ev_embs @ q_emb.T).ravel()
    recent_set = {e["idx"] for e in events[-5:]}
    top_pos = set(int(i) for i in np.argsort(sims)[-k:])
    top_pos |= {i for i, e in enumerate(events) if e["idx"] in recent_set}
    return [events[i]["text"] for i in range(len(events)) if i in top_pos]


def get_retained_ce(events: list[dict], query: str, ce_model,
                    k: int = 20, first_k: int = 50) -> list[str]:
    from rank_bm25 import BM25Okapi
    recent_set = {e["idx"] for e in events[-5:]}
    corpus = [re.findall(r'\w+', e["text"].lower()) for e in events]
    bm = BM25Okapi(corpus) if any(corpus) else None
    if bm is None:
        return [e["text"] for e in events[-k:]]
    q_toks = re.findall(r'\w+', query.lower())
    bm_scores = bm.get_scores(q_toks) if q_toks else [0.0] * len(events)
    top_bm25 = sorted(range(len(bm_scores)), key=lambda i: -bm_scores[i])[:first_k]
    pairs = [(query, events[i]["text"]) for i in top_bm25]
    ce_scores = ce_model.predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(ce_scores, top_bm25), key=lambda x: -x[0])
    top_pos = {i for _, i in ranked[:k]}
    top_pos |= {i for i, e in enumerate(events) if e["idx"] in recent_set}
    return [events[i]["text"] for i in range(len(events)) if i in top_pos]


def get_retained_qccs(events: list[dict], query: str, gate, tok,
                       k: int = 20) -> list[str]:
    ctx, scores = qccs_compress(events, query, gate, tok, keep_top_k=k)
    recent_set = {e["idx"] for e in events[-5:]}
    top_pos = set(sorted(range(len(scores)), key=lambda i: -scores[i])[:k])
    top_pos |= {i for i, e in enumerate(events) if e["idx"] in recent_set}
    return [events[i]["text"] for i in range(len(events)) if i in top_pos]


def main():
    print("Loading MedAlign test split...")
    df = pd.read_csv(TSV, sep="\t").dropna(subset=["question", "evidence", "filename"])
    patients = df["filename"].unique()
    _, test_patients = train_test_split(patients, test_size=0.30, random_state=42)
    df_test = (df[df["filename"].isin(set(test_patients))]
               .drop_duplicates(subset=["filename", "question"]))
    print(f"Test: {len(test_patients)} patients, {len(df_test)} instructions")

    # Load existing lexical Stage 1 recall for comparison
    lex_path = FIGURES / "exp3_extended_stage1.csv"
    lex_df = pd.read_csv(lex_path) if lex_path.exists() else pd.DataFrame()

    # Load models
    print("Loading NLI cross-encoder (nli-deberta-v3-small, CPU)...")
    nli_model = CrossEncoder(NLI_MODEL_ID, max_length=512, device="cpu")

    print("Loading dense encoder (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading CE reranker (ms-marco-MiniLM-L-6-v2)...")
    ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)

    print("Loading QCCS gate...")
    gate_path = FIGURES / "qccs_gate.pt"
    gate = QCCSGate()
    state = torch.load(str(gate_path), map_location="cpu", weights_only=False)
    gate.load_state_dict(state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state)
    gate.eval()
    tok = CharNgramTokenizer()

    xml_cache: dict = {}
    rows = []
    t0 = time.time()

    for i, (_, row) in enumerate(df_test.iterrows()):
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

        # Get retained sentences for each arm
        texts_bm25  = get_retained_bm25(events, question, k=20, filter_headers=False)
        texts_bm25f = get_retained_bm25(events, question, k=20, filter_headers=True)
        texts_dense = get_retained_dense(events, question, encoder, k=20)
        texts_ce    = get_retained_ce(events, question, ce_model, k=20, first_k=50)
        texts_qccs  = get_retained_qccs(events, question, gate, tok, k=20)

        # Compute semantic hit for each arm
        def any_semantic_hit(texts):
            for t in texts:
                if semantic_hit(nli_model, t, evidence):
                    return 1
            return 0

        sem_bm25  = any_semantic_hit(texts_bm25)
        sem_bm25f = any_semantic_hit(texts_bm25f)
        sem_dense = any_semantic_hit(texts_dense)
        sem_ce    = any_semantic_hit(texts_ce)
        sem_qccs  = any_semantic_hit(texts_qccs)

        # Lexical hits (recompute for confirmation)
        lex_bm25  = int(any(lexical_hit(t, evidence) for t in texts_bm25))
        lex_bm25f = int(any(lexical_hit(t, evidence) for t in texts_bm25f))
        lex_dense = int(any(lexical_hit(t, evidence) for t in texts_dense))
        lex_ce    = int(any(lexical_hit(t, evidence) for t in texts_ce))
        lex_qccs  = int(any(lexical_hit(t, evidence) for t in texts_qccs))

        rows.append({
            "filename": fname, "question": question[:80], "evidence": evidence[:100],
            # Semantic hits
            "bm25_sem":         sem_bm25,
            "bm25_filtered_sem":sem_bm25f,
            "dense_sem":        sem_dense,
            "ce_sem":           sem_ce,
            "qccs_sem":         sem_qccs,
            # Lexical hits (for cross-check)
            "bm25_lex":         lex_bm25,
            "bm25_filtered_lex":lex_bm25f,
            "dense_lex":        lex_dense,
            "ce_lex":           lex_ce,
            "qccs_lex":         lex_qccs,
        })

        elapsed = time.time() - t0
        print(f"  {i+1}/{len(df_test)}  ({elapsed:.0f}s)  "
              f"bm25_sem={sem_bm25} bm25_lex={lex_bm25}  "
              f"qccs_sem={sem_qccs} qccs_lex={lex_qccs}")

    result = pd.DataFrame(rows)
    out = FIGURES / "exp3_nli_hits.csv"
    result.to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Summary comparison
    print("\n=== Semantic vs. Lexical Stage 1 Recall ===")
    print(f"{'Arm':<18} {'Semantic':>10} {'Lexical':>10} {'Sem-only':>10} {'Lex-only':>10}")
    for arm in ["bm25", "bm25_filtered", "dense", "ce", "qccs"]:
        s = result[f"{arm}_sem"].mean() * 100
        l = result[f"{arm}_lex"].mean() * 100
        # Cases semantic hits but lexical misses (true semantic support, lexically distant)
        sem_only = ((result[f"{arm}_sem"]==1) & (result[f"{arm}_lex"]==0)).mean() * 100
        lex_only = ((result[f"{arm}_sem"]==0) & (result[f"{arm}_lex"]==1)).mean() * 100
        print(f"  {arm:<18} {s:>9.1f}%  {l:>9.1f}%  {sem_only:>9.1f}%  {lex_only:>9.1f}%")

    print("\nKey finding for conditional accuracy analysis:")
    print("  If BM25 semantic recall ≈ lexical recall (~98%), the conditional")
    print("  accuracy story is unchanged under the semantic criterion.")
    print(f"\n  BM25 semantic:  {result['bm25_sem'].mean()*100:.1f}%")
    print(f"  BM25 lexical:   {result['bm25_lex'].mean()*100:.1f}%")
    print(f"  QCCS semantic:  {result['qccs_sem'].mean()*100:.1f}%")
    print(f"  QCCS lexical:   {result['qccs_lex'].mean()*100:.1f}%")


if __name__ == "__main__":
    main()
