"""
Modal App: GPU compute for Experiments 2 & 3
=============================================
Runs Experiment 2 (all 5 EHRSHOT lab tasks, full dataset) and
Experiment 3 (Mistral-7B LLM re-inference on QCCS-compressed MedAlign records)
on Modal GPU instances.

Usage:
  modal run modal_app.py::run_exp2_all       # Full EHRSHOT experiments (A10G)
  modal run modal_app.py::run_llm_inference  # LLM re-inference for Exp 3 (A100)

Setup:
  pip install modal
  modal token new
"""

import modal
from pathlib import Path

# ── Modal app definition ──────────────────────────────────────────────────────
app = modal.App("clinical-litm-experiments")

LOCAL_ROOT = Path("/Users/sanjaybasu/waymark-local")
EXPERIMENTS_DIR = Path(__file__).parent

# Shared volumes
results_vol = modal.Volume.from_name("clinical-litm-results", create_if_missing=True)
data_vol    = modal.Volume.from_name("clinical-litm-data",    create_if_missing=True)

# Base image — experiment scripts baked in via add_local_dir (Modal 1.3.5+)
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.3.0", "pandas", "numpy", "scikit-learn",
        "matplotlib", "pyarrow", "transformers==4.41.0",
        "accelerate", "bitsandbytes",
        "sentencepiece", "protobuf",
    ])
    .add_local_dir(str(EXPERIMENTS_DIR), remote_path="/experiments")
)


# ── Data upload helper (run once before experiments) ─────────────────────────

@app.local_entrypoint()
def upload_data():
    """Upload EHRSHOT + MedAlign data to Modal Volume (run once).

    modal run modal_app.py::upload_data
    """
    EHRSHOT_FILES = LOCAL_ROOT / "data" / "ehrshot" / "EHRSHOT_files"
    LOCAL_MEDALIGN = LOCAL_ROOT / "data" / "medalign"

    # Only upload what Exp 2 needs (~330 MB), not the 1.1 GB CLMBR model
    needed = [
        (EHRSHOT_FILES / "EHRSHOT_MEDS" / "data",     "/ehrshot/EHRSHOT_files/EHRSHOT_MEDS/data"),
        (EHRSHOT_FILES / "EHRSHOT_MEDS" / "metadata", "/ehrshot/EHRSHOT_files/EHRSHOT_MEDS/metadata"),
        (EHRSHOT_FILES / "EHRSHOT_ASSETS" / "benchmark",
                                                       "/ehrshot/EHRSHOT_files/EHRSHOT_ASSETS/benchmark"),
    ]
    print("Uploading EHRSHOT (needed subset, ~330 MB)...")
    with data_vol.batch_upload(force=True) as batch:
        for local_p, remote_p in needed:
            print(f"  {local_p.name} → {remote_p}")
            batch.put_directory(str(local_p), remote_p)
    print("Uploading MedAlign (~438 MB)...")
    with data_vol.batch_upload(force=True) as batch:
        batch.put_directory(str(LOCAL_MEDALIGN), "/medalign")
    print("Done (~770 MB total). Now run: modal run modal_app.py::run_exp2_all")


# ── Experiment 2: Full EHRSHOT on GPU ─────────────────────────────────────────

@app.function(
    gpu="A10G",
    timeout=7200,
    memory=32768,
    image=base_image,
    volumes={"/results": results_vol, "/mnt-data": data_vol},
)
def run_exp2_task(task: str, epochs: int = 40):
    """Run Experiment 2 for one EHRSHOT lab task on GPU."""
    import sys
    sys.path.insert(0, "/experiments")
    from exp2_ehrshot_diffattn import run_exp2, LAB_TASKS

    # Override paths to use volume-mounted data
    import exp2_ehrshot_diffattn as exp2
    from pathlib import Path
    exp2.MEDS_BASE   = Path("/mnt-data/ehrshot/EHRSHOT_files/EHRSHOT_MEDS")
    exp2.ASSETS_BASE = Path("/mnt-data/ehrshot/EHRSHOT_files/EHRSHOT_ASSETS")
    exp2.OUT_DIR     = Path("/results")

    result = run_exp2(task=task, epochs=epochs, device_str="cuda",
                      n_patients=0)  # 0 = full dataset
    results_vol.commit()
    return result.to_dict(orient="records")


@app.local_entrypoint()
def run_exp2_all():
    """Run all 5 lab tasks in parallel on separate A10G instances."""
    from exp2_ehrshot_diffattn import LAB_TASKS
    tasks = list(LAB_TASKS.keys())
    print(f"Launching {len(tasks)} tasks in parallel: {tasks}")

    results = list(run_exp2_task.starmap([(t, 40) for t in tasks]))
    for task, res in zip(tasks, results):
        print(f"\n{task}: {res}")


# ── Experiment 3 v2: Qwen2.5-7B LLM inference — 3-arm (Full / QCCS / BM25) ───
#
# KEY FIXES vs original:
#   1. Contexts built ON Modal from volume data (no payload-size truncation)
#   2. Full-context baseline uses 32k-token window (vs. 4096 previously)
#   3. BM25 added as third comparison arm alongside QCCS
#   4. Same 30% test split (seed=42) as exp3_baselines.py for consistency

LOCAL_FIGURES = Path("/Users/sanjaybasu/waymark-local/notebooks/inhibitory-attention-ehr/figures")

llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.3.0", "transformers==4.41.0", "accelerate",
        "bitsandbytes", "pandas", "numpy", "sentencepiece",
        "pyarrow", "rank-bm25", "scikit-learn", "matplotlib",
    ])
    .add_local_dir(str(EXPERIMENTS_DIR), remote_path="/experiments")
    .add_local_file(str(LOCAL_FIGURES / "qccs_gate.pt"), remote_path="/gate/qccs_gate.pt")
)


@app.function(
    gpu="A100",
    timeout=10800,   # 3 hours
    memory=80000,
    image=llm_image,
    volumes={"/results": results_vol, "/mnt-data": data_vol},
)
def run_llm_inference_v2(records: list[dict],
                         model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                         keep_top_k: int = 20) -> list[dict]:
    """
    Three-arm LLM evaluation on MedAlign test split.

    Contexts are built here from volume-mounted EHR data so that the full
    EHR text is available without payload-size truncation.

    Arms:
      baseline  — full chronological EHR, truncated at 32k tokens (model window)
      qccs      — top-k sentences by learned gate + 5 most-recent events
      bm25      — top-k sentences by BM25 + 5 most-recent events

    Each record: {filename, question, expected, position}
    Returns list of {filename, question, position,
                     baseline_correct, qccs_correct, bm25_correct,
                     baseline_response, qccs_response, bm25_response}
    """
    import os, re, sys, torch
    import xml.etree.ElementTree as ET
    import pandas as pd
    from pathlib import Path as _Path
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from rank_bm25 import BM25Okapi

    sys.path.insert(0, "/experiments")
    from exp3_qccs_gate import (parse_ehr_sentences, sentences_to_context,
                                 qccs_compress, QCCSGate, CharNgramTokenizer)

    EHR_DIR = _Path("/mnt-data/medalign/MedAlign_files/medalign_instructions_v1_3/ehrs")

    # ── Load gate ──────────────────────────────────────────────────────────────
    gate_path = _Path("/gate/qccs_gate.pt")
    gate = QCCSGate()
    state = torch.load(str(gate_path), map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        gate.load_state_dict(state["state_dict"])
    else:
        gate.load_state_dict(state)
    gate.eval()
    tok_fn = CharNgramTokenizer()

    # ── Load LLM ───────────────────────────────────────────────────────────────
    print(f"Loading {model_id} on GPU...")
    hf_token = os.environ.get("HF_TOKEN")
    llm_tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, token=hf_token,
    )
    model.eval()
    print(f"Model loaded. Processing {len(records)} records...")

    # ── Helpers ────────────────────────────────────────────────────────────────
    def build_prompt(context: str, question: str) -> str:
        return (f"<|im_start|>system\nYou are a clinical assistant.<|im_end|>\n"
                f"<|im_start|>user\nBased ONLY on the patient record below, "
                f"answer the question briefly.\n\nPATIENT RECORD:\n"
                f"{context}\n\nQUESTION: {question}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def generate(context: str, question: str,
                 max_new_tokens: int = 128, ctx_token_limit: int = 32768) -> str:
        prompt = build_prompt(context, question)
        inputs = llm_tok(prompt, return_tensors="pt",
                         truncation=True, max_length=ctx_token_limit).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=1.0)
        return llm_tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

    def score_response(response: str, expected: str) -> float:
        r, e = response.lower(), expected.lower()
        if e in r:
            return 1.0
        toks_e = set(re.findall(r'\w+', e))
        toks_r = set(re.findall(r'\w+', r))
        if toks_e and len(toks_e & toks_r) / len(toks_e) >= 0.5:
            return 0.5
        return 0.0

    # ── Cache parsed EHRs and per-patient BM25 indices ────────────────────────
    xml_cache: dict   = {}
    bm25_cache: dict  = {}   # fname → BM25Okapi (built once per patient)
    corpus_cache: dict = {}  # fname → tokenized corpus (for BM25)

    def get_events(fname: str) -> list[dict]:
        if fname not in xml_cache:
            xml_path = EHR_DIR / fname
            xml_cache[fname] = parse_ehr_sentences(xml_path) if xml_path.exists() else []
        return xml_cache[fname]

    def get_bm25(fname: str, events: list[dict]):
        """Return cached (corpus, BM25Okapi) for this patient."""
        if fname not in bm25_cache:
            corpus_cache[fname] = [re.findall(r'\w+', ev["text"].lower())
                                   for ev in events]
            bm25_cache[fname]   = BM25Okapi(corpus_cache[fname]) if any(corpus_cache[fname]) else None
        return corpus_cache[fname], bm25_cache[fname]

    def bm25_compress(fname: str, events: list[dict],
                      query: str, keep_top_k: int = 20) -> str:
        """Select top-k events by BM25 score + 5 most-recent, then serialize."""
        corpus, bm25_idx = get_bm25(fname, events)
        recent_idx = {ev["idx"] for ev in events[-5:]}
        if bm25_idx is None:
            kept = events[-keep_top_k:]
        else:
            q_tokens = re.findall(r'\w+', query.lower())
            scores = bm25_idx.get_scores(q_tokens) if q_tokens else [0.0] * len(events)
            scored = sorted(range(len(scores)), key=lambda i: -scores[i])
            keep = set(scored[:keep_top_k]) | recent_idx
            kept = [ev for ev in events if ev["idx"] in keep]
        kept.sort(key=lambda e: e["timestamp"])
        return sentences_to_context(kept)

    # ── Main loop with incremental saves every 50 records ─────────────────────
    SAVE_EVERY = 10   # 83 unique records total; checkpoint every 10
    results = []
    for i, rec in enumerate(records):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}")

        fname  = rec["filename"]
        events = get_events(fname)
        if not events:
            continue

        question = rec["question"]
        expected = rec.get("expected", "")

        ctx_full = sentences_to_context(events, max_chars=400000)   # no cap; tokenizer truncates
        ctx_qccs, _ = qccs_compress(events, question, gate, tok_fn, keep_top_k=keep_top_k)
        ctx_bm25 = bm25_compress(fname, events, question, keep_top_k=keep_top_k)

        # Full-context arm: 16k-token window (32k caused OOM during logits.float() prefill on A100 80GB)
        # Compressed arms: 4k tokens is sufficient for ~20 selected sentences
        bl_resp   = generate(ctx_full,  question, ctx_token_limit=16384)
        qccs_resp = generate(ctx_qccs,  question, ctx_token_limit=4096)
        bm25_resp = generate(ctx_bm25,  question, ctx_token_limit=4096)

        results.append({
            "filename":          fname,
            "question":          question[:100],
            "position":          rec.get("position", float("nan")),
            "baseline_correct":  score_response(bl_resp,   expected),
            "qccs_correct":      score_response(qccs_resp, expected),
            "bm25_correct":      score_response(bm25_resp, expected),
            "baseline_response": bl_resp[:200],
            "qccs_response":     qccs_resp[:200],
            "bm25_response":     bm25_resp[:200],
        })

        # Incremental checkpoint every SAVE_EVERY records
        if len(results) % SAVE_EVERY == 0:
            pd.DataFrame(results).to_csv("/results/exp3_v2_llm_results.csv", index=False)
            results_vol.commit()
            print(f"  [checkpoint] saved {len(results)} rows")

    pd.DataFrame(results).to_csv("/results/exp3_v2_llm_results.csv", index=False)
    results_vol.commit()
    print(f"Saved {len(results)} results to /results/exp3_v2_llm_results.csv")
    return results


@app.local_entrypoint()
def run_llm_v2_entrypoint():
    """
    Prepare 30%-test-split MedAlign records and run 3-arm LLM inference on Modal A100.

    Run: modal run modal_app.py::run_llm_v2_entrypoint
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    BASE    = LOCAL_ROOT / "data" / "medalign" / "MedAlign_files"
    TSV     = BASE / "medalign_instructions_v1_3" / "clinician-reviewed-model-responses.tsv"
    OUT_D   = LOCAL_FIGURES

    df = pd.read_csv(TSV, sep="\t")
    df = df.dropna(subset=["question", "evidence", "filename"])

    # Same 30% test split as exp3_baselines.py (seed=42)
    patients = df["filename"].unique()
    _, test_patients = train_test_split(patients, test_size=0.30, random_state=42)
    df_test = df[df["filename"].isin(set(test_patients))].copy()

    # Deduplicate: the TSV has ~8 rows per (filename, question) — one per model
    # evaluated in the MedAlign benchmark. Since Qwen2.5 inference is deterministic
    # (do_sample=False), all copies give identical results; we keep one per unique pair.
    df_test = df_test.drop_duplicates(subset=["filename", "question"])
    print(f"Test split: {len(test_patients)} patients, {len(df_test)} unique instructions")

    # Load gold positions from Exp 1
    pos_path = OUT_D / "exp1_litm_results.csv"
    pos_df = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame()

    records = []
    for _, row in df_test.iterrows():
        fname    = str(row["filename"])
        question = str(row["question"])
        # Use 'evidence' (short gold answer directly from EHR) as expected.
        # 'clinician_response' is a model-specific long-form annotation and
        # does not match what a 7B model generates.
        expected = str(row.get("evidence", "")).strip()

        pos_row = pos_df[
            (pos_df["filename"] == fname) &
            (pos_df["question"]  == question)
        ] if not pos_df.empty else pd.DataFrame()
        pos = float(pos_row["position"].iloc[0]) if len(pos_row) > 0 else float("nan")

        records.append({"filename": fname, "question": question,
                         "expected": expected, "position": pos})

    print(f"Dispatching {len(records)} records to Modal A100 (3 arms × {len(records)} = "
          f"{len(records)*3} LLM calls)...")

    # Use spawn() so the A100 job continues even if the local process exits.
    # Results are saved incrementally to the Modal volume (every 50 records),
    # and in full upon completion. Download with:
    #   modal volume get clinical-litm-results exp3_v2_llm_results.csv <local_path>
    handle = run_llm_inference_v2.spawn(records)
    print(f"Spawned detached job. Function call ID: {handle.object_id}")
    print("Job is running on A100. Check progress:")
    print("  modal volume ls clinical-litm-results")
    print("  modal app logs ap-...  (see app ID in Modal dashboard)")
    print("Download results when done:")
    print("  modal volume get clinical-litm-results exp3_v2_llm_results.csv "
          f"{OUT_D}/exp3_v2_llm_results.csv")


# ── Experiment 3 v3: 5-arm evaluation (Full / QCCS / BM25 / BM25-filtered / Dense) ───
#
# Extends v2 by adding:
#   4. dense         — sentence-transformer cosine similarity top-k
#   5. bm25_filtered — BM25 after removing section-header false positives
# Both ablations test the hypothesis that BM25's failure is due to header contamination
# (bm25_filtered) and that semantic retrieval bridges recall–accuracy gap (dense).

llm_image_v3 = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.3.0", "transformers==4.41.0", "accelerate",
        "bitsandbytes", "pandas", "numpy", "sentencepiece",
        "pyarrow", "rank-bm25", "scikit-learn", "matplotlib",
        "sentence-transformers",
    ])
    .add_local_dir(str(EXPERIMENTS_DIR), remote_path="/experiments")
    .add_local_file(str(LOCAL_FIGURES / "qccs_gate.pt"), remote_path="/gate/qccs_gate.pt")
)


@app.function(
    gpu="A100",
    timeout=14400,   # 4 hours — 5 arms × 83 records
    memory=80000,
    image=llm_image_v3,
    volumes={"/results": results_vol, "/mnt-data": data_vol},
)
def run_llm_inference_v3(records: list[dict],
                          model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                          keep_top_k: int = 20) -> list[dict]:
    """
    Five-arm LLM evaluation on MedAlign test split.

    Arms:
      baseline      — full chronological EHR, truncated at 16k tokens
      qccs          — top-k sentences by learned gate + 5 most-recent events
      bm25          — top-k by BM25 score + 5 most-recent events
      bm25_filtered — BM25 after filtering section-header false positives
      dense         — top-k by sentence-transformer cosine similarity + 5 most-recent

    Each record: {filename, question, expected, position}
    Output file: /results/exp3_v3_llm_results.csv
    """
    import os, re, sys, torch
    import numpy as np
    import pandas as pd
    from pathlib import Path as _Path
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer

    sys.path.insert(0, "/experiments")
    from exp3_qccs_gate import (parse_ehr_sentences, sentences_to_context,
                                 qccs_compress, QCCSGate, CharNgramTokenizer)

    EHR_DIR = _Path("/mnt-data/medalign/MedAlign_files/medalign_instructions_v1_3/ehrs")

    # ── Section-header filter (matches exp3_extended_baselines.py) ─────────────
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

    def _is_header(text: str) -> bool:
        t = text.strip()
        if len(t) < 5:
            return True
        for pat in _HEADER_PATS:
            if pat.match(t):
                return True
        if len(t) <= 40 and t.upper() == t and re.match(r'^[A-Z\\s/:-]+$', t):
            return True
        return False

    # ── Load QCCS gate ─────────────────────────────────────────────────────────
    gate = QCCSGate()
    state = torch.load("/gate/qccs_gate.pt", map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        gate.load_state_dict(state["state_dict"])
    else:
        gate.load_state_dict(state)
    gate.eval()
    tok_fn = CharNgramTokenizer()

    # ── Load dense encoder ─────────────────────────────────────────────────────
    print("Loading sentence encoder (all-MiniLM-L6-v2)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    # ── Load LLM ───────────────────────────────────────────────────────────────
    print(f"Loading {model_id} on GPU...")
    hf_token = os.environ.get("HF_TOKEN")
    llm_tok = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16, token=hf_token,
    )
    model.eval()
    print(f"Model loaded. Processing {len(records)} records (5 arms each)...")

    # ── Helpers ────────────────────────────────────────────────────────────────
    def build_prompt(context: str, question: str) -> str:
        return (f"<|im_start|>system\nYou are a clinical assistant.<|im_end|>\n"
                f"<|im_start|>user\nBased ONLY on the patient record below, "
                f"answer the question briefly.\n\nPATIENT RECORD:\n"
                f"{context}\n\nQUESTION: {question}<|im_end|>\n"
                f"<|im_start|>assistant\n")

    def generate(context: str, question: str,
                 max_new_tokens: int = 128, ctx_token_limit: int = 32768) -> str:
        prompt = build_prompt(context, question)
        inputs = llm_tok(prompt, return_tensors="pt",
                         truncation=True, max_length=ctx_token_limit).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=1.0)
        return llm_tok.decode(out[0][inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip()

    def score_response(response: str, expected: str) -> float:
        r, e = response.lower(), expected.lower()
        if e in r:
            return 1.0
        toks_e = set(re.findall(r'\w+', e))
        toks_r = set(re.findall(r'\w+', r))
        if toks_e and len(toks_e & toks_r) / len(toks_e) >= 0.5:
            return 0.5
        return 0.0

    # ── Context builders ───────────────────────────────────────────────────────
    xml_cache: dict = {}

    def get_events(fname: str) -> list[dict]:
        if fname not in xml_cache:
            xml_path = EHR_DIR / fname
            xml_cache[fname] = parse_ehr_sentences(xml_path) if xml_path.exists() else []
        return xml_cache[fname]

    def _bm25_compress_inner(events: list[dict], query: str,
                              keep_top_k: int, filter_headers: bool) -> str:
        """Shared BM25 logic; filter_headers controls header removal."""
        ev = ([e for e in events if not _is_header(e["text"])]
              if filter_headers else events)
        if not ev:
            ev = events  # fall back to unfiltered if all removed
        recent_idx_set = {e["idx"] for e in events[-5:]}
        corpus = [re.findall(r'\w+', e["text"].lower()) for e in ev]
        bm25_idx = BM25Okapi(corpus) if any(corpus) else None
        if bm25_idx is None:
            kept = list(events[-keep_top_k:])
        else:
            q_toks = re.findall(r'\w+', query.lower())
            scores = bm25_idx.get_scores(q_toks) if q_toks else [0.0] * len(ev)
            scored = sorted(range(len(scores)), key=lambda i: -scores[i])
            keep_pos = set(scored[:keep_top_k])
            kept = [ev[i] for i in range(len(ev)) if i in keep_pos]
            for e in events:  # add recency buffer from full list
                if e["idx"] in recent_idx_set and e not in kept:
                    kept.append(e)
        kept.sort(key=lambda e: e["timestamp"])
        return sentences_to_context(kept)

    def dense_compress(events: list[dict], query: str, keep_top_k: int) -> str:
        texts = [e["text"] for e in events]
        q_emb   = encoder.encode([query], normalize_embeddings=True)
        ev_embs = encoder.encode(texts, normalize_embeddings=True, batch_size=128)
        sims    = (ev_embs @ q_emb.T).ravel()
        recent_idx_set = {e["idx"] for e in events[-5:]}
        top_pos = set(int(i) for i in np.argsort(sims)[-keep_top_k:])
        top_pos |= {i for i, e in enumerate(events) if e["idx"] in recent_idx_set}
        kept = [events[i] for i in range(len(events)) if i in top_pos]
        kept.sort(key=lambda e: e["timestamp"])
        return sentences_to_context(kept)

    # ── Load any existing partial results (resume support) ────────────────────
    _partial = _Path("/results/exp3_v3_llm_results.csv")
    if _partial.exists():
        existing = pd.read_csv(_partial)
        results = existing.to_dict("records")
        print(f"  Loaded {len(results)} existing rows from partial run")
    else:
        results = []

    # ── Main inference loop ────────────────────────────────────────────────────
    for i, rec in enumerate(records):
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(records)}")

        fname   = rec["filename"]
        events  = get_events(fname)
        if not events:
            continue

        question = rec["question"]
        expected = rec.get("expected", "")

        try:
            ctx_full  = sentences_to_context(events, max_chars=400000)
            ctx_qccs, _ = qccs_compress(events, question, gate, tok_fn,
                                         keep_top_k=keep_top_k)
            ctx_bm25  = _bm25_compress_inner(events, question, keep_top_k,
                                              filter_headers=False)
            ctx_bm25f = _bm25_compress_inner(events, question, keep_top_k,
                                              filter_headers=True)
            ctx_dense = dense_compress(events, question, keep_top_k)

            bl_resp    = generate(ctx_full,  question, ctx_token_limit=16384)
            qccs_resp  = generate(ctx_qccs,  question, ctx_token_limit=4096)
            bm25_resp  = generate(ctx_bm25,  question, ctx_token_limit=4096)
            bm25f_resp = generate(ctx_bm25f, question, ctx_token_limit=4096)
            dns_resp   = generate(ctx_dense, question, ctx_token_limit=4096)
        except Exception as exc:
            print(f"  [ERROR] record {i} ({fname}): {exc} — skipping")
            bl_resp = qccs_resp = bm25_resp = bm25f_resp = dns_resp = ""

        results.append({
            "filename":                fname,
            "question":                question[:100],
            "position":                rec.get("position", float("nan")),
            "baseline_correct":        score_response(bl_resp,    expected),
            "qccs_correct":            score_response(qccs_resp,  expected),
            "bm25_correct":            score_response(bm25_resp,  expected),
            "bm25_filtered_correct":   score_response(bm25f_resp, expected),
            "dense_correct":           score_response(dns_resp,   expected),
            "baseline_response":       bl_resp[:200],
            "qccs_response":           qccs_resp[:200],
            "bm25_response":           bm25_resp[:200],
            "bm25_filtered_response":  bm25f_resp[:200],
            "dense_response":          dns_resp[:200],
        })

        if len(results) % 10 == 0:
            pd.DataFrame(results).to_csv("/results/exp3_v3_llm_results.csv", index=False)
            results_vol.commit()
            print(f"  [checkpoint] {len(results)} rows saved")

    pd.DataFrame(results).to_csv("/results/exp3_v3_llm_results.csv", index=False)
    results_vol.commit()
    print(f"Saved {len(results)} rows to /results/exp3_v3_llm_results.csv")
    return results


@app.local_entrypoint()
def run_llm_v3_entrypoint():
    """
    Five-arm LLM evaluation: Full / QCCS / BM25 / BM25-filtered / Dense.

    Run: modal run modal_app.py::run_llm_v3_entrypoint
    Download: modal volume get clinical-litm-results exp3_v3_llm_results.csv <local_path>
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    BASE  = LOCAL_ROOT / "data" / "medalign" / "MedAlign_files"
    TSV   = BASE / "medalign_instructions_v1_3" / "clinician-reviewed-model-responses.tsv"
    OUT_D = LOCAL_FIGURES

    df = pd.read_csv(TSV, sep="\t").dropna(subset=["question", "evidence", "filename"])
    patients = df["filename"].unique()
    _, test_patients = train_test_split(patients, test_size=0.30, random_state=42)
    df_test = (df[df["filename"].isin(set(test_patients))]
               .drop_duplicates(subset=["filename", "question"]))
    print(f"Test split: {len(test_patients)} patients, {len(df_test)} unique instructions")

    pos_df = pd.read_csv(OUT_D / "exp1_litm_results.csv")

    # Resume: skip records already saved in a partial run
    partial_path = OUT_D / "exp3_v3_llm_results.csv"
    done_keys: set = set()
    if partial_path.exists():
        done_df = pd.read_csv(partial_path)
        done_keys = set(zip(done_df["filename"].astype(str),
                            done_df["question"].astype(str).str[:100]))
        print(f"Resuming: {len(done_keys)} records already done")

    records = []
    for _, row in df_test.iterrows():
        fname    = str(row["filename"])
        question = str(row["question"])
        if (fname, question[:100]) in done_keys:
            continue
        expected = str(row.get("evidence", "")).strip()
        pos_row  = pos_df[(pos_df["filename"] == fname) &
                          (pos_df["question"]  == question)]
        pos = float(pos_row["position"].iloc[0]) if len(pos_row) > 0 else float("nan")
        records.append({"filename": fname, "question": question,
                        "expected": expected, "position": pos})

    print(f"Dispatching {len(records)} remaining records (5 arms × {len(records)} = "
          f"{len(records)*5} LLM calls) to A100...")
    handle = run_llm_inference_v3.spawn(records)
    print(f"Spawned job: {handle.object_id}")
    print("Download when done:")
    print(f"  modal volume get clinical-litm-results exp3_v3_llm_results.csv "
          f"{OUT_D}/exp3_v3_llm_results.csv")
