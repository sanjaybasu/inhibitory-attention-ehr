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


# ── Experiment 3: Qwen2.5-7B LLM inference for QCCS evaluation ───────────────

llm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch==2.3.0", "transformers==4.41.0", "accelerate",
        "bitsandbytes", "pandas", "numpy", "sentencepiece",
        "pyarrow",
    ])
    .add_local_dir(str(EXPERIMENTS_DIR), remote_path="/experiments")
)


@app.function(
    gpu="A100",
    timeout=10800,   # 3 hours
    memory=80000,
    image=llm_image,
    volumes={"/results": results_vol, "/mnt-data": data_vol},
    # Add secret when ready: secrets=[modal.Secret.from_name("huggingface-token")]
    # Create via: modal secret create huggingface-token HF_TOKEN=hf_xxxxx
)
def run_llm_inference(records: list[dict],
                      model_id: str = "Qwen/Qwen2.5-7B-Instruct",
                      use_qccs: bool = True,
                      keep_top_k: int = 20) -> list[dict]:
    """
    Run Qwen2.5-7B-Instruct on MedAlign records (with and without QCCS gate).
    Each record: {filename, question, position, context_full, context_qccs, expected}
    Returns: [{filename, question, position, baseline_correct, qccs_correct}]
    """
    import os, torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {model_id} on GPU...")
    hf_token = os.environ.get("HF_TOKEN")  # optional for public models
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16,
        token=hf_token,
    )  # No 4-bit quant needed — Qwen2.5-7B fits in fp16 on A100 80GB (~14 GB VRAM)
    # Qwen2.5 uses a different prompt template
    if "Qwen" in model_id:
        def _qwen_prompt(context: str, question: str) -> str:
            return (f"<|im_start|>system\nYou are a clinical assistant.<|im_end|>\n"
                    f"<|im_start|>user\nBased ONLY on the patient record below, "
                    f"answer the question briefly.\n\nPATIENT RECORD:\n"
                    f"{context[:16000]}\n\nQUESTION: {question}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
    else:
        _qwen_prompt = None
    model.eval()
    print(f"Model loaded. Processing {len(records)} records...")

    def generate(context: str, question: str, max_tokens: int = 128) -> str:
        if _qwen_prompt is not None:
            prompt = _qwen_prompt(context, question)
        else:
            prompt = (
                f"[INST] You are a clinical assistant. Based ONLY on the patient "
                f"record below, answer the question briefly.\n\n"
                f"PATIENT RECORD:\n{context[:16000]}\n\n"
                f"QUESTION: {question} [/INST]"
            )
        inputs = tokenizer(prompt, return_tensors="pt",
                           truncation=True, max_length=4096).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens,
                                 do_sample=False, temperature=1.0)
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True).strip()

    def score_response(response: str, expected: str) -> float:
        import re
        r, e = response.lower(), expected.lower()
        if e in r:
            return 1.0
        toks_e = set(re.findall(r'\w+', e))
        toks_r = set(re.findall(r'\w+', r))
        if toks_e and len(toks_e & toks_r) / len(toks_e) >= 0.5:
            return 0.5
        return 0.0

    results = []
    for i, rec in enumerate(records):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(records)}")

        # Baseline (full context)
        baseline_resp  = generate(rec["context_full"], rec["question"])
        baseline_score = score_response(baseline_resp, rec.get("expected", ""))

        # QCCS (compressed context)
        qccs_resp  = generate(rec["context_qccs"], rec["question"])
        qccs_score = score_response(qccs_resp, rec.get("expected", ""))

        results.append({
            "filename": rec["filename"],
            "question": rec["question"][:100],
            "position": rec.get("position", float("nan")),
            "baseline_correct": baseline_score,
            "qccs_correct": qccs_score,
            "baseline_response": baseline_resp[:200],
            "qccs_response": qccs_resp[:200],
        })

    # Save to volume
    import pandas as pd
    pd.DataFrame(results).to_csv("/results/exp3_llm_inference_results.csv",
                                 index=False)
    results_vol.commit()
    print(f"Saved {len(results)} results to volume.")
    return results


@app.local_entrypoint()
def run_llm_inference_entrypoint():
    """
    Prepare MedAlign records and dispatch LLM inference to Modal.
    Run locally: modal run modal_app.py::run_llm_inference_entrypoint
    """
    import sys, pandas as pd
    sys.path.insert(0, str(Path(__file__).parent))
    from exp3_qccs_gate import (parse_ehr_sentences, sentences_to_context,
                                 qccs_compress, QCCSGate, CharNgramTokenizer)
    import torch

    BASE    = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
    TSV     = BASE / "medalign_instructions_v1_3/clinician-reviewed-model-responses.tsv"
    EHR_DIR = BASE / "medalign_instructions_v1_3/ehrs"
    OUT_D   = Path("/Users/sanjaybasu/waymark-local/notebooks/inhibitory-attention-ehr/figures")

    df = pd.read_csv(TSV, sep="\t", low_memory=False)
    df = df[df["is_used_eval"].astype(str).str.lower().isin(["true","yes","1"])]

    # Load gate (trained by exp3_qccs_gate.py)
    tokenizer = CharNgramTokenizer()
    gate = QCCSGate()
    gate_path = OUT_D / "qccs_gate.pt"
    if gate_path.exists():
        gate.load_state_dict(torch.load(gate_path, map_location="cpu"))
        print("Loaded QCCS gate from disk.")
    else:
        print("WARNING: gate not trained yet. Run exp3_qccs_gate.py first.")

    # Load positions from Exp1
    pos_path = OUT_D / "exp1_litm_results.csv"
    pos_df = pd.read_csv(pos_path) if pos_path.exists() else pd.DataFrame()

    records = []
    for fname in df["filename"].unique():
        xml_path = EHR_DIR / fname
        if not xml_path.exists():
            continue
        events = parse_ehr_sentences(xml_path)
        context_full = sentences_to_context(events)

        patient_rows = df[df["filename"] == fname]
        for _, row in patient_rows.head(3).iterrows():  # max 3 per patient
            query = str(row.get("question", ""))
            expected = str(row.get("clinician_response", ""))

            ctx_qccs, _ = qccs_compress(events, query, gate, tokenizer,
                                         keep_top_k=20)

            pos_row = pos_df[
                (pos_df["filename"] == fname) &
                (pos_df["question"] == query)
            ]
            pos = float(pos_row["position"].iloc[0]) if len(pos_row) > 0 else float("nan")

            records.append({"filename": fname, "question": query,
                             "expected": expected, "position": pos,
                             "context_full": context_full[:20000],  # pre-truncate for Modal payload limit
                             "context_qccs": ctx_qccs[:20000]})

    print(f"Dispatching {len(records)} records to Modal A100...")
    results = run_llm_inference.remote(records)
    print(f"Done. Results: {len(results)} rows")

    pd.DataFrame(results).to_csv(
        OUT_D / "exp3_llm_inference_results.csv", index=False)
    print(f"Saved to {OUT_D / 'exp3_llm_inference_results.csv'}")
