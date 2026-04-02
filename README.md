# Inhibitory Attention for Clinical Long-Context Reasoning

Code for: **"Inhibitory Attention for Clinical Long-Context Reasoning: Characterizing and Mitigating Lost-in-the-Middle Effects in EHR Processing"**

Sanjay Basu (Waymark Care / UCSF), Sadiq Patel (Waymark Care / UPenn)

*Under review, 2026*

---

## Overview

This repository contains all experiment code for:

| Experiment | File | Description |
|---|---|---|
| Exp 1: CLitM Characterization | `experiments/exp1_medalign_litm.py` | U-curve positional bias on MedAlign (6 LLMs) |
| Exp 2: EHRSHOT Structured Prediction | `experiments/exp2_ehrshot_diffattn.py` | Differential Transformer vs standard attention on EHRSHOT lab tasks |
| Exp 3: QCCS Gate Training | `experiments/exp3_qccs_gate.py` | Train the QCCS gate on MedAlign |
| Exp 3 Stage 1: Extended Baselines | `experiments/exp3_extended_baselines.py` | BM25 / BM25-filtered / Dense / QCCS retrieval recall (N=83 dedup) |
| Exp 3 Stage 2: LLM Re-inference via Modal | `experiments/modal_app.py` | 5-arm Qwen2.5-7B-Instruct re-inference (Full / BM25 / B25f / Dense / QCCS) |
| Exp 3 LLM Judge | `experiments/exp3_llm_judge.py` | Claude Haiku semantic judging of Stage 2 responses |
| Analysis + Figures | `experiments/analyze_results.py` | Build all tables and figures from raw CSVs |

## Setup

```bash
pip install -r requirements.txt
```

GPU recommended for Exp 2 (A10G or better). Exp 3 gate trains on CPU in ~15 minutes.

Exp 3 LLM re-inference runs on [Modal](https://modal.com) (requires Modal account):
```bash
pip install modal
modal setup
```

## Data Access

See `data/README.md` for instructions to obtain MedAlign and EHRSHOT.

## Reproducing Results

### Experiment 1 (CLitM U-curve)

```bash
python experiments/exp1_medalign_litm.py
# Output: figures/exp1_litm_ucurve.png, figures/exp1_litm_results.csv
```

### Experiment 2 (EHRSHOT structured prediction)

```bash
# Run on Modal GPU (A10G) — all 5 lab tasks
modal run experiments/modal_app.py::run_exp2
# Output: figures/exp2_lab_<task>_results.csv for each task
```

### Experiment 3 (QCCS gate + LLM re-inference)

```bash
# Step 1: Train QCCS gate locally (CPU, ~15 min)
python experiments/exp3_qccs_gate.py
# Output: figures/qccs_gate.pt

# Step 2: Stage 1 retrieval recall — BM25, BM25-filtered, Dense, QCCS
python experiments/exp3_extended_baselines.py
# Output: figures/exp3_extended_stage1.csv (N=83, 4 methods)

# Step 3: 5-arm LLM re-inference on Modal GPU (A100 80GB)
modal run experiments/modal_app.py::run_llm_v3_entrypoint
# Output: figures/exp3_v3_llm_results.csv (N=83, 5 arms)

# Step 4: LLM-as-judge scoring (Claude Haiku)
python experiments/exp3_llm_judge.py --v3
# Output: figures/exp3_v3_llm_results_judged.csv
```

### Build all tables and figures

```bash
python experiments/analyze_results.py
# Outputs: figures/table1_exp2_full.csv, figures/table2_exp3_llm.csv
#          figures/exp2_full_auroc.png, figures/exp3_llm_ucurve.png
```

## Citation

```bibtex
@inproceedings{basu2026inhibitory,
  title     = {Inhibitory Attention for Clinical Long-Context Reasoning:
               Characterizing and Mitigating Lost-in-the-Middle Effects
               in {EHR} Processing},
  author    = {Basu, Sanjay and Patel, Sadiq},
  note      = {Under review, 2026},
  year      = {2026}
}
```

## License

Code: MIT License. See LICENSE.

Data: Subject to respective data use agreements (MedAlign DUA, EHRSHOT PhysioNet).
