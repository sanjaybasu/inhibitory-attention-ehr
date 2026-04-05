"""
Compute Hanley-McNeil 95% CIs for QCCS-DiffAttn (exp2b) results.
Run after exp2_qccs_diffattn_results.csv is available.

Usage:
  python experiments/compute_qccs_diffattn_ci.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

FIGURES = Path(__file__).parent.parent / "figures"

# n1/n0 for QCCS-DiffAttn test set (EHRSHOT official splits: 34.1% train, 65.9% non-train)
# run_task uses ~train_mask (tuning + held_out) as test set.
# Estimated by scaling full-dataset counts (from bmfvjqhwa task output) by 0.659.
# NOTE: Standard/DiffAttn/H2O used random 70/15/15 split (much smaller test sets).
TEST_N = {
    "lab_anemia":           {"n1": 51710, "n0": 23665},
    "lab_hyperkalemia":     {"n1": 1937,  "n0": 79661},
    "lab_hyponatremia":     {"n1": 24272, "n0": 62388},
    "lab_thrombocytopenia": {"n1": 24386, "n0": 48659},
}

TASK_DISPLAY = {
    "lab_anemia":           "Anemia",
    "lab_hyperkalemia":     "Hyperkalemia",
    "lab_hyponatremia":     "Hyponatremia",
    "lab_thrombocytopenia": "Thrombocytopenia",
}


def hanley_mcneil_ci(auroc: float, n1: int, n0: int, alpha: float = 0.05):
    """Hanley-McNeil 95% CI for AUROC."""
    A = auroc
    Q1 = A / (2 - A)
    Q2 = 2 * A**2 / (1 + A)
    se2 = (A * (1 - A) + (n1 - 1) * (Q1 - A**2) + (n0 - 1) * (Q2 - A**2)) / (n1 * n0)
    se = np.sqrt(se2)
    z = 1.95996
    return A - z * se, A + z * se


def fmt_auroc(auroc, lo, hi):
    return f"{auroc:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    results_path = FIGURES / "exp2_qccs_diffattn_results.csv"
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        print("Run: modal run experiments/modal_app.py::run_exp2b_all")
        return

    df = pd.read_csv(results_path)
    print(f"\nQCCS-DiffAttn results ({len(df)} tasks):")
    print(df.to_string(index=False))
    print()

    rows = []
    print("=== QCCS-DiffAttn AUROC with Hanley-McNeil 95% CIs ===")
    for _, row in df.iterrows():
        task = row["task"]
        auroc = float(row["test_auroc"])
        auprc = float(row["test_auprc"])
        n = TEST_N.get(task)
        if n is None:
            print(f"  {task}: no n1/n0 — skipping CI")
            continue
        lo, hi = hanley_mcneil_ci(auroc, n["n1"], n["n0"])
        auroc_str = fmt_auroc(auroc, lo, hi)
        disp = TASK_DISPLAY.get(task, task)
        print(f"  {disp}: {auroc_str} / {auprc:.3f}")
        rows.append({
            "Task": disp,
            "Model": "QCCS-DiffAttn (Eq.3)",
            "test_AUROC": auroc,
            "CI_lo": lo,
            "CI_hi": hi,
            "AUROC_str": auroc_str,
            "test_AUPRC": auprc,
        })

    if rows:
        out_df = pd.DataFrame(rows)
        out_path = FIGURES / "exp2b_qccs_diffattn_ci.csv"
        out_df.to_csv(out_path, index=False)
        print(f"\nSaved: {out_path}")

        print("\n=== LaTeX table rows (copy into tab:ehrshot) ===")
        # Assumes bold = best per task; we don't know yet without comparing
        for r in rows:
            print(f"{r['Task']:20s}  & {r['AUROC_str']} / {r['test_AUPRC']:.3f} \\\\")

    # Also load reference CIs for comparison
    ci_path = FIGURES / "exp2_auroc_with_ci.csv"
    if ci_path.exists():
        ref = pd.read_csv(ci_path)
        print("\n=== Comparison with existing models ===")
        for task_key, disp in TASK_DISPLAY.items():
            print(f"\n{disp}:")
            task_ref = ref[ref["Task"] == disp]
            for _, r in task_ref.iterrows():
                print(f"  {r['Model']:30s}: {r['AUROC_str']}")
            # QCCS-DiffAttn
            matching = [r for r in rows if r["Task"] == disp]
            if matching:
                print(f"  {'QCCS-DiffAttn (Eq.3)':30s}: {matching[0]['AUROC_str']}")


if __name__ == "__main__":
    main()
