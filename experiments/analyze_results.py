"""
Analyze and visualize full Exp 2 and Exp 3 LLM results.

Run AFTER download_results.sh completes:
    python analyze_results.py

Outputs:
  figures/table1_exp2_full.csv    — Table 1 for paper (all 5 tasks × 3 models)
  figures/table2_exp3_llm.csv     — Table 2 for paper (LLM re-inference by position)
  figures/exp2_full_auroc.png     — Figure: AUROC comparison across all 5 tasks
  figures/exp3_llm_ucurve.png     — Figure: baseline vs QCCS U-curve (LLM accuracy)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

FIGURES = Path(__file__).parent.parent / "figures"
TASKS = ["lab_anemia", "lab_hyperkalemia", "lab_hypoglycemia",
         "lab_hyponatremia", "lab_thrombocytopenia"]

TASK_LABELS = {
    "lab_anemia":           "Anemia",
    "lab_hyperkalemia":     "Hyperkalemia",
    "lab_hypoglycemia":     "Hypoglycemia",
    "lab_hyponatremia":     "Hyponatremia",
    "lab_thrombocytopenia": "Thrombocytopenia",
}
MODEL_ORDER = ["Standard Transformer", "Differential Transformer", "H2O (keep 50%)"]
PALETTE = {"Standard Transformer": "#4C72B0",
           "Differential Transformer": "#DD8452",
           "H2O (keep 50%)": "#55A868"}


# ── Experiment 2: Table 1 ────────────────────────────────────────────────────

def build_table1() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        csv = FIGURES / f"exp2_{task}_results.csv"
        if not csv.exists():
            print(f"  MISSING: {csv.name}")
            continue
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            rows.append({
                "Task": TASK_LABELS.get(task, task),
                "Model": row["model"],
                "val_AUROC": row.get("val_auc", float("nan")),
                "test_AUROC": row.get("test_auc", float("nan")),
                "test_AUPRC": row.get("test_auprc", float("nan")),
            })
    if not rows:
        print("No Exp 2 results found.")
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_exp2(table1: pd.DataFrame) -> None:
    if table1.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 2: Inhibitory Attention on EHRSHOT Lab Prediction Tasks",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, metric, ylabel in zip(axes,
                                   ["test_AUROC", "test_AUPRC"],
                                   ["Test AUROC", "Test AUPRC"]):
        pivot = (table1.pivot_table(index="Task", columns="Model",
                                    values=metric, aggfunc="mean")
                       .reindex(columns=MODEL_ORDER, fill_value=float("nan")))
        x = np.arange(len(pivot))
        w = 0.25
        for i, model in enumerate(MODEL_ORDER):
            if model not in pivot.columns:
                continue
            bars = ax.bar(x + i * w, pivot[model], w, label=model,
                          color=PALETTE[model], edgecolor="white", linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                if not np.isnan(h):
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.005,
                            f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
        ax.set_xticks(x + w)
        ax.set_xticklabels(pivot.index, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=8, framealpha=0.9)
        ax.set_title(ylabel, fontsize=11)

    plt.tight_layout()
    out = FIGURES / "exp2_full_auroc.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Experiment 3 LLM: Table 2 ────────────────────────────────────────────────

def build_table2() -> pd.DataFrame:
    csv = FIGURES / "exp3_llm_inference_results.csv"
    if not csv.exists():
        print(f"  MISSING: {csv.name}")
        return pd.DataFrame()

    df = pd.read_csv(csv)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna(subset=["position"])

    bins = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.01]
    labels = ["0–10%", "10–30%", "30–50%", "50–70%", "70–90%", "90–100%"]
    df["pos_band"] = pd.cut(df["position"], bins=bins, labels=labels,
                             include_lowest=True)

    # Use binary "any credit" (score > 0) as the accuracy metric
    df["baseline_anycredit"] = (df["baseline_correct"] > 0).astype(float)
    df["qccs_anycredit"]     = (df["qccs_correct"] > 0).astype(float)

    agg = (df.groupby("pos_band", observed=False)
             .agg(n=("baseline_anycredit", "count"),
                  baseline_acc=("baseline_anycredit", "mean"),
                  qccs_acc=("qccs_anycredit", "mean"))
             .reset_index())
    agg["improvement_pp"] = (agg["qccs_acc"] - agg["baseline_acc"]) * 100
    agg["baseline_acc"] = (agg["baseline_acc"] * 100).round(1)
    agg["qccs_acc"] = (agg["qccs_acc"] * 100).round(1)
    agg["improvement_pp"] = agg["improvement_pp"].round(1)

    # Aggregate middle vs edge summary
    df["is_middle"] = (df["position"] >= 0.3) & (df["position"] <= 0.7)
    summary = (df.groupby("is_middle")
                 .agg(n=("baseline_anycredit", "count"),
                      baseline_acc=("baseline_anycredit", "mean"),
                      qccs_acc=("qccs_anycredit", "mean"))
                 .reset_index())
    summary["band"] = summary["is_middle"].map({True: "Middle (0.3–0.7)", False: "Edge"})
    summary = summary[["band", "n", "baseline_acc", "qccs_acc"]]
    summary["improvement_pp"] = ((summary["qccs_acc"] - summary["baseline_acc"]) * 100).round(1)
    summary["baseline_acc"] = (summary["baseline_acc"] * 100).round(1)
    summary["qccs_acc"] = (summary["qccs_acc"] * 100).round(1)
    print("\nLLM re-inference summary (middle vs edge):")
    print(summary.to_string(index=False))

    return agg


def plot_exp3_llm(df_raw: pd.DataFrame) -> None:
    if df_raw.empty:
        return

    csv = FIGURES / "exp3_llm_inference_results.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df = df.dropna(subset=["position"])

    bins = np.linspace(0, 1, 11)
    labels = [f"{int(b*100)}–{int(bins[i+1]*100)}%" for i, b in enumerate(bins[:-1])]
    df["decile"] = pd.cut(df["position"], bins=bins, labels=labels,
                           include_lowest=True)

    df["baseline_anycredit"] = (df["baseline_correct"] > 0).astype(float)
    df["qccs_anycredit"]     = (df["qccs_correct"] > 0).astype(float)
    agg = (df.groupby("decile", observed=False)
             .agg(baseline=("baseline_anycredit", "mean"),
                  qccs=("qccs_anycredit", "mean"))
             .reset_index())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(agg))
    ax.plot(x, agg["baseline"] * 100, "o-", color="#4C72B0",
            linewidth=2, markersize=6, label="Baseline (Qwen2.5-7B, full context)")
    ax.plot(x, agg["qccs"] * 100, "s-", color="#DD8452",
            linewidth=2, markersize=6, label="QCCS-compressed context")
    ax.fill_between(x, agg["baseline"] * 100, agg["qccs"] * 100,
                    alpha=0.15, color="#DD8452")
    ax.axvspan(2, 7, alpha=0.08, color="gray", label="Middle region (30–70%)")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["decile"], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("LLM Instruction-Following Accuracy (%)", fontsize=10)
    ax.set_xlabel("Position of Gold-Standard Answer in EHR Timeline", fontsize=10)
    ax.set_title("Experiment 3: LLM Re-inference — Baseline vs QCCS Context\n"
                 "(Qwen2.5-7B-Instruct, MedAlign, N=819 instructions)", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    out = FIGURES / "exp3_llm_ucurve.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Run all ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Building Table 1: Exp 2 EHRSHOT Results ===")
    t1 = build_table1()
    if not t1.empty:
        out1 = FIGURES / "table1_exp2_full.csv"
        t1.to_csv(out1, index=False)
        print(f"Saved: {out1}")
        print(t1.to_string(index=False))
        plot_exp2(t1)

    print("\n=== Building Table 2: Exp 3 LLM Re-inference Results ===")
    t2 = build_table2()
    if not t2.empty:
        out2 = FIGURES / "table2_exp3_llm.csv"
        t2.to_csv(out2, index=False)
        print(f"Saved: {out2}")
        print(t2.to_string(index=False))
        plot_exp3_llm(t2)
