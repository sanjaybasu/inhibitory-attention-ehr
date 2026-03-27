"""
Experiment 1: Lost-in-the-Middle Analysis of MedAlign
======================================================
Uses existing binary_correct scores from clinician-reviewed-model-responses.tsv.
For each instruction, locates the answer evidence in the patient XML EHR,
computes its relative temporal position, and plots accuracy vs. position.

NO MODEL TRAINING REQUIRED — pure analysis of existing evaluation data.

Output: figures/exp1_litm_ucurve.png, figures/exp1_litm_results.csv
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from difflib import SequenceMatcher
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path("/Users/sanjaybasu/waymark-local/data/medalign/MedAlign_files")
TSV_RESPONSES = BASE / "medalign_instructions_v1_3/clinician-reviewed-model-responses.tsv"
TSV_INSTRUCTIONS = BASE / "medalign_instructions_v1_3/clinician-instruction-responses.tsv"
EHR_DIR = BASE / "medalign_instructions_v1_3/ehrs"
OUT_DIR = Path("/Users/sanjaybasu/waymark-local/notebooks/inhibitory-attention-ehr/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── XML parser ──────────────────────────────────────────────────────────────

def parse_ehr_xml(xml_path: Path) -> list[dict]:
    """
    Parse MedAlign XML into a list of events sorted by timestamp.
    Each event: {timestamp, type, name, text, note_id}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()  # <eventstream>
    events = []

    for encounter in root.findall("encounter"):
        for entry in encounter.findall(".//entry"):
            ts_str = entry.get("timestamp", "")
            try:
                ts = pd.Timestamp(ts_str)
            except Exception:
                continue

            for event in entry.findall("event"):
                etype = event.get("type", "")
                name  = event.get("name", "")
                nid   = event.get("note_id", "")
                text  = (event.text or "").strip()
                events.append({
                    "timestamp": ts,
                    "type": etype,
                    "name": name,
                    "note_id": nid,
                    "text": text,
                })

    events.sort(key=lambda e: e["timestamp"])
    return events


def compute_evidence_position(events: list[dict], evidence: str) -> float | None:
    """
    Find which event best matches the evidence phrase and return its
    relative temporal position (0 = first event, 1 = last event).
    Returns None if evidence cannot be located.
    """
    if not evidence or not events:
        return None

    evidence_lower = evidence.lower().strip()
    # Tokenize evidence for fuzzy matching
    ev_tokens = set(re.findall(r'\w+', evidence_lower))

    best_score = 0.0
    best_idx = None

    for i, ev in enumerate(events):
        candidate = (ev["text"] + " " + ev["name"]).lower()
        # Token overlap score
        cand_tokens = set(re.findall(r'\w+', candidate))
        if not cand_tokens:
            continue
        overlap = len(ev_tokens & cand_tokens) / len(ev_tokens) if ev_tokens else 0
        # Sequence matcher for substring match
        sm = SequenceMatcher(None, evidence_lower[:200], candidate[:500],
                             autojunk=False).ratio()
        score = 0.6 * overlap + 0.4 * sm
        if score > best_score:
            best_score = score
            best_idx = i

    if best_idx is None or best_score < 0.2:
        return None

    return best_idx / max(len(events) - 1, 1)


# ── Main analysis ────────────────────────────────────────────────────────────

def run_exp1():
    print("Loading MedAlign evaluation data...")
    df = pd.read_csv(TSV_RESPONSES, sep="\t", low_memory=False)
    print(f"  Loaded {len(df):,} rows, {df['filename'].nunique()} patients")
    print(f"  Models: {df['model_name'].unique()}")
    print(f"  binary_correct values: {df['binary_correct'].value_counts().to_dict()}")

    # Focus on eval-eligible rows
    df = df[df["is_used_eval"].astype(str).str.lower().isin(["true", "yes", "1"])]
    print(f"  After is_used_eval filter: {len(df):,} rows")

    # Parse each patient XML once (cache)
    print("\nParsing patient XML files...")
    xml_cache: dict[str, list] = {}
    missing = []
    for fname in df["filename"].unique():
        xml_path = EHR_DIR / fname
        if xml_path.exists():
            xml_cache[fname] = parse_ehr_xml(xml_path)
        else:
            missing.append(fname)
    print(f"  Parsed {len(xml_cache)} XMLs, {len(missing)} missing")

    # Compute evidence position for each row
    print("\nComputing evidence positions...")
    positions = []
    for _, row in df.iterrows():
        fname = row["filename"]
        evidence = str(row.get("evidence", ""))
        events = xml_cache.get(fname, [])
        pos = compute_evidence_position(events, evidence)
        positions.append(pos)

    df["position"] = positions
    df["binary_correct_num"] = pd.to_numeric(df["binary_correct"], errors="coerce")

    located = df["position"].notna().sum()
    print(f"  Evidence located in {located}/{len(df)} rows "
          f"({100*located//len(df)}%)")

    df_located = df[df["position"].notna()].copy()

    # Save raw results
    out_csv = OUT_DIR / "exp1_litm_results.csv"
    df_located[["filename", "model_name", "question", "evidence",
                "binary_correct_num", "position",
                "submitter_specialty"]].to_csv(out_csv, index=False)
    print(f"  Saved raw results → {out_csv}")

    # ── Figures ──────────────────────────────────────────────────────────
    plot_ucurve(df_located, OUT_DIR / "exp1_litm_ucurve.png")
    plot_by_model(df_located, OUT_DIR / "exp1_litm_by_model.png")
    plot_by_specialty(df_located, OUT_DIR / "exp1_litm_by_specialty.png")
    print_summary(df_located)


def plot_ucurve(df: pd.DataFrame, out_path: Path):
    """Main U-curve: accuracy vs. position decile, all models combined."""
    df = df.copy()
    df["decile"] = pd.cut(df["position"], bins=10,
                          labels=[f"{i*10}–{i*10+10}%" for i in range(10)])

    agg = (df.groupby("decile", observed=True)["binary_correct_num"]
             .agg(["mean", "sem", "count"])
             .reset_index())
    agg.columns = ["decile", "accuracy", "sem", "n"]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(agg))
    ax.bar(x, agg["accuracy"] * 100, color="#4A90D9", alpha=0.85,
           yerr=agg["sem"] * 1.96 * 100, capsize=4, error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(agg["decile"], rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Clinician-verified retrieval accuracy (%)", fontsize=12)
    ax.set_xlabel("Position of answer in patient EHR timeline", fontsize=12)
    ax.set_title("Lost-in-the-Middle in Clinical EHR Instruction-Following\n"
                 f"MedAlign (n={len(df):,} instructions, {df['filename'].nunique()} patients)",
                 fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.axhspan(0, df["binary_correct_num"].mean() * 100,
               alpha=0.08, color="gray", label="Overall mean")
    ax.grid(axis="y", alpha=0.3)

    # Annotate middle dip
    mid_mask = (df["position"] >= 0.3) & (df["position"] <= 0.7)
    edge_mask = (df["position"] < 0.2) | (df["position"] > 0.8)
    mid_acc  = df.loc[mid_mask, "binary_correct_num"].mean()
    edge_acc = df.loc[edge_mask, "binary_correct_num"].mean()
    if not np.isnan(mid_acc) and not np.isnan(edge_acc):
        ax.text(0.5, 0.05,
                f"Middle penalty: −{(edge_acc - mid_acc)*100:.1f} pp\n"
                f"(edge {edge_acc*100:.1f}% vs. middle {mid_acc*100:.1f}%)",
                transform=ax.transAxes, ha="center", fontsize=10,
                color="#c0392b", bbox=dict(boxstyle="round,pad=0.3",
                facecolor="white", edgecolor="#c0392b", alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out_path}")


def plot_by_model(df: pd.DataFrame, out_path: Path):
    """U-curve by model — shows effect is consistent across GPT-4, others."""
    models = df["model_name"].dropna().unique()
    fig, axes = plt.subplots(1, min(len(models), 4), figsize=(14, 4),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models[:4]):
        sub = df[df["model_name"] == model].copy()
        sub["decile"] = pd.cut(sub["position"], bins=5, labels=False)
        agg = sub.groupby("decile")["binary_correct_num"].mean()
        ax.plot(agg.index, agg.values * 100, "o-", linewidth=2, markersize=6)
        ax.set_title(model[:30], fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Position quintile")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    plt.suptitle("LitM Effect by Model (MedAlign)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out_path}")


def plot_by_specialty(df: pd.DataFrame, out_path: Path):
    """Accuracy vs position by submitter specialty."""
    top_specs = df["submitter_specialty"].value_counts().head(4).index
    fig, axes = plt.subplots(1, min(len(top_specs), 4), figsize=(14, 4),
                             sharey=True)
    if len(top_specs) == 1:
        axes = [axes]

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0"]
    for ax, spec, color in zip(axes, top_specs, colors):
        sub = df[df["submitter_specialty"] == spec].copy()
        if len(sub) < 10:
            continue
        sub["decile"] = pd.cut(sub["position"], bins=5, labels=False)
        agg = sub.groupby("decile")["binary_correct_num"].mean()
        ax.plot(agg.index, agg.values * 100, "o-", color=color,
                linewidth=2, markersize=6)
        ax.set_title(f"{spec}\n(n={len(sub)})", fontsize=9)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Position quintile")
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Accuracy (%)")
    plt.suptitle("LitM Effect by Clinical Specialty (MedAlign)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure saved → {out_path}")


def print_summary(df: pd.DataFrame):
    df["decile_num"] = (df["position"] * 10).astype(int).clip(0, 9)
    agg = df.groupby("decile_num")["binary_correct_num"].agg(["mean","sem","count"])
    print("\n=== Experiment 1 Summary ===")
    print(agg.round(3).to_string())
    mid  = df[df["position"].between(0.3, 0.7)]["binary_correct_num"].mean()
    edge = df[~df["position"].between(0.3, 0.7)]["binary_correct_num"].mean()
    print(f"\nEdge accuracy (outer 40%): {edge*100:.1f}%")
    print(f"Middle accuracy (middle 40%): {mid*100:.1f}%")
    print(f"Middle penalty: -{(edge-mid)*100:.1f} pp")


if __name__ == "__main__":
    run_exp1()
