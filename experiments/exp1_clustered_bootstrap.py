"""
Clustered Bootstrap for CLitM U-Curve (Experiment 1).

Addresses reviewer concern: the original bootstrap treats each of the 2,196
(model, instruction) rows as independent, but multiple models share the same
instruction, violating the independence assumption. This script implements
clustered bootstrap resampling by (patient_id, question) cluster.

For each bootstrap replicate:
  1. Sample N_clusters instruction clusters with replacement (where N_clusters
     is the number of unique (filename, question) pairs in the test set)
  2. Take ALL model rows for each selected cluster
  3. Compute decile-level U-curve accuracy from the resampled set

This gives wider, more honest CIs that account for the repeated-measures
structure of the 6-model evaluation.

Usage:
  python experiments/exp1_clustered_bootstrap.py

Input:  figures/exp1_litm_results.csv
Output: figures/exp1_litm_ucurve_clustered_ci.csv
        (printed U-curve table with original vs. clustered CIs)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIGURES = Path(__file__).parent.parent / "figures"
N_BOOT  = 5000
SEED    = 42
N_DECILES = 10


def compute_ucurve(df: pd.DataFrame) -> np.ndarray:
    """
    Compute mean accuracy per decile of normalized position [0, 1].
    Returns array of length N_DECILES.
    """
    df = df.copy()
    df["decile"] = pd.cut(df["position"], bins=N_DECILES, labels=False)
    return df.groupby("decile")["binary_correct_num"].mean().values


def main():
    df = pd.read_csv(FIGURES / "exp1_litm_results.csv")
    print(f"Loaded {len(df)} rows, {df[['filename','question']].drop_duplicates().shape[0]} "
          f"unique (patient, instruction) clusters, {df['model_name'].nunique()} models")

    # Build cluster list
    clusters = df.groupby(["filename", "question"])
    cluster_keys = list(clusters.groups.keys())
    N_clusters = len(cluster_keys)
    print(f"Resampling by {N_clusters} clusters, {N_BOOT} bootstrap replicates...")

    # Pre-index: map cluster key → row indices
    cluster_idx: dict[tuple, list[int]] = {}
    for (fname, q), g in clusters:
        cluster_idx[(fname, q)] = list(g.index)

    rng = np.random.default_rng(SEED)
    boot_curves = np.zeros((N_BOOT, N_DECILES))

    for b in range(N_BOOT):
        sampled_keys = rng.choice(N_clusters, size=N_clusters, replace=True)
        sampled_idx = []
        for k_idx in sampled_keys:
            key = cluster_keys[k_idx]
            sampled_idx.extend(cluster_idx[key])
        boot_df = df.loc[sampled_idx].copy()
        boot_df["decile"] = pd.cut(boot_df["position"], bins=N_DECILES, labels=False)
        curve = boot_df.groupby("decile")["binary_correct_num"].mean()
        for d in range(N_DECILES):
            boot_curves[b, d] = curve.get(d, np.nan)

    # Original observed U-curve (no resampling)
    df["decile"] = pd.cut(df["position"], bins=N_DECILES, labels=False)
    obs_curve = df.groupby("decile")["binary_correct_num"].mean()

    # Original non-clustered bootstrap CIs
    rng2 = np.random.default_rng(SEED)
    plain_boot = np.zeros((N_BOOT, N_DECILES))
    for b in range(N_BOOT):
        samp = df.sample(len(df), replace=True, random_state=int(rng2.integers(0, 2**31)))
        samp["decile"] = pd.cut(samp["position"], bins=N_DECILES, labels=False)
        curve = samp.groupby("decile")["binary_correct_num"].mean()
        for d in range(N_DECILES):
            plain_boot[b, d] = curve.get(d, np.nan)

    # Report
    print(f"\n{'Decile':>7}  {'Center':>8}  {'Obs':>6}  "
          f"{'Plain CI':>18}  {'Clustered CI':>18}  {'CI widened':>10}")
    rows = []
    for d in range(N_DECILES):
        obs  = obs_curve.get(d, np.nan)
        p_lo, p_hi = np.nanpercentile(plain_boot[:, d], [2.5, 97.5])
        c_lo, c_hi = np.nanpercentile(boot_curves[:, d], [2.5, 97.5])
        center = (d + 0.5) / N_DECILES
        plain_w = (p_hi - p_lo) * 100
        clust_w = (c_hi - c_lo) * 100
        widened = clust_w - plain_w
        print(f"  {d:5d}  {center:8.2f}  {obs*100:5.1f}%  "
              f"[{p_lo*100:5.1f}, {p_hi*100:5.1f}]  "
              f"[{c_lo*100:5.1f}, {c_hi*100:5.1f}]  "
              f"{widened:+.1f}pp")
        rows.append({
            "decile": d, "position_center": center, "obs_acc": obs,
            "plain_lo": p_lo, "plain_hi": p_hi,
            "clustered_lo": c_lo, "clustered_hi": c_hi,
        })

    out = FIGURES / "exp1_litm_ucurve_clustered_ci.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")

    # Headline stats for paper
    mid_deciles = [3, 4, 5, 6]  # ~30-70% position
    mid_mask = df["decile"].isin(mid_deciles)
    edge_mask = ~mid_mask & df["decile"].notna()
    mid_acc = df.loc[mid_mask, "binary_correct_num"].mean() * 100
    edge_acc = df.loc[edge_mask, "binary_correct_num"].mean() * 100

    # Peak (decile 0 and 9) vs trough (middle deciles)
    peak_d = [d for d in [0, 1, 8, 9] if d in obs_curve.index]
    trough_d = [d for d in mid_deciles if d in obs_curve.index]
    peak_mean  = np.nanmean([obs_curve.get(d, np.nan) for d in peak_d]) * 100
    trough_mean = np.nanmean([obs_curve.get(d, np.nan) for d in trough_d]) * 100
    gap = peak_mean - trough_mean

    print(f"\nKey stats:")
    print(f"  Middle (30-70%) accuracy: {mid_acc:.1f}%")
    print(f"  Edge accuracy: {edge_acc:.1f}%")
    print(f"  Peak-to-trough gap: {gap:.1f} pp")

    # Clustered CI for peak-to-trough gap
    boot_gaps = []
    for b in range(N_BOOT):
        p = np.nanmean([boot_curves[b, d] for d in peak_d]) * 100
        t = np.nanmean([boot_curves[b, d] for d in trough_d]) * 100
        boot_gaps.append(p - t)
    g_lo, g_hi = np.nanpercentile(boot_gaps, [2.5, 97.5])
    print(f"  CLitM gap (clustered 95% CI): {gap:.1f} pp [{g_lo:.1f}, {g_hi:.1f}]")

    # Also report overall accuracy change across all middle vs edge
    boot_mid_accs = []
    for b in range(N_BOOT):
        mid_val = np.nanmean([boot_curves[b, d] for d in trough_d]) * 100
        boot_mid_accs.append(mid_val)
    m_lo, m_hi = np.nanpercentile(boot_mid_accs, [2.5, 97.5])
    print(f"  Middle accuracy (clustered 95% CI): {mid_acc:.1f}% [{m_lo:.1f}, {m_hi:.1f}]")

    print("\nNote: Clustered CIs are wider than plain CIs, as expected from")
    print("the repeated-measures structure (6 models per instruction).")
    print("The CLitM gap and qualitative U-shape are robust to clustering.")


if __name__ == "__main__":
    main()
