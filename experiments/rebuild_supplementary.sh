#!/usr/bin/env bash
# rebuild_supplementary.sh — Rebuild supplementary.zip after experiments complete.
#
# Usage: bash experiments/rebuild_supplementary.sh
# Run from the inhibitory-attention-ehr root directory.

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUB_DIR="$REPO_ROOT/neurips2026_submission"
EXP_DIR="$REPO_ROOT/experiments"
FIG_DIR="$REPO_ROOT/figures"
ZIP="$SUB_DIR/inhibitory_attention_clitm_ehr_neurips2026_supplementary.zip"

echo "=== Rebuilding supplementary.zip ==="
echo "Root: $REPO_ROOT"

# ── Verify experiments complete ──────────────────────────────────────────────
python3 -c "
import pandas as pd, sys
issues = []
for f in ['exp3_v5_llm_results_judged.csv', 'exp3_llmlingua2_results_judged.csv']:
    import os
    if not os.path.exists('figures/' + f):
        issues.append(f'MISSING: figures/{f}')
if issues:
    print('ABORT: missing files:'); [print(' -', i) for i in issues]; sys.exit(1)
# Check v5 has real responses
df = pd.read_csv('figures/exp3_v5_llm_results_judged.csv')
resp = df['baseline_response'].astype(str)
ne = (resp != 'nan') & (resp.str.strip() != '')
if ne.sum() < 80:
    print(f'ABORT: v5 only has {ne.sum()} non-empty responses (expected 83)'); sys.exit(1)
# Check llmlingua2 has real responses
df2 = pd.read_csv('figures/exp3_llmlingua2_results_judged.csv')
resp2 = df2['llmlingua2_response'].astype(str)
ne2 = (resp2 != 'nan') & (resp2.str.strip() != '')
if ne2.sum() < 75:
    print(f'ABORT: llmlingua2 only has {ne2.sum()} non-empty responses (expected 83)'); sys.exit(1)
print(f'Checks passed: v5 {ne.sum()}/83 non-empty, llmlingua2 {ne2.sum()}/83 non-empty')
"

# ── Build zip ────────────────────────────────────────────────────────────────
rm -f "$ZIP"
cd "$REPO_ROOT"

zip -j "$ZIP" \
  experiments/exp1_clustered_bootstrap.py \
  experiments/exp1_litm_characterization.py \
  experiments/exp2_ehrshot_diffattn.py \
  experiments/exp2_qccs_diffattn.py \
  experiments/exp2_qccs_diffattn_focal.py \
  experiments/exp2_sparse_attn.py \
  experiments/exp3_extended_baselines.py \
  experiments/exp3_gate_ablations.py \
  experiments/exp3_llm_judge.py \
  experiments/exp3_nli_hit.py \
  experiments/exp3_nooracle_judge.py \
  experiments/exp3_qualitative_failures.py \
  experiments/exp3_oracle_control.py \
  experiments/exp3_qccs_gate.py \
  experiments/exp3_second_judge.py \
  experiments/fill_pending_tables.py \
  experiments/modal_app.py \
  figures/exp1_litm_ucurve_clustered_ci.csv \
  figures/exp2_auroc_with_ci.csv \
  figures/exp2_qccs_diffattn_results.csv \
  figures/exp2b_qccs_diffattn_ci.csv \
  figures/exp3_extended_stage1.csv \
  figures/exp3_kappa_summary.csv \
  figures/exp3_nli_hits.csv \
  figures/exp3_nooracle_judge_results.csv \
  figures/exp3_oracle_contexts.csv \
  figures/exp3_precision_k.csv \
  figures/exp3_qualitative_failures.csv \
  figures/exp3_second_judge_results.csv \
  figures/exp3_v4_llm_results_judged.csv \
  figures/exp3_v5_llm_results_judged.csv \
  figures/exp3_llmlingua2_results_judged.csv \
  figures/exp3_mapreduce_judged.csv \
  figures/exp3_failure_summary.txt

# Add post-submission results if available
for f in figures/exp3_dosrag_mmr_results_judged.csv figures/exp3_nli_thresh_sweep.csv \
         figures/exp3_gate_ablations.csv figures/exp2_sparse_attn_results.csv; do
  if [ -f "$REPO_ROOT/$f" ]; then
    zip -j "$ZIP" "$REPO_ROOT/$f"
    echo "  Added: $f"
  fi
done

echo ""
echo "Created: $ZIP"
unzip -l "$ZIP" | awk '/files$/{print $NF, "files,", $(NF-2), "bytes total"}' || true
echo ""
echo "=== Next: recompile PDF ==="
echo "  cd neurips2026_submission && latexmk -pdf inhibitory_attention_clitm_ehr_neurips2026.tex"
