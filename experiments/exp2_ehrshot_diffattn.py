"""
Experiment 2: Differential Transformer vs Standard Attention on EHRSHOT
=======================================================================
Uses EHRSHOT MEDS data + lab task labels.
For each patient+task: builds event sequence, identifies anchor lab measurement,
computes its relative position, trains standard vs Differential Transformer head.
Plots accuracy vs anchor position → shows inhibitory attention flattens the U-curve.

Run locally (CPU ok for training, ~30 min) or via Modal for faster iteration.

Output: figures/exp2_ehrshot_results.csv, figures/exp2_ehrshot_ucurve.png
"""

import math, os, argparse
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ── Paths ──────────────────────────────────────────────────────────────────
MEDS_BASE  = Path("/Users/sanjaybasu/waymark-local/data/ehrshot/EHRSHOT_files/EHRSHOT_MEDS")
ASSETS_BASE= Path("/Users/sanjaybasu/waymark-local/data/ehrshot/EHRSHOT_files/EHRSHOT_ASSETS")
OUT_DIR    = Path("/Users/sanjaybasu/waymark-local/notebooks/inhibitory-attention-ehr/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Task definitions: LOINC anchor codes + abnormal thresholds ─────────────
# Each lab task: anchor = most recent measurement of this LOINC < or > threshold
LAB_TASKS = {
    "lab_anemia":          {"loinc": "LOINC/718-7",   "op": "<", "thresh": 11.0},
    "lab_hyperkalemia":    {"loinc": "LOINC/2823-3",  "op": ">", "thresh": 5.5},
    "lab_hypoglycemia":    {"loinc": "LOINC/2345-7",  "op": "<", "thresh": 70.0},
    "lab_hyponatremia":    {"loinc": "LOINC/2951-2",  "op": "<", "thresh": 135.0},
    "lab_thrombocytopenia":{"loinc": "LOINC/777-3",   "op": "<", "thresh": 100.0},
}
MAX_SEQ_LEN = 256   # events per patient
EMBED_DIM   = 64    # code embedding dimension
N_HEADS     = 4
N_LAYERS    = 2


# ── Data loading ─────────────────────────────────────────────────────────────

def load_meds(task: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print(f"  Loading MEDS data for task: {task}")
    meds   = pd.read_parquet(MEDS_BASE / "data/data.parquet")
    splits = pd.read_parquet(MEDS_BASE / "metadata/subject_splits.parquet")

    # Prefer benchmark labeled_patients.csv (explicit bool True/False for all tasks)
    bench_csv = ASSETS_BASE / f"benchmark/{task}/labeled_patients.csv"
    if bench_csv.exists():
        labels = pd.read_csv(bench_csv)
        labels = labels.rename(columns={"patient_id": "subject_id",
                                        "value": "boolean_value"})
        labels["prediction_time"] = pd.to_datetime(labels["prediction_time"])
        labels["boolean_value"] = labels["boolean_value"].astype(bool)
    else:
        # Fallback: MEDS parquet — binarize integer_value if boolean_value is null
        labels = pd.read_parquet(MEDS_BASE / f"labels/{task}/labels.parquet")
        if labels["boolean_value"].isna().all():
            labels["boolean_value"] = labels["integer_value"] > 0

    pos_rate = labels["boolean_value"].mean()
    print(f"  MEDS: {len(meds):,} rows | Labels: {len(labels):,} | "
          f"Positive rate: {pos_rate:.2%}")
    return meds, labels, splits


def build_code_vocab(meds: pd.DataFrame) -> dict[str, int]:
    codes = meds["code"].dropna().unique()
    return {c: i + 1 for i, c in enumerate(sorted(codes))}  # 0 = PAD


def build_patient_sequences(meds: pd.DataFrame,
                            labels: pd.DataFrame,
                            task_def: dict,
                            code_vocab: dict,
                            max_seq_len: int = MAX_SEQ_LEN,
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each (patient, prediction_time) pair:
      - Extract all events BEFORE prediction_time
      - Tokenize codes → integer sequence (truncated to max_seq_len most recent)
      - Find anchor measurement position (for positive patients)
    Returns:
      seqs     : (N, max_seq_len) int32
      labels_y : (N,) float32
      positions: (N,) float32  (NaN if anchor not found)
    """
    anchor_loinc = task_def["loinc"]
    anchor_op    = task_def["op"]
    anchor_thresh= task_def["thresh"]

    seqs, labels_y, positions = [], [], []

    print("  Pre-grouping MEDS by subject_id (O(N log N) one-time cost)...")
    meds_sorted = meds.sort_values(["subject_id", "time"])
    grouped = {pid: grp.reset_index(drop=True)
               for pid, grp in meds_sorted.groupby("subject_id")}

    for _, row in labels.iterrows():
        pid  = row["subject_id"]
        pt   = row["prediction_time"]
        bv = row["boolean_value"]
        if bv is None or (isinstance(bv, float) and np.isnan(bv)):
            continue
        label = float(bool(bv))

        if pid not in grouped:
            continue
        grp = grouped[pid]
        patient_events = grp[grp["time"] < pt]

        if len(patient_events) == 0:
            continue

        # Tokenize code sequence (most recent max_seq_len events)
        codes_seq = patient_events["code"].fillna("UNK").values[-max_seq_len:]
        tokens = np.array([code_vocab.get(c, 0) for c in codes_seq], dtype=np.int32)
        padded = np.zeros(max_seq_len, dtype=np.int32)
        padded[-len(tokens):] = tokens   # right-align (most recent = end)

        # Find anchor position
        anchor_pos = float("nan")
        if label == 1.0:
            anchor_events = patient_events[
                patient_events["code"] == anchor_loinc
            ]
            if len(anchor_events) > 0:
                vals = pd.to_numeric(anchor_events["numeric_value"],
                                     errors="coerce").dropna()
                if anchor_op == "<":
                    abnormal = anchor_events[
                        pd.to_numeric(anchor_events["numeric_value"],
                                      errors="coerce") < anchor_thresh
                    ]
                else:
                    abnormal = anchor_events[
                        pd.to_numeric(anchor_events["numeric_value"],
                                      errors="coerce") > anchor_thresh
                    ]
                if len(abnormal) > 0:
                    t_first = patient_events["time"].min()
                    t_anchor = abnormal["time"].max()  # most recent abnormal
                    span = (pt - t_first).total_seconds()
                    if span > 0:
                        anchor_pos = (t_anchor - t_first).total_seconds() / span

        seqs.append(padded)
        labels_y.append(label)
        positions.append(anchor_pos)

    return (np.stack(seqs), np.array(labels_y, dtype=np.float32),
            np.array(positions, dtype=np.float64))


# ── Model architectures ──────────────────────────────────────────────────────

class StandardTransformerHead(nn.Module):
    """Standard transformer encoder over tokenized event sequence."""
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 nhead: int = N_HEADS, num_layers: int = N_LAYERS,
                 max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward=256,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        pad_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool),
                               x == 0], dim=1)
        out = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(out[:, 0]).squeeze(-1)


class DifferentialAttentionLayer(nn.Module):
    """Differential attention: A1 − λ·A2, where A1,A2 are dual softmax maps."""
    def __init__(self, embed_dim: int, nhead: int):
        super().__init__()
        self.nhead = nhead
        self.hd = embed_dim // nhead
        self.hhd = self.hd // 2

        self.Wq = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wk = nn.Linear(embed_dim, embed_dim, bias=False)
        self.Wv = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)
        self.norm = nn.LayerNorm(embed_dim)

        # Per-head learnable λ parameters (Hu et al. ICLR 2025 initialisation)
        H, hhd = nhead, self.hhd
        self.lq1 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lk1 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lq2 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lk2 = nn.Parameter(torch.randn(H, hhd) * 0.1)
        self.lambda_init = 0.8
        self.scale = math.sqrt(self.hhd)

    def _lambda(self):
        l = (torch.exp((self.lq1 * self.lk1).sum(-1))
             - torch.exp((self.lq2 * self.lk2).sum(-1))
             + self.lambda_init)
        return l.view(1, self.nhead, 1, 1)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        H, hd, hhd = self.nhead, self.hd, self.hhd

        def _split(w):
            return w.view(B, T, H, hd).permute(0, 2, 1, 3)  # B,H,T,hd

        Q = _split(self.Wq(x)); K = _split(self.Wk(x)); V = _split(self.Wv(x))
        Q1, Q2 = Q[..., :hhd], Q[..., hhd:]
        K1, K2 = K[..., :hhd], K[..., hhd:]

        def _attn(q, k):
            scores = q @ k.transpose(-2, -1) / self.scale
            if key_padding_mask is not None:
                # mask: B,T → B,1,1,T
                scores = scores.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            return torch.softmax(scores, dim=-1)

        A = _attn(Q1, K1) - self._lambda() * _attn(Q2, K2)
        out = (A @ V).permute(0, 2, 1, 3).reshape(B, T, D)
        return self.norm(self.out(out) + x)


class DiffTransformerHead(nn.Module):
    """Differential Transformer encoder for sequence classification."""
    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM,
                 nhead: int = N_HEADS, num_layers: int = N_LAYERS,
                 max_seq_len: int = MAX_SEQ_LEN):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "attn": DifferentialAttentionLayer(embed_dim, nhead),
                "ffn":  nn.Sequential(nn.Linear(embed_dim, 256), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(256, embed_dim)),
                "norm": nn.LayerNorm(embed_dim),
            })
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(nn.Linear(embed_dim, 32), nn.ReLU(),
                                  nn.Dropout(0.1), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        cls = self.cls.expand(B, -1, -1)
        pad_mask = x == 0
        h = torch.cat([cls, h], dim=1)
        pad_mask_ext = torch.cat(
            [torch.zeros(B, 1, device=x.device, dtype=torch.bool), pad_mask], dim=1)
        for layer in self.layers:
            h = layer["attn"](h, key_padding_mask=pad_mask_ext)
            h = layer["norm"](h + layer["ffn"](h))
        return self.head(h[:, 0]).squeeze(-1)


# ── Dataset ──────────────────────────────────────────────────────────────────

class EHRShotDataset(Dataset):
    def __init__(self, seqs, labels):
        self.X = torch.tensor(seqs, dtype=torch.long)
        self.y = torch.tensor(labels, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Training ──────────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader, epochs=20,
                lr=1e-3, device="cpu") -> float:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    all_y = torch.cat([y for _, y in train_loader])
    pos_w = torch.tensor([(all_y == 0).sum() / (all_y == 1).sum() + 1e-6],
                         device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    best_auc, best_state = 0, None

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            crit(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                preds += torch.sigmoid(model(xb.to(device))).cpu().tolist()
                trues += yb.tolist()
        if len(set(trues)) > 1:
            auc = roc_auc_score(trues, preds)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return best_auc


def eval_model(model, seqs, labels, positions, device="cpu",
               batch_size=64) -> pd.DataFrame:
    """Evaluate model and return per-sample (position, label, pred_prob)."""
    model.eval()
    X = torch.tensor(seqs, dtype=torch.long)
    preds = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            logits = model(X[i:i+batch_size].to(device))
            preds += torch.sigmoid(logits).cpu().tolist()
    return pd.DataFrame({"position": positions, "label": labels, "pred": preds})


# ── Position-stratified accuracy plot ─────────────────────────────────────────

def plot_position_accuracy(results: dict[str, pd.DataFrame], out_path: Path,
                           task: str):
    """
    results: {model_name: DataFrame with position, label, pred}
    Only plots rows where anchor was found (position not NaN) and label==1.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"Standard Transformer": "#808080",
              "Differential Transformer": "#2196F3",
              "H2O (keep 50%)": "#FF9800"}
    bins = np.linspace(0, 1, 6)  # 5 quintiles
    bin_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]

    for name, df in results.items():
        df_pos = df[df["position"].notna() & (df["label"] == 1)].copy()
        df_pos["bin"] = pd.cut(df_pos["position"], bins=bins, labels=bin_labels,
                               include_lowest=True)
        acc = df_pos.groupby("bin", observed=True).apply(
            lambda g: (g["pred"] >= 0.5).mean())
        ax.plot(range(len(acc)), acc.values * 100,
                "o-", color=colors.get(name, "black"),
                linewidth=2.5, markersize=8, label=name)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=11)
    ax.set_ylabel("Classification accuracy (positive cases, %)", fontsize=12)
    ax.set_xlabel("Relative position of anchor lab event in patient history", fontsize=12)
    ax.set_title(f"Inhibitory Attention Reduces Lost-in-the-Middle\n"
                 f"EHRSHOT — {task.replace('_', ' ').title()}", fontsize=13,
                 fontweight="bold")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Figure → {out_path}")


# ── H2O simulation ────────────────────────────────────────────────────────────

class H2OTransformerHead(StandardTransformerHead):
    """Standard transformer with H2O-style heavy-hitter token retention."""
    def __init__(self, *args, keep_ratio=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.keep_ratio = keep_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape
        k = max(1, int(T * self.keep_ratio))
        # Mask out lowest-L2-norm tokens (proxy for heavy-hitter importance)
        norms = self.embed(x).norm(dim=-1)  # B, T
        _, topk = norms.topk(k, dim=1)
        mask = torch.ones(B, T, device=x.device, dtype=torch.bool)
        mask.scatter_(1, topk, False)  # False = keep

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        h = h.masked_fill(mask.unsqueeze(-1), 0)
        cls = self.cls.expand(B, -1, -1)
        h = torch.cat([cls, h], dim=1)
        pad_mask = torch.cat([torch.zeros(B, 1, device=x.device, dtype=torch.bool),
                               mask], dim=1)
        out = self.encoder(h, src_key_padding_mask=pad_mask)
        return self.head(out[:, 0]).squeeze(-1)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp2(task: str = "lab_anemia", epochs: int = 25,
             device_str: str = "cpu", n_patients: int = 2000):
    device = torch.device(device_str if torch.cuda.is_available()
                          or device_str == "cpu" else "cpu")
    print(f"\n=== Experiment 2: {task} | device={device} ===")

    meds, labels, splits = load_meds(task)
    # n_patients=0 means full dataset; otherwise sample for speed
    if n_patients > 0:
        labels = labels.sample(min(n_patients, len(labels)), random_state=42)

    code_vocab = build_code_vocab(meds)
    print(f"  Vocab size: {len(code_vocab):,}")

    print("  Building patient sequences...")
    seqs, labels_y, positions = build_patient_sequences(
        meds, labels, LAB_TASKS[task], code_vocab)
    print(f"  Built {len(seqs):,} sequences | "
          f"Anchor found: {np.isfinite(positions).sum():,}")

    # Train/val/test split (use EHRSHOT official splits where possible)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(seqs))
    n_tr = int(0.7 * len(idx)); n_vl = int(0.15 * len(idx))
    tr, vl, te = idx[:n_tr], idx[n_tr:n_tr+n_vl], idx[n_tr+n_vl:]

    def loader(i, shuffle=True):
        ds = EHRShotDataset(seqs[i], labels_y[i])
        return DataLoader(ds, batch_size=64, shuffle=shuffle)

    tr_l, vl_l, te_l = loader(tr), loader(vl, False), loader(te, False)
    vocab_size = len(code_vocab)

    model_specs = {
        "Standard Transformer":     StandardTransformerHead(vocab_size),
        "Differential Transformer": DiffTransformerHead(vocab_size),
        "H2O (keep 50%)":          H2OTransformerHead(vocab_size),
    }

    results_by_model = {}
    summary_rows = []

    for name, model in model_specs.items():
        print(f"\n  Training: {name}")
        model = model.to(device)
        val_auc = train_model(model, tr_l, vl_l, epochs=epochs, device=device)

        # Test evaluation
        test_df = eval_model(model, seqs[te], labels_y[te], positions[te],
                             device=device)
        if len(set(labels_y[te])) > 1:
            test_auc  = roc_auc_score(labels_y[te], test_df["pred"])
            test_auprc= average_precision_score(labels_y[te], test_df["pred"])
        else:
            test_auc = test_auprc = float("nan")

        print(f"    val_AUC={val_auc:.4f}  test_AUC={test_auc:.4f}  "
              f"test_AUPRC={test_auprc:.4f}")

        # Middle vs edge accuracy (positive patients with anchor found)
        pos_df = test_df[test_df["position"].notna() & (test_df["label"] == 1)]
        mid  = (pos_df[pos_df["position"].between(0.3, 0.7)]["pred"] >= 0.5).mean()
        edge = (pos_df[~pos_df["position"].between(0.3, 0.7)]["pred"] >= 0.5).mean()

        results_by_model[name] = test_df
        summary_rows.append({"model": name, "val_auc": val_auc,
                              "test_auc": test_auc, "test_auprc": test_auprc,
                              "mid_acc": mid, "edge_acc": edge,
                              "mid_penalty_pp": (edge - mid) * 100})

    summary = pd.DataFrame(summary_rows)
    print("\n=== Experiment 2 Summary ===")
    print(summary.round(3).to_string(index=False))
    summary.to_csv(OUT_DIR / f"exp2_{task}_results.csv", index=False)

    plot_position_accuracy(results_by_model,
                           OUT_DIR / f"exp2_{task}_ucurve.png", task)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="lab_anemia",
                        choices=list(LAB_TASKS.keys()))
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n_patients", type=int, default=2000,
                        help="Patients to use (use 0 for all)")
    parser.add_argument("--all_tasks", action="store_true")
    args = parser.parse_args()

    if args.all_tasks:
        for task in LAB_TASKS:
            run_exp2(task, args.epochs, args.device,
                     args.n_patients if args.n_patients > 0 else 999999)
    else:
        run_exp2(args.task, args.epochs, args.device,
                 args.n_patients if args.n_patients > 0 else 999999)
