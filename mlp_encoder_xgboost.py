import re
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from xgboost import XGBClassifier

# =================================================
# Config  (ίδιο με το encoder script)
# =================================================
IN_DIR     = Path("../../../data/labram_classification_1s")
FILE_GLOB  = "*.npz"

IN_DIM  = 200
HIDDEN  = (256, 128)
OUT_DIM = 64
DROPOUT = 0.3

LOSS        = "supcon"
TEMPERATURE = 0.1
MARGIN      = 0.3

EPOCHS     = 80
BATCH_SIZE = 128
LR         = 1e-3
WEIGHT_DECAY = 1e-4

EVAL_THRESHOLD = 0.5

FEATURE_KEYS = ("embeddings", "X", "features", "emb")
LABEL_KEYS   = ("labels", "y", "label")

SEED   = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CSV_OUTPUT = Path("results_mlp_encoder_xgb.csv")

# =================================================
# Data loading  (ίδιο)
# =================================================
def patient_from_filename(name: str):
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def _first_present(candidates, available):
    for k in candidates:
        if k in available:
            return k
    return None

def load_npz(path: Path):
    z = np.load(path, allow_pickle=True)
    keys = set(z.files)
    xk = _first_present(FEATURE_KEYS, keys)
    yk = _first_present(LABEL_KEYS, keys)
    if xk is None or yk is None:
        raise KeyError(f"{path.name}: no feature/label keys. Found {sorted(keys)}")
    X = np.asarray(z[xk]).astype(np.float32)
    y = np.asarray(z[yk]).astype(np.int64).ravel()
    return X, y

def build_patient_data(in_dir: Path):
    files = sorted(in_dir.glob(FILE_GLOB))
    if not files:
        raise RuntimeError(f"No {FILE_GLOB} in {in_dir}")
    per_patient = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        X, y = load_npz(f)
        per_patient.setdefault(pid, []).append((X, y))
    data = {}
    for pid, chunks in per_patient.items():
        X = np.concatenate([c[0] for c in chunks], axis=0)
        y = np.concatenate([c[1] for c in chunks], axis=0)
        data[pid] = (X, y)
    return data

# =================================================
# Encoder  (ίδιο)
# =================================================
class MLPEncoder(nn.Module):
    def __init__(self, in_dim=IN_DIM, hidden=HIDDEN, out_dim=OUT_DIM, dropout=DROPOUT):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [
                nn.Linear(a, b),
                nn.BatchNorm1d(b),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], out_dim)

    def forward(self, x):
        z = self.head(self.backbone(x))
        return F.normalize(z, dim=1)

# =================================================
# Losses  (ίδιο)
# =================================================
def supcon_loss(emb, labels, temperature=TEMPERATURE):
    device = emb.device
    b = emb.shape[0]
    sim = (emb @ emb.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True)[0].detach()
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    self_mask = torch.eye(b, device=device)
    pos_mask = pos_mask - self_mask
    exp_sim = torch.exp(sim) * (1 - self_mask)
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)
    pos_count = pos_mask.sum(dim=1)
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_count.clamp(min=1.0)
    has_pos = pos_count > 0
    return -(mean_log_prob_pos * has_pos).sum() / has_pos.sum().clamp(min=1.0)

def batch_hard_triplet_loss(emb, labels, margin=MARGIN):
    device = emb.device
    dist = torch.cdist(emb, emb)
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T)
    eye = torch.eye(len(emb), dtype=torch.bool, device=device)
    pos_mask_noself = pos_mask & ~eye
    neg_mask = ~pos_mask
    hardest_pos = dist.masked_fill(~pos_mask_noself, float("-inf")).max(dim=1)[0]
    hardest_neg = dist.masked_fill(~neg_mask, float("inf")).min(dim=1)[0]
    valid = torch.isfinite(hardest_pos) & torch.isfinite(hardest_neg)
    loss = torch.clamp(hardest_pos - hardest_neg + margin, min=0.0)[valid]
    if loss.numel() == 0:
        return torch.zeros((), device=device, requires_grad=True)
    return loss.mean()

def compute_loss(emb, labels):
    if LOSS == "supcon":
        return supcon_loss(emb, labels)
    elif LOSS == "triplet":
        return batch_hard_triplet_loss(emb, labels)
    raise ValueError(f"Unknown LOSS: {LOSS}")

def balanced_batches(y, batch_size, n_batches, rng):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    half = batch_size // 2
    for _ in range(n_batches):
        p = rng.choice(pos, size=half, replace=len(pos) < half)
        n = rng.choice(neg, size=half, replace=len(neg) < half)
        idx = np.concatenate([p, n])
        rng.shuffle(idx)
        yield idx

# =================================================
# Train encoder  (ίδιο)
# =================================================
def train_encoder(X_train, y_train, seed=SEED):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    scaler = StandardScaler().fit(X_train)
    Xs = scaler.transform(X_train).astype(np.float32)
    Xs_t = torch.from_numpy(Xs).to(DEVICE)
    y_t  = torch.from_numpy(y_train.astype(np.int64)).to(DEVICE)

    enc = MLPEncoder().to(DEVICE)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    n_batches = max(1, len(y_train) // BATCH_SIZE)
    enc.train()
    for epoch in range(EPOCHS):
        ep_loss = 0.0
        for idx in balanced_batches(y_train, BATCH_SIZE, n_batches, rng):
            idx_t = torch.from_numpy(idx).to(DEVICE)
            emb = enc(Xs_t[idx_t])
            loss = compute_loss(emb, y_t[idx_t])
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}  loss={ep_loss/n_batches:.4f}")

    return enc, scaler

@torch.no_grad()
def encode(enc, scaler, X):
    enc.eval()
    Xs = scaler.transform(X).astype(np.float32)
    z = enc(torch.from_numpy(Xs).to(DEVICE))
    return z.cpu().numpy()

# =================================================
# XGBoost  (ίδιο config με το baseline script σου)
# =================================================
def train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0
    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model

# =================================================
# Metrics
# =================================================
def safe_auc(fn, y, p):
    return float(fn(y, p)) if len(np.unique(y)) == 2 else float("nan")

def evaluate(y_true, y_proba, thr=EVAL_THRESHOLD):
    y_pred = (y_proba >= thr).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc":       float(accuracy_score(y_true, y_pred)),
        "bacc":      float(balanced_accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   safe_auc(roc_auc_score, y_true, y_proba),
        "pr_auc":    safe_auc(average_precision_score, y_true, y_proba),
        "tp": float(tp), "fp": float(fp),
        "tn": float(tn), "fn": float(fn),
    }

def summarize(values):
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))

# =================================================
# LOPO
# =================================================
def main():
    data = build_patient_data(IN_DIR)
    patients = sorted(data.keys())
    print(f"Patients: {patients} (n={len(patients)})")
    print(f"Loss={LOSS}  in_dim={IN_DIM} -> enc_dim={OUT_DIM}  device={DEVICE}\n")

    metric_keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected   = {k: [] for k in metric_keys}
    agg         = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    fold_rows   = []

    for test_pid in patients:
        print(f"[fold] held-out = {test_pid}")
        train_pids = [p for p in patients if p != test_pid]

        X_train_raw = np.concatenate([data[p][0] for p in train_pids], axis=0)
        y_train     = np.concatenate([data[p][1] for p in train_pids], axis=0)
        X_test_raw, y_test = data[test_pid]

        # --- 1. Εκπαίδευση encoder (scaler fit μόνο στους train) ---
        enc, scaler = train_encoder(X_train_raw, y_train)

        # --- 2. Encode train + test ---
        Z_train = encode(enc, scaler, X_train_raw)   # (N_train, 64)
        Z_test  = encode(enc, scaler, X_test_raw)    # (N_test,  64)

        # --- 3. XGBoost στον encoded χώρο ---
        xgb = train_xgb(Z_train, y_train)
        y_proba = xgb.predict_proba(Z_test)[:, 1]

        m = evaluate(y_test, y_proba)
        fold_rows.append({"patient": test_pid, **m})

        for k in metric_keys:
            collected[k].append(m[k])
        for c in ["tp", "fp", "tn", "fn"]:
            agg[c] += m[c]

        print(
            f"  n={len(y_test)} pos={int(y_test.sum())} | "
            f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
            f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}\n"
        )

    # ---- Summary ----
    print("=== LOPO summary (encoder → XGBoost, mean ± std) ===")
    summary_row = {"patient": "MEAN±STD"}
    for k in metric_keys:
        mean, std = summarize(collected[k])
        summary_row[k] = mean
        print(f"  {k:>10}: {mean:.4f} ± {std:.4f}")

    print(f"\n=== Aggregate confusion (thr={EVAL_THRESHOLD}) ===")
    print(f"  TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

    worst = sorted(fold_rows, key=lambda r: r["f1"])[:3]
    print("\nWorst 3 folds by F1:")
    for r in worst:
        print(f"  {r['patient']}: f1={r['f1']:.3f} (prec={r['precision']:.3f}, rec={r['recall']:.3f})")

    # ---- CSV ----
    all_rows = fold_rows + [summary_row]
    with open(CSV_OUTPUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fold_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    print(f"\nSaved: {CSV_OUTPUT}")


if __name__ == "__main__":
    main()