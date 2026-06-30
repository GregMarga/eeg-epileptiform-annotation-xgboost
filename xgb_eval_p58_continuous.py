"""
xgb_eval_p58.py
================
Train XGBoost on all patients EXCEPT P58, evaluate on P58.

Configure the flags below:
  MODE : "handcrafted" | "labram"
  THR  : decision threshold for the positive class
"""

# -------------------------------------------------
# *** CONFIGURE HERE ***
# -------------------------------------------------
MODE = "handcrafted"   # "handcrafted" | "labram"
THR  = 0.5
# -------------------------------------------------

import re
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBClassifier
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

# -------------------------------------------------
# Paths
# -------------------------------------------------
TRAIN_HANDCRAFTED_DIR = Path("../../../data/80hz_freq_time_features_pyprep_1s")
TRAIN_LABRAM_DIR      = Path("../../../data/labram_embeddings_1s_new")

EVAL_HANDCRAFTED_DIR  = Path("../../../data/evaluation_recordings/features_1s_labeled")
EVAL_LABRAM_DIR       = Path("../../../data/evaluation_recordings/labeled")

EVAL_PATIENT_ID = "P58"
IGNORE_LABEL    = -1

FEATURE_NAMES_16 = [
    "zero_cross", "maxima", "minima", "rms", "skew", "kurt_excess",
    "total_power_1_40", "peak_freq_1_40",
    "mean_band_delta", "mean_band_theta", "mean_band_alpha", "mean_band_beta",
    "norm_band_delta", "norm_band_theta", "norm_band_alpha", "norm_band_beta",
]

PATIENT_PATTERN = {
    "P20": "LPD", "P28": "LPD", "P36": "LPD",
    "P48": "GPD", "P49": "GPD", "P54": "GPD",
    "P55": "LRDA", "P58": "LPD", "P70": "GPD", "P73": "LPD",
}
PATTERN_COLORS = {"LPD": "tab:blue", "GPD": "tab:orange", "LRDA": "tab:green"}


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_patient_index(directory: Path, suffix: str) -> dict[str, list[Path]]:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"No *{suffix} files found in {directory}")
    index: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"  [SKIP] could not parse patient id: {f.name}")
            continue
        index.setdefault(pid, []).append(f)
    return index


# -------------------------------------------------
# Loaders
# -------------------------------------------------

def _zscore(X: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std  = X.std(axis=0,  keepdims=True)
    return (X - mean) / (std + eps)


def load_handcrafted_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples  = X.shape[0]
    X = X.reshape(n_samples, n_channels, 16).mean(axis=1)
    y = z["y"].astype(np.int64).ravel()
    keep = y != IGNORE_LABEL
    return X[keep], y[keep].astype(np.uint8)


def load_patient_handcrafted(file_list: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate files then apply per-patient z-score."""
    xs, ys = [], []
    for f in file_list:
        x, y = load_handcrafted_file(f)
        xs.append(x); ys.append(y)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    return _zscore(X), y


def load_labram_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["embeddings"].astype(np.float32)
    y = z["labels"].astype(np.int64).ravel()
    keep = y != IGNORE_LABEL
    return X[keep], y[keep].astype(np.uint8)


def load_patient_labram(file_list: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in file_list:
        x, y = load_labram_file(f)
        xs.append(x); ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# -------------------------------------------------
# XGBoost
# -------------------------------------------------

def train_xgb(X_train: np.ndarray, y_train: np.ndarray, n_features: int):
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
        random_state=42,
    )
    model.fit(X_train, y_train)

    raw_score = model.get_booster().get_score(importance_type="gain")
    imp_vec = np.zeros(n_features)
    for k, v in raw_score.items():
        idx = int(k[1:])
        if idx < n_features:
            imp_vec[idx] = v

    return model, imp_vec


# -------------------------------------------------
# Metrics
# -------------------------------------------------

def safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def safe_pr_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def evaluate(y_true, y_proba, thr: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= thr).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc":       float(accuracy_score(y_true, y_pred)),
        "bacc":      float(balanced_accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   safe_roc_auc(y_true, y_proba),
        "pr_auc":    safe_pr_auc(y_true, y_proba),
        "tp": float(tp), "fp": float(fp),
        "tn": float(tn), "fn": float(fn),
    }


# -------------------------------------------------
# Plot
# -------------------------------------------------

def plot_results(m: dict, train_patients: list[str], mode: str, thr: float):
    mode_label = (
        "Handcrafted features (per-patient z-score)"
        if mode == "handcrafted"
        else "LaBraM embeddings"
    )

    metrics = [
        ("bacc",    "Balanced Accuracy"),
        ("roc_auc", "ROC AUC"),
        ("f1",      "F1"),
        ("pr_auc",  "PR AUC"),
    ]
    values = [m[k] for k, _ in metrics]
    labels = [label for _, label in metrics]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["tab:blue", "tab:orange", "tab:green", "tab:purple"])
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title(
        f"XGBoost — train: {', '.join(train_patients)}\n"
        f"test: {EVAL_PATIENT_ID} ({PATIENT_PATTERN.get(EVAL_PATIENT_ID, '?')}) — {mode_label}",
        fontsize=9,
    )
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    confusion_text = (
        f"TP={int(m['tp'])}  FP={int(m['fp'])}  "
        f"TN={int(m['tn'])}  FN={int(m['fn'])}  |  thr={thr}"
    )
    fig.text(0.5, 0.01, confusion_text, ha="center", fontsize=9, color="dimgray")
    fig.tight_layout(rect=[0, 0.04, 1, 1])

    plot_path = f"results_xgb_eval_p58_{mode}.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.show()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    mode = MODE.strip().lower()
    thr  = THR

    assert mode in ("handcrafted", "labram"), \
        f"MODE must be 'handcrafted' or 'labram', got '{mode}'"

    print(f"\n{'='*60}")
    print(f"  Mode : {mode.upper()}")
    print(f"  Test : {EVAL_PATIENT_ID}  |  threshold : {thr}")
    print(f"{'='*60}")

    # ---- mode-specific setup ----
    if mode == "handcrafted":
        train_dir    = TRAIN_HANDCRAFTED_DIR
        eval_dir     = EVAL_HANDCRAFTED_DIR
        suffix_train = "_features.npz"
        suffix_eval  = "_features.npz"
        load_patient = load_patient_handcrafted
        feature_names = FEATURE_NAMES_16
        n_features    = len(FEATURE_NAMES_16)
    else:
        train_dir    = TRAIN_LABRAM_DIR
        eval_dir     = EVAL_LABRAM_DIR
        suffix_train = "_embeddings.npz"
        suffix_eval  = "_embeddings_labeled.npz"
        load_patient = load_patient_labram
        feature_names = None
        n_features    = None

    # ---- build train index (all patients except P58) ----
    train_index = build_patient_index(train_dir, suffix_train)
    train_index.pop(EVAL_PATIENT_ID, None)
    train_patients = sorted(train_index.keys())
    print(f"Train patients ({len(train_patients)}): {train_patients}")

    # ---- load and concatenate all train data ----
    xs, ys = [], []
    for pid in train_patients:
        x, y = load_patient(train_index[pid])
        xs.append(x); ys.append(y)
        print(f"  {pid}: windows={len(y)}  pos={int(y.sum())}  neg={int((y==0).sum())}")

    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    print(f"\nTotal train: windows={len(y_train)}  pos={int(y_train.sum())}  "
          f"neg={int((y_train==0).sum())}")

    # ---- infer embedding dim for LaBraM ----
    if n_features is None:
        n_features    = X_train.shape[1]
        feature_names = [f"f{i}" for i in range(n_features)]
        print(f"LaBraM embedding dim: {n_features}")

    # ---- load eval (P58) ----
    eval_index = build_patient_index(eval_dir, suffix_eval)
    if EVAL_PATIENT_ID not in eval_index:
        raise RuntimeError(
            f"Eval patient {EVAL_PATIENT_ID} not found in {eval_dir} "
            f"(suffix={suffix_eval})"
        )
    X_eval, y_eval = load_patient(eval_index[EVAL_PATIENT_ID])
    print(f"\nTest  ({EVAL_PATIENT_ID}): windows={len(y_eval)}  "
          f"pos={int(y_eval.sum())}  neg={int((y_eval==0).sum())}")

    # ---- train ----
    print(f"\nTraining XGBoost...")
    model, imp_vec = train_xgb(X_train, y_train, n_features)

    # ---- evaluate ----
    y_proba = model.predict_proba(X_eval)[:, 1]
    m = evaluate(y_eval, y_proba, thr=thr)

    print(f"\n=== Results on {EVAL_PATIENT_ID} ===")
    for k in ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]:
        print(f"  {k:>10}: {m[k]:.4f}")
    print(f"\n  TP={int(m['tp'])}  FP={int(m['fp'])}  "
          f"TN={int(m['tn'])}  FN={int(m['fn'])}")

    # ---- feature importance (handcrafted only) ----
    if mode == "handcrafted":
        print(f"\n=== Feature importance (gain) ===")
        order = np.argsort(imp_vec)[::-1]
        for i in order:
            print(f"  {feature_names[i]:20s} {imp_vec[i]:.4f}")

    # ---- CSV ----
    csv_path = f"results_xgb_eval_p58_{mode}.csv"
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc",
            "tp", "fp", "tn", "fn"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerow({k: m[k] for k in keys})
    print(f"\nResults saved to: {csv_path}")

    # ---- plot ----
    plot_results(m, train_patients, mode, thr)


if __name__ == "__main__":
    main()