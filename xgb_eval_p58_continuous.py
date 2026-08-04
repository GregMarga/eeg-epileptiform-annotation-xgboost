"""
xgb_eval_p58.py
================
Train XGBoost on all patients EXCEPT P58, evaluate on P58.

Runs BOTH modes ("handcrafted" and "labram") in a single execution and plots
the results as ONE grouped bar chart so the two representations can be compared
directly: for each metric, the handcrafted bar (solid) sits next to the LaBraM
bar (hatched). Colour encodes the metric; fill vs hatch encodes representation.

  THR  : decision threshold for the positive class
"""

# -------------------------------------------------
# *** CONFIGURE HERE ***
# -------------------------------------------------
THR = 0.5
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

# EVAL_HANDCRAFTED_DIR is the OUT_DIR written by extract_features_from_raw.py
# (preprocess -> PD/NON_PD-segment-restricted sliding windows -> 16-dim
# handcrafted features per channel, same layout as the training features:
# X is (n_windows, n_channels*16), channel-major, 16 feats per channel).
# No loader changes are needed — the format is identical to before.
EVAL_HANDCRAFTED_DIR = Path("../../../data/evaluation_recordings/features_1s_labeled_persegment")
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
# Mode-specific data loading configuration
# -------------------------------------------------

def _mode_config(mode: str):
    if mode == "handcrafted":
        return dict(
            train_dir=TRAIN_HANDCRAFTED_DIR,
            eval_dir=EVAL_HANDCRAFTED_DIR,
            suffix_train="_features.npz",
            suffix_eval="_features.npz",
            load_patient=load_patient_handcrafted,
            feature_names=FEATURE_NAMES_16,
            n_features=len(FEATURE_NAMES_16),
        )
    else:
        return dict(
            train_dir=TRAIN_LABRAM_DIR,
            eval_dir=EVAL_LABRAM_DIR,
            suffix_train="_embeddings.npz",
            suffix_eval="_embeddings_labeled.npz",
            load_patient=load_patient_labram,
            feature_names=None,
            n_features=None,
        )


def run_mode(mode: str, thr: float):
    """Run the full train/eval pipeline for a single mode.

    Returns (metrics_dict, train_patients, imp_vec, feature_names).
    """
    cfg = _mode_config(mode)

    print(f"\n{'='*60}")
    print(f"  Mode : {mode.upper()}")
    print(f"  Test : {EVAL_PATIENT_ID}  |  threshold : {thr}")
    print(f"{'='*60}")

    # ---- build train index (all patients except P58) ----
    train_index = build_patient_index(cfg["train_dir"], cfg["suffix_train"])
    train_index.pop(EVAL_PATIENT_ID, None)
    train_patients = sorted(train_index.keys())
    print(f"Train patients ({len(train_patients)}): {train_patients}")

    # ---- load and concatenate all train data ----
    xs, ys = [], []
    for pid in train_patients:
        x, y = cfg["load_patient"](train_index[pid])
        xs.append(x); ys.append(y)
        print(f"  {pid}: windows={len(y)}  pos={int(y.sum())}  neg={int((y==0).sum())}")

    X_train = np.concatenate(xs, axis=0)
    y_train = np.concatenate(ys, axis=0)
    print(f"\nTotal train: windows={len(y_train)}  pos={int(y_train.sum())}  "
          f"neg={int((y_train==0).sum())}")

    # ---- infer embedding dim for LaBraM ----
    n_features = cfg["n_features"]
    feature_names = cfg["feature_names"]
    if n_features is None:
        n_features    = X_train.shape[1]
        feature_names = [f"f{i}" for i in range(n_features)]
        print(f"LaBraM embedding dim: {n_features}")

    # ---- load eval (P58) ----
    eval_index = build_patient_index(cfg["eval_dir"], cfg["suffix_eval"])
    if EVAL_PATIENT_ID not in eval_index:
        raise RuntimeError(
            f"Eval patient {EVAL_PATIENT_ID} not found in {cfg['eval_dir']} "
            f"(suffix={cfg['suffix_eval']})"
        )
    X_eval, y_eval = cfg["load_patient"](eval_index[EVAL_PATIENT_ID])
    print(f"\nTest  ({EVAL_PATIENT_ID}): windows={len(y_eval)}  "
          f"pos={int(y_eval.sum())}  neg={int((y_eval==0).sum())}")

    # ---- train ----
    print(f"\nTraining XGBoost...")
    model, imp_vec = train_xgb(X_train, y_train, n_features)

    # ---- evaluate ----
    y_proba = model.predict_proba(X_eval)[:, 1]
    m = evaluate(y_eval, y_proba, thr=thr)

    print(f"\n=== Results on {EVAL_PATIENT_ID} ({mode}) ===")
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

    return m, train_patients, imp_vec, feature_names


# -------------------------------------------------
# Plot: ONE grouped bar chart, handcrafted vs LaBraM side by side per metric
# -------------------------------------------------
# colour  -> metric (kept from the original per-metric colours)
# fill/hatch -> representation (handcrafted = solid, LaBraM = hatched),
#               matching the Figure 6 convention.

FIG_OUTPUT = Path("fig_xgb_eval_p58_paired.pdf")


def plot_results_combined(
    m_handcrafted: dict, train_patients_hc: list[str],
    m_labram: dict, train_patients_lb: list[str],
    thr: float, out_path: Path = FIG_OUTPUT,
):
    metrics = [
        ("bacc",    "Balanced Accuracy"),
        ("roc_auc", "ROC AUC"),
        ("f1",      "F1"),
        ("pr_auc",  "PR AUC"),
    ]
    metric_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]
    labels = [lab for _, lab in metrics]
    vals_hc = [m_handcrafted[k] for k, _ in metrics]
    vals_lb = [m_labram[k] for k, _ in metrics]

    x = np.arange(len(metrics))
    width = 0.4  # two half-width bars per metric

    plt.rcParams["hatch.linewidth"] = 0.8
    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_hc = ax.bar(
        x - width / 2, vals_hc, width,
        color=metric_colors, edgecolor="black", linewidth=0.5,
    )
    bars_lb = ax.bar(
        x + width / 2, vals_lb, width,
        color=metric_colors, edgecolor="black", linewidth=0.5, hatch="///",
    )

    # Value labels on top of each bar.
    for bars, vals in ((bars_hc, vals_hc), (bars_lb, vals_lb)):
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

    pat = PATIENT_PATTERN.get(EVAL_PATIENT_ID, "?")
    train_note = (
        ", ".join(train_patients_hc)
        if train_patients_hc == train_patients_lb
        else "all patients except " + EVAL_PATIENT_ID
    )
    fig.suptitle(
        f"XGBoost baseline on {EVAL_PATIENT_ID} ({pat}) — Handcrafted vs LaBraM",
        fontsize=12,
    )
    ax.set_title(f"train: {train_note}", fontsize=8, color="dimgray")

    # Representation legend (solid vs hatched).
    rep_legend = [
        Patch(facecolor="0.8", edgecolor="black", label="Handcrafted"),
        Patch(facecolor="0.8", edgecolor="black", hatch="///", label="LaBraM"),
    ]
    ax.legend(handles=rep_legend, title="Representation",
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    # Confusion counts for both representations, below the axes.
    conf_text = (
        f"Handcrafted:  TP={int(m_handcrafted['tp'])}  FP={int(m_handcrafted['fp'])}  "
        f"TN={int(m_handcrafted['tn'])}  FN={int(m_handcrafted['fn'])}      "
        f"LaBraM:  TP={int(m_labram['tp'])}  FP={int(m_labram['fp'])}  "
        f"TN={int(m_labram['tn'])}  FN={int(m_labram['fn'])}      (thr={thr})"
    )
    ax.text(
        0.5, -0.12, conf_text, ha="center", va="top", fontsize=8,
        color="dimgray", transform=ax.transAxes,
    )

    fig.subplots_adjust(left=0.08, right=0.86, top=0.90, bottom=0.16)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"\nCombined plot saved to: {out_path} and {out_path.with_suffix('.png')}")
    plt.show()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    thr = THR

    m_hc, train_patients_hc, _, _ = run_mode("handcrafted", thr)
    m_lb, train_patients_lb, _, _ = run_mode("labram", thr)

    plot_results_combined(m_hc, train_patients_hc, m_lb, train_patients_lb, thr)


if __name__ == "__main__":
    main()