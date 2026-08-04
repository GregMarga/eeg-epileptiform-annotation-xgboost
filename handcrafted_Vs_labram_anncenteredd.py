"""
Unified LOPO evaluation for handcrafted features and LaBraM embeddings,
producing a single paired bar plot (Figure 6 replacement).

Design follows the supervisor's note: instead of two separate subplots,
each patient gets two adjacent half-width bars so the reader can directly
compare LaBraM vs handcrafted. Colour encodes the PD pattern (kept from the
original figure); fill vs hatch encodes the representation:
    - Handcrafted : solid fill
    - LaBraM      : same colour, hatched (///)

The two pipelines keep their original, distinct preprocessing:
    - Handcrafted : 16-dim channel-averaged features, per-patient z-score
    - LaBraM      : 200-dim embeddings used as-is (no standardisation)
"""

import re
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
# Config
# -------------------------------------------------

LABRAM_DIR = Path("../../../data/labram_embeddings_1s_new")
HANDCRAFTED_DIR = Path("../../../data/80hz_freq_time_features_pyprep_1s")

NORMALIZE_HANDCRAFTED = True    # per-patient z-score (as in your handcrafted script)
NORMALIZE_LABRAM = False        # LaBraM embeddings used as-is

THRESHOLD = 0.5

# Bar ordering on the x-axis:
#   "mean"        -> sort by mean(handcrafted, labram) accuracy  (default)
#   "handcrafted" -> sort by handcrafted accuracy
#   "labram"      -> sort by LaBraM accuracy
#   "patient"     -> sort by patient id
SORT_BY = "mean"

OUT_FIG = Path("fig6_lopo_accuracy_paired.pdf")

# LaBraM npz key candidates (tolerant loader)
FEATURE_KEYS = ("X", "embeddings", "features", "emb")
LABEL_KEYS = ("y", "labels", "label")

N_FEATURES_HAND = 16
N_FEATURES_LABRAM = 200

PATIENT_PATTERN = {
    "P20": "LPD", "P28": "LPD", "P36": "LPD",
    "P48": "GPD", "P49": "GPD", "P54": "GPD",
    "P55": "LRDA", "P58": "LPD", "P70": "GPD", "P73": "LPD",
}
PATTERN_COLORS = {"LPD": "tab:blue", "GPD": "tab:orange", "LRDA": "tab:green"}


# -------------------------------------------------
# Shared helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    """P20_GHB_00015_0000348_*.npz -> 'P20'."""
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_patient_index(in_dir: Path, glob_pat: str) -> dict[str, list[Path]]:
    files = sorted(in_dir.glob(glob_pat))
    if not files:
        raise RuntimeError(f"No {glob_pat} files found in {in_dir}")

    patient_files: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        patient_files.setdefault(pid, []).append(f)

    if not patient_files:
        raise RuntimeError(f"No patient files parsed in {in_dir}. Check filename pattern.")
    return patient_files


def _first_present(candidates: tuple[str, ...], available: set[str]) -> str | None:
    for k in candidates:
        if k in available:
            return k
    return None


def train_xgb(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
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
    model.fit(x_train, y_train)
    return model


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def evaluate_fold(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= thr).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "pr_auc": safe_pr_auc(y_true, y_proba),
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
    }


def summarize_metric(values: list[float]) -> tuple[float, float]:
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


# -------------------------------------------------
# Handcrafted loader (16-dim, channel-averaged, per-patient z-score)
# -------------------------------------------------

def zscore_per_patient(X: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    return (X - mean) / (std + eps)


def _load_handcrafted_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, N_FEATURES_HAND)  # (samples, channels, 16)
    X_avg = X.mean(axis=1)                                  # (samples, 16)
    y = z["y"].astype(np.uint8).ravel()
    return X_avg, y


def load_handcrafted_patient(file_list: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in file_list:
        x, y = _load_handcrafted_npz(f)
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if NORMALIZE_HANDCRAFTED:
        X = zscore_per_patient(X)  # THIS patient's stats only -> no cross-patient leakage
    return X, y


# -------------------------------------------------
# LaBraM loader (200-dim, no standardisation)
# -------------------------------------------------

def _load_labram_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)
    xk = _first_present(FEATURE_KEYS, keys)
    yk = _first_present(LABEL_KEYS, keys)
    if xk is None or yk is None:
        raise KeyError(
            f"Could not find feature/label keys in {npz_path.name}. "
            f"Found keys: {sorted(keys)}"
        )
    X = np.asarray(z[xk]).astype(np.float32)
    y = np.asarray(z[yk]).astype(np.uint8).ravel()
    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got {X.shape} in {npz_path.name}")
    if len(X) != len(y):
        raise ValueError(f"Row mismatch in {npz_path.name}: {len(X)} vs {len(y)}")
    return X, y


def load_labram_patient(file_list: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in file_list:
        x, y = _load_labram_npz(f)
        xs.append(x)
        ys.append(y)
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    if NORMALIZE_LABRAM:
        X = zscore_per_patient(X)
    return X, y


# -------------------------------------------------
# Generic LOPO runner
# -------------------------------------------------

def run_lopo(patient_files: dict[str, list[Path]], load_fn, tag: str) -> dict[str, dict[str, float]]:
    patients = sorted(patient_files.keys())
    print(f"\n=== LOPO [{tag}] — patients: {patients} (n={len(patients)}) ===")

    fold_metrics: dict[str, dict[str, float]] = {}
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}
    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}

    for test_pid in patients:
        train_pids = [p for p in patients if p != test_pid]

        x_train_list, y_train_list = [], []
        for pid in train_pids:
            x, y = load_fn(patient_files[pid])
            x_train_list.append(x)
            y_train_list.append(y)
        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        x_test, y_test = load_fn(patient_files[test_pid])

        model = train_xgb(x_train, y_train)
        y_proba = model.predict_proba(x_test)[:, 1]

        m = evaluate_fold(y_test, y_proba, thr=THRESHOLD)
        fold_metrics[test_pid] = m
        for k in keys:
            collected[k].append(m[k])
        for c in ("tp", "fp", "tn", "fn"):
            agg[c] += m[c]

        print(
            f"[{tag}][{test_pid}] n={len(y_test)} pos={int(y_test.sum())} "
            f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
            f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}"
        )

    print(f"\n--- LOPO summary [{tag}] (mean +/- std across patients) ---")
    for k in keys:
        mean, std = summarize_metric(collected[k])
        print(f"{k:>10}: {mean:.4f} +/- {std:.4f}")
    print(f"Aggregate confusion [{tag}]: "
          f"TP={int(agg['tp'])} FP={int(agg['fp'])} TN={int(agg['tn'])} FN={int(agg['fn'])}")

    return fold_metrics


# -------------------------------------------------
# Paired bar plot
# -------------------------------------------------

def plot_paired(m_hand: dict[str, dict], m_lab: dict[str, dict], out_path: Path = OUT_FIG):
    common = sorted(set(m_hand) & set(m_lab))
    only_one = set(m_hand) ^ set(m_lab)
    if only_one:
        print(f"\n[warn] patients not present in BOTH representations, dropped from plot: "
              f"{sorted(only_one)}")

    acc_h = {p: m_hand[p]["acc"] for p in common}
    acc_l = {p: m_lab[p]["acc"] for p in common}

    if SORT_BY == "handcrafted":
        order = sorted(common, key=lambda p: acc_h[p])
    elif SORT_BY == "labram":
        order = sorted(common, key=lambda p: acc_l[p])
    elif SORT_BY == "patient":
        order = sorted(common)
    else:  # "mean"
        order = sorted(common, key=lambda p: 0.5 * (acc_h[p] + acc_l[p]))

    colors = [PATTERN_COLORS.get(PATTERN_PATTERN_LOOKUP(p), "gray") for p in order]

    x = np.arange(len(order))
    width = 0.4  # two half-width bars per patient

    plt.rcParams["hatch.linewidth"] = 0.8
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(
        x - width / 2, [acc_h[p] for p in order], width,
        color=colors, edgecolor="black", linewidth=0.5,
    )
    ax.bar(
        x + width / 2, [acc_l[p] for p in order], width,
        color=colors, edgecolor="black", linewidth=0.5, hatch="///",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Patient")
    ax.set_title("LOPO Accuracy per Patient: Handcrafted vs LaBraM")
    ax.grid(axis="y", alpha=0.3)

    # Two legends below the axes (horizontal), so no wide empty margin:
    #   1) colour  -> PD pattern      (bottom-left)
    #   2) texture -> representation  (bottom-right)
    pattern_legend = [
        Patch(facecolor=c, edgecolor="black", label=lab)
        for lab, c in PATTERN_COLORS.items()
    ]
    rep_legend = [
        Patch(facecolor="0.75", edgecolor="black", label="Handcrafted"),
        Patch(facecolor="0.75", edgecolor="black", hatch="///", label="LaBraM"),
    ]
    leg1 = ax.legend(
        handles=pattern_legend, title="PD pattern", ncol=3,
        loc="upper left", bbox_to_anchor=(0.0, -0.14),
        frameon=True, columnspacing=1.0, handletextpad=0.5,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=rep_legend, title="Representation", ncol=2,
        loc="upper right", bbox_to_anchor=(1.0, -0.14),
        frameon=True, columnspacing=1.0, handletextpad=0.5,
    )

    # Reserve room at the bottom for the two legends. tight_layout() is NOT used
    # here because it does not account for legends anchored outside the axes,
    # which is exactly why they were getting clipped / pushed to the side.
    fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.26)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to {out_path} and {out_path.with_suffix('.png')}")
    plt.show()


def PATTERN_PATTERN_LOOKUP(pid: str) -> str:
    return PATIENT_PATTERN.get(pid, "UNK")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    hand_files = build_patient_index(HANDCRAFTED_DIR, "*_features.npz")
    labram_files = build_patient_index(LABRAM_DIR, "*.npz")

    m_hand = run_lopo(hand_files, load_handcrafted_patient, tag="handcrafted")
    m_lab = run_lopo(labram_files, load_labram_patient, tag="labram")

    plot_paired(m_hand, m_lab)


if __name__ == "__main__":
    main()