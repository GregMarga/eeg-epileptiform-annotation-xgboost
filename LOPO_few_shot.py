import re
from pathlib import Path
import csv
import numpy as np
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

FEATURE_NAMES_16 = [
    "zero_cross",
    "maxima",
    "minima",
    "rms",
    "skew",
    "kurt_excess",
    "total_power_1_40",
    "peak_freq_1_40",
    "mean_band_delta",
    "mean_band_theta",
    "mean_band_alpha",
    "mean_band_beta",
    "norm_band_delta",
    "norm_band_theta",
    "norm_band_alpha",
    "norm_band_beta",
]

# -------------------------------------------------
# Config
# -------------------------------------------------

N_SHOT_RANGE = list(range(1, 21))   # 1..20 inclusive
THRESHOLD = 0.5
CSV_OUTPUT = Path("results_xgb_fewshot_perfile.csv")


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_features(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_samples = X.shape[0]
    n_channels = len(z["ch_names"])
    X = X.reshape(n_samples, n_channels, 16)
    X_avg = X.mean(axis=1)             # (samples, 16) average across channels

    y = z["y"].astype(np.uint8).ravel()
    center_sec = z["center_sec"].astype(np.float64)
    center_hmsms = z["center_hmsms"]
    source_edf = str(z["source_edf"])

    return X_avg, y, center_sec, center_hmsms, source_edf


def build_patient_index(in_dir: Path) -> dict[str, list[Path]]:
    files = sorted(in_dir.glob("*_features.npz"))
    if not files:
        raise RuntimeError(f"No *_features.npz files found in {in_dir}")

    patient_files: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        patient_files.setdefault(pid, []).append(f)

    if not patient_files:
        raise RuntimeError("No patient files parsed. Check filename pattern.")
    return patient_files


def split_few_shot(
    X: np.ndarray,
    y: np.ndarray,
    n_pos: int,
    n_neg: int,
):
    """
    Take the first n_pos positives and n_neg negatives as the training set.
    Everything else becomes the test set.
    Returns None if there are not enough samples of either class.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < n_pos or len(neg_idx) < n_neg:
        return None

    train_idx = np.concatenate([pos_idx[:n_pos], neg_idx[:n_neg]])
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def train_xgb(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=1,
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


def summarize_metric(values: list[float]) -> tuple[float, float]:
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


# -------------------------------------------------
# Sweep one N_SHOT value across all files
# -------------------------------------------------

def run_for_n_shot(
    n_shot: int,
    patient_files: dict[str, list[Path]],
    threshold: float,
) -> dict[str, float]:
    """
    For the given n_shot, train on first n_shot pos + n_shot neg of each file
    and test on the rest. Aggregate metrics across all files.
    """
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}
    n_files_used = 0
    n_files_skipped = 0
    agg_tp = agg_fp = agg_tn = agg_fn = 0

    for pid, file_list in patient_files.items():
        for npz_path in file_list:
            X, y, _cs, _ch, _se = load_features(npz_path)

            split = split_few_shot(X, y, n_pos=n_shot, n_neg=n_shot)
            if split is None:
                n_files_skipped += 1
                continue

            X_train, y_train, X_test, y_test = split
            if len(y_test) == 0 or len(np.unique(y_test)) < 2:
                # Need both classes in the test set for AUC metrics to make sense.
                # Files where the "positive" class is exhausted by the support
                # would otherwise produce NaN for AUC.
                n_files_skipped += 1
                continue

            model = train_xgb(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            m = evaluate_fold(y_test, y_proba, thr=threshold)
            for k in keys:
                collected[k].append(m[k])

            agg_tp += m["tp"]; agg_fp += m["fp"]
            agg_tn += m["tn"]; agg_fn += m["fn"]
            n_files_used += 1

    row = {
        "n_shot": n_shot,
        "n_files_used": n_files_used,
        "n_files_skipped": n_files_skipped,
        "tp_sum": int(agg_tp),
        "fp_sum": int(agg_fp),
        "tn_sum": int(agg_tn),
        "fn_sum": int(agg_fn),
    }

    # Means and stds across files
    for k in keys:
        mean, std = summarize_metric(collected[k])
        row[k] = mean
        row[f"{k}_std"] = std

    return row


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    in_dir = Path("../../../data/80hz_freq_time_features_cache_basic")
    patient_files = build_patient_index(in_dir)
    patients = sorted(patient_files.keys())

    n_files_total = sum(len(v) for v in patient_files.values())
    print(f"Patients: {patients} (n={len(patients)})")
    print(f"Total files: {n_files_total}")
    print(f"Sweeping N_SHOT: {N_SHOT_RANGE} (threshold={THRESHOLD})\n")

    rows = []
    for n_shot in N_SHOT_RANGE:
        row = run_for_n_shot(n_shot, patient_files, THRESHOLD)
        rows.append(row)
        print(
            f"n_shot={n_shot:2d} | files_used={row['n_files_used']:3d} "
            f"(skipped={row['n_files_skipped']:3d}) | "
            f"acc={row['acc']:.4f} ± {row['acc_std']:.4f} | "
            f"bacc={row['bacc']:.4f} | f1={row['f1']:.4f} | "
            f"roc_auc={row['roc_auc']:.4f}"
        )

    # Save CSV
    fieldnames = list(rows[0].keys())
    with CSV_OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to: {CSV_OUTPUT.resolve()}")


if __name__ == "__main__":
    main()