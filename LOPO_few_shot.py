import re
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
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
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    """
    Extract patient id from filenames like:
    P20_GHB_00015_000348_features.npz -> 'P20'
    """
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_features(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_samples = X.shape[0]
    X = X.reshape(n_samples, 19, 16)   # (samples, channels, features_per_channel)
    X_avg = X.mean(axis=1)             # (samples, 16) average feature per channel

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


def concat_patient_files(
        file_list: list[Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs, ys, cs, ch, edfs = [], [], [], [], []

    for f in file_list:
        x, y, center_sec, center_hmsms, source_edf = load_features(f)
        xs.append(x)
        ys.append(y)
        cs.append(center_sec)
        ch.append(center_hmsms)
        edfs.extend([source_edf] * len(y))

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(cs, axis=0),
        np.concatenate(ch, axis=0),
        np.array(edfs, dtype=object),
    )


# -------------------------------------------------
# NEW: split a single file into train (5 pos + 5 neg)
#      and test (the remaining windows)
# -------------------------------------------------
def split_few_shot(
    X: np.ndarray,
    y: np.ndarray,
    center_sec: np.ndarray,
    center_hmsms: np.ndarray,
    source_edf: np.ndarray,
    n_pos: int = 5,
    n_neg: int = 5,
):
    """
    Take the first n_pos positive and n_neg negative windows as the
    training set.  Everything else becomes the test set.

    Returns (X_train, y_train, X_test, y_test, center_sec_test,
             center_hmsms_test, source_edf_test)
    or None if there are not enough samples of either class.
    """
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < n_pos or len(neg_idx) < n_neg:
        return None   # skip files with too few examples

    train_idx = np.concatenate([pos_idx[:n_pos], neg_idx[:n_neg]])
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]
    cs_test = center_sec[test_mask]
    ch_test = center_hmsms[test_mask]
    se_test = source_edf[test_mask]

    return X_train, y_train, X_test, y_test, cs_test, ch_test, se_test


def train_xgb(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    # handle class imbalance (in case train split is not perfectly balanced)
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
    score = model.get_booster().get_score(importance_type="gain")

    return model, score


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def sec_to_hmsms(sec: float) -> str:
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    millis = int((sec - int(sec)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def evaluate_fold(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= thr).astype(np.uint8)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    ra = safe_roc_auc(y_true, y_proba)
    pa = safe_pr_auc(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "acc": acc,
        "bacc": bacc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": ra,
        "pr_auc": pa,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def summarize_metric(values: list[float]) -> tuple[float, float]:
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


def print_feature_stats(X: np.ndarray, feature_names: list[str], title: str = "Feature stats"):
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    print(f"\n=== {title} ===")
    for name, mu, sigma in zip(feature_names, means, stds):
        print(f"{name:20s} mean={mu:12.6e}  std={sigma:12.6e}")


# -------------------------------------------------
# Main: per-file few-shot train → test on remainder
# -------------------------------------------------

def main():
    in_dir = Path("../data/freq_time_features_cache_basic")
    patient_files = build_patient_index(in_dir)
    patients = sorted(patient_files.keys())

    print(f"Patients: {patients} (n={len(patients)})")

    thr = 0.5
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]

    # Collect metrics per file (across all patients/files)
    file_metrics: list[dict] = []   # each entry: {pid, filename, **metrics}
    collected = {k: [] for k in keys}
    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}

    n_features = len(FEATURE_NAMES_16)
    all_importances = []

    for pid in patients:
        for npz_path in patient_files[pid]:
            X, y, center_sec, center_hmsms, source_edf = load_features(npz_path)

            # Split: first 5 pos + 5 neg → train; rest → test
            split = split_few_shot(X, y, center_sec, center_hmsms,
                                   np.array([source_edf] * len(y), dtype=object))

            if split is None:
                print(f"  [{pid}] Skipping {npz_path.name}: not enough pos/neg windows")
                continue

            X_train, y_train, X_test, y_test, cs_test, ch_test, se_test = split

            if len(y_test) == 0:
                print(f"  [{pid}] Skipping {npz_path.name}: no test windows left")
                continue

            model, score = train_xgb(X_train, y_train)

            # Feature importance
            imp_vec = np.zeros(n_features)
            for k, v in score.items():
                idx = int(k[1:])   # "f6" → 6
                imp_vec[idx] = v
            all_importances.append(imp_vec)

            y_proba = model.predict_proba(X_test)[:, 1]
            m = evaluate_fold(y_test, y_proba, thr=thr)

            file_metrics.append({
                "pid": pid,
                "file": npz_path.name,
                **m,
            })

            for k in keys:
                collected[k].append(m[k])
            for c in ["tp", "fp", "tn", "fn"]:
                agg[c] += m[c]

            print(
                f"[{pid}] {npz_path.name} | "
                f"train={len(y_train)} (pos={int(y_train.sum())}, neg={int((y_train==0).sum())}) "
                f"test={len(y_test)} pos={int(y_test.sum())} | "
                f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
                f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
                f"roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}"
            )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    if all_importances:
        all_importances = np.array(all_importances)
        mean_imp = all_importances.mean(axis=0)
        std_imp = all_importances.std(axis=0)

        print("\n=== Feature importance across files (gain) ===")
        for name, mu, sigma in zip(FEATURE_NAMES_16, mean_imp, std_imp):
            print(f"{name:20s} mean={mu:6.4f}  std={sigma:6.4f}")

    print(f"\n=== Summary: mean ± std across {len(file_metrics)} files ===")
    for k in keys:
        mean, std = summarize_metric(collected[k])
        print(f"{k:>10}: {mean:.4f} ± {std:.4f}")

    print("\n=== Aggregate confusion (sum across all files, at thr=0.5) ===")
    print(f"TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

    # Worst files by F1
    worst = sorted(file_metrics, key=lambda d: d["f1"])[:5]
    print("\nWorst files by F1:")
    for d in worst:
        print(f"  [{d['pid']}] {d['file']}: f1={d['f1']:.3f} "
              f"(prec={d['precision']:.3f}, rec={d['recall']:.3f})")

    # -------------------------------------------------
    # Bar plot: Accuracy per file, colored by patient
    # -------------------------------------------------
    patient_pattern = {
        "P20": "LPD",
        "P28": "LPD",
        "P36": "LPD",
        "P48": "GPD",
        "P49": "GPD",
        "P54": "GPD",
        "P55": "LRDA",
        "P58": "LPD",
        "P70": "GPD",
        "P73": "LPD",
    }
    pattern_colors = {
        "LPD": "tab:blue",
        "GPD": "tab:orange",
        "LRDA": "tab:green",
    }

    sorted_metrics = sorted(file_metrics, key=lambda d: d["acc"])
    labels = [f"{d['pid']}\n{d['file'][:21]}" for d in sorted_metrics]
    accs = [d["acc"] for d in sorted_metrics]
    colors = [
        pattern_colors.get(patient_pattern.get(d["pid"], "UNK"), "gray")
        for d in sorted_metrics
    ]

    x_pos = range(len(labels))
    plt.figure(figsize=(max(10, len(labels) * 0.9), 4))
    plt.bar(x_pos, accs, color=colors)
    plt.xticks(ticks=x_pos, labels=labels, rotation=45, ha="right", fontsize=7)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("File (sorted by accuracy)")
    plt.title("Per-file Accuracy – 5 pos + 5 neg train, rest = test")
    plt.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="tab:blue", label="LPD"),
        Patch(facecolor="tab:orange", label="GPD"),
        Patch(facecolor="tab:green", label="LRDA"),
    ]
    plt.legend(handles=legend_elems, title="Pattern")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()