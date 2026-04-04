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
# Patient-level pattern mapping
PATIENT_PATTERN = {
    "P20": "LPD",
    "P36": "LPD",
    "P58": "LPD",
    "P73": "LPD",
    "P48": "GPD",
    "P49": "GPD",
    "P54": "GPD",
    "P70": "GPD",
    # P28: LPD - excluded (test patient)
    # P55: LRDA - ignored
}

PATTERN_GROUPS = {
    "LPD": ["P20", "P36", "P58", "P73"],
    "GPD": ["P48", "P49", "P54", "P70"],
}
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
    # print(z["feature_names"])
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, 16)   # (samples, channels, features_per_channel)
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


def train_xgb(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    # handle class imbalance (in case train split is not perfectly balanced)
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
    score = model.get_booster().get_score(importance_type="gain")


    return model,score


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    # roc_auc needs both classes present
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def sec_to_hmsms(sec: float) -> str:
    """
    Convert seconds to HH:MM:SS.mmm
    """
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = int(sec % 60)
    millis = int((sec - int(sec)) * 1000)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"


def safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    # average_precision also benefits from both classes; if only one class, it's degenerate
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


def save_hard_errors_csv(
    out_csv: Path,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    center_sec: np.ndarray,
    center_hmsms: np.ndarray,
    source_edf: np.ndarray,
    thr: float,
    top_k: int = 5,
):
    y_pred = (y_proba >= thr).astype(np.uint8)

    idx = np.arange(len(y_true))

    fp_idx = idx[(y_true == 0) & (y_pred == 1)]
    fp_sorted = fp_idx[np.argsort(y_proba[fp_idx])[::-1]]
    fp_top = fp_sorted[:top_k]

    fn_idx = idx[(y_true == 1) & (y_pred == 0)]
    fn_sorted = fn_idx[np.argsort(y_proba[fn_idx])]
    fn_top = fn_sorted[:top_k]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "error_type",
            "window_center_time",
            "y_true",
            "y_pred",
            "y_proba",
            "recording_file",
        ])

        for i in fp_top:
            w.writerow([
                "false_positive",
                sec_to_hmsms(center_sec[i]),
                int(y_true[i]),
                int(y_pred[i]),
                f"{float(y_proba[i]):.8f}",
                source_edf[i],
            ])

        for i in fn_top:
            w.writerow([
                "false_negative",
                sec_to_hmsms(center_sec[i]),
                int(y_true[i]),
                int(y_pred[i]),
                f"{float(y_proba[i]):.8f}",
                source_edf[i],
            ])

def print_feature_stats(X: np.ndarray, feature_names: list[str], title: str = "Feature stats"):
    means = X.mean(axis=0)
    stds = X.std(axis=0)

    print(f"\n=== {title} ===")
    for name, mu, sigma in zip(feature_names, means, stds):
        print(f"{name:20s} mean={mu:12.6e}  std={sigma:12.6e}")
# -------------------------------------------------
# LOPO CV
# -------------------------------------------------

def main():
    in_dir = Path("../data/80hz_freq_time_features_cache_basic")
    patient_files = build_patient_index(in_dir)

    thr = 0.5
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    n_features = len(FEATURE_NAMES_16)

    pattern_colors = {"LPD": "tab:blue", "GPD": "tab:orange"}

    for pattern, group_pids in PATTERN_GROUPS.items():
        print(f"\n{'='*60}")
        print(f"Pattern: {pattern} | Patients: {group_pids}")
        print(f"{'='*60}")

        # keep only patients that have files
        available = [p for p in group_pids if p in patient_files]
        if len(available) < 2:
            print(f"  Not enough patients for LOPO, skipping.")
            continue

        fold_metrics: dict[str, dict[str, float]] = {}
        collected = {k: [] for k in keys}
        agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
        all_importances = []

        for test_pid in available:
            train_pids = [p for p in available if p != test_pid]

            x_train_list, y_train_list = [], []
            for pid in train_pids:
                x, y, _, _, _ = concat_patient_files(patient_files[pid])
                x_train_list.append(x)
                y_train_list.append(y)

            x_train = np.concatenate(x_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)

            x_test, y_test, _, _, _ = concat_patient_files(patient_files[test_pid])

            model, score = train_xgb(x_train, y_train)

            imp_vec = np.zeros(n_features)
            for k, v in score.items():
                idx = int(k[1:])
                imp_vec[idx] = v
            all_importances.append(imp_vec)

            y_proba = model.predict_proba(x_test)[:, 1]
            m = evaluate_fold(y_test, y_proba, thr=thr)
            fold_metrics[test_pid] = m

            for k in keys:
                collected[k].append(m[k])
            for c in ["tp", "fp", "tn", "fn"]:
                agg[c] += m[c]

            print(
                f"[{test_pid}] n={len(y_test)} pos={int(y_test.sum())} "
                f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
                f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
                f"roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}"
            )

        # Summary per pattern
        all_importances = np.array(all_importances)
        mean_imp = all_importances.mean(axis=0)
        std_imp = all_importances.std(axis=0)

        print(f"\n=== [{pattern}] Feature importance (gain) ===")
        for name, mu, sigma in zip(FEATURE_NAMES_16, mean_imp, std_imp):
            print(f"{name:20s} mean={mu:6.4f}  std={sigma:6.4f}")

        print(f"\n=== [{pattern}] LOPO summary ===")
        for k in keys:
            mean, std = summarize_metric(collected[k])
            print(f"{k:>10}: {mean:.4f} ± {std:.4f}")

        print(f"\n=== [{pattern}] Aggregate confusion ===")
        print(f"TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

        # Bar plot per pattern
        items = sorted(fold_metrics.items(), key=lambda kv: kv[1]["acc"])
        pids = [pid for pid, _ in items]
        accs = [m["acc"] for _, m in items]

        plt.figure(figsize=(8, 4))
        plt.bar(pids, accs, color=pattern_colors[pattern])
        plt.ylim(0.0, 1.0)
        plt.ylabel("Accuracy")
        plt.xlabel("Patient (sorted)")
        plt.title(f"LOPO Accuracy — {pattern} patients")
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
