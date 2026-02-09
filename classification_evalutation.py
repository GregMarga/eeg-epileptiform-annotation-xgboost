import re
from pathlib import Path
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
    y = z["y"].astype(np.uint8).ravel()
    return X, y


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


def concat_patient_files(file_list: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for f in file_list:
        x, y = load_features(f)
        xs.append(x)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


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
        colsample_bytree=1.0,
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
    # roc_auc needs both classes present
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


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


# -------------------------------------------------
# LOPO CV
# -------------------------------------------------

def main():
    in_dir = Path("../data/features_cache_basic")
    patient_files = build_patient_index(in_dir)
    patients = sorted(patient_files.keys())

    print(f"Patients: {patients} (n={len(patients)})")

    thr = 0.5  # fixed threshold for classification metrics

    fold_metrics: dict[str, dict[str, float]] = {}
    # store arrays for summary
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}

    # aggregate confusion counts across folds
    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}

    for test_pid in patients:
        # build train set from all other patients
        train_pids = [p for p in patients if p != test_pid]

        x_train_list, y_train_list = [], []
        for pid in train_pids:
            x, y = concat_patient_files(patient_files[pid])
            x_train_list.append(x)
            y_train_list.append(y)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # test set is that patient's files
        x_test, y_test = concat_patient_files(patient_files[test_pid])

        # train model
        model = train_xgb(x_train, y_train)

        # predict
        y_proba = model.predict_proba(x_test)[:, 1]

        # metrics
        m = evaluate_fold(y_test, y_proba, thr=thr)
        fold_metrics[test_pid] = m

        for k in keys:
            collected[k].append(m[k])

        for c in ["tp", "fp", "tn", "fn"]:
            agg[c] += m[c]

        print(
            f"[{test_pid}] "
            f"n={len(y_test)} pos={int(y_test.sum())} "
            f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
            f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}"
        )

    print("\n=== LOPO summary (mean ± std across patients) ===")
    for k in keys:
        mean, std = summarize_metric(collected[k])
        print(f"{k:>10}: {mean:.4f} ± {std:.4f}")

    print("\n=== Aggregate confusion (sum across folds, at thr=0.5) ===")
    print(f"TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

    # Optional: identify worst patients by F1
    worst = sorted(fold_metrics.items(), key=lambda kv: kv[1]["f1"])[:5]
    print("\nWorst by F1:")
    for pid, m in worst:
        print(f"  {pid}: f1={m['f1']:.3f} (prec={m['precision']:.3f}, rec={m['recall']:.3f})")


if __name__ == "__main__":
    main()