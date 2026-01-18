import re
from pathlib import Path
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
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


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


# -------------------------------------------------
# Main LOPO pipeline
# -------------------------------------------------

def main():
    in_dir = Path("../../../data/features_cache_basic")
    files = sorted(in_dir.glob("*_features.npz"))
    if not files:
        raise RuntimeError("No *_features.npz files found")

    # group files by patient (Pxx)
    patient_files: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        patient_files.setdefault(pid, []).append(f)

    patients = sorted(patient_files.keys())
    print(f"Found {len(files)} files, {len(patients)} patients:")
    print(patients)

    results = []

    # -------------------------------------------------
    # Leave-One-Patient-Out
    # -------------------------------------------------
    for test_pid in patients:
        print(f"\n[LOPO] Test patient = {test_pid}")

        x_train_list, y_train_list = [], []
        x_test_list, y_test_list = [], []

        for pid, flist in patient_files.items():
            for f in flist:
                x, y = load_features(f)
                if pid == test_pid:
                    x_test_list.append(x)
                    y_test_list.append(y)
                else:
                    x_train_list.append(x)
                    y_train_list.append(y)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)
        x_test = np.concatenate(x_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)

        print(
            f"  Train windows: {len(y_train)} (pos={int(y_train.sum())}) | "
            f"Test windows: {len(y_test)} (pos={int(y_test.sum())})"
        )

        # handle class imbalance
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
        )

        model.fit(x_train, y_train)

        # inference
        y_proba = model.predict_proba(x_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(np.uint8)

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        sens, spec = sensitivity_specificity(y_test, y_pred)

        if len(np.unique(y_test)) == 2:
            roc = roc_auc_score(y_test, y_proba)
            pr = average_precision_score(y_test, y_proba)
        else:
            roc = np.nan
            pr = np.nan

        print(
            f"  bAcc={bacc:.3f} F1={f1:.3f} "
            f"Sens={sens:.3f} Spec={spec:.3f} "
            f"ROC-AUC={roc if not np.isnan(roc) else 'NA'}"
        )

        results.append(
            dict(
                patient=test_pid,
                n_train=len(y_train),
                n_test=len(y_test),
                pos_train=int(y_train.sum()),
                pos_test=int(y_test.sum()),
                acc=acc,
                bacc=bacc,
                f1=f1,
                sens=sens,
                spec=spec,
                roc_auc=roc,
                pr_auc=pr,
            )
        )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    print("\n=== LOPO summary (mean ± std) ===")

    def mean_std(key):
        vals = np.array([r[key] for r in results], dtype=float)
        vals = vals[~np.isnan(vals)]
        return vals.mean(), vals.std()

    for k in ["acc", "bacc", "f1", "sens", "spec", "roc_auc", "pr_auc"]:
        m, s = mean_std(k)
        print(f"{k:7s}: {m:.4f} ± {s:.4f}")

    out = Path("../../../data/lopo_xgb_results.npy")
    np.save(out, results, allow_pickle=True)
    print(f"\nSaved per-patient results to {out}")


if __name__ == "__main__":
    main()
