import re
from operator import truediv
from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt

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


def load_features_X_only(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    return X


def sensitivity_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return sens, spec


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------

def main():
    in_dir = Path("../data/features_cache_basic")
    test_dir = Path("../test/features_cache_basic")
    files = sorted(in_dir.glob("*_features.npz"))
    if not files:
        raise RuntimeError("No *_features.npz files found")

    # group files by patient (Pxx)
    raw = mne.io.read_raw_edf("../data/P20_GHB_00015_0000348.edf", preload=False, verbose="ERROR")
    ann = raw.annotations

    pos_ann_onsets = np.array([on for on, desc in zip(ann.onset, ann.description) if "*" in desc], dtype=float)
    pos_ann_onsets.sort()

    win_sec=0.5
    hop_sec=0.25

    patient_files: dict[str, list[Path]] = {}
    skipped_first_p20 = False
    for f in files:
        pid = patient_from_filename(f.name)
        if pid == "P20" and not skipped_first_p20:
            print(f"Skipping first P20: {f.name}")
            skipped_first_p20 = True
            continue
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        patient_files.setdefault(pid, []).append(f)

    patients = sorted(patient_files.keys())
    print(f"Found {len(files)} files, {len(patients)} patients:")
    print(patients)

    results = []

    # -------------------------------------------------
    # Use all patients, except p20, for training and test on p20 full eeg segmented into sliding windows
    # -------------------------------------------------

    x_train_list, y_train_list = [], []

    for pid, flist in patient_files.items():
        for f in flist:
            x, y = load_features(f)
            x_train_list.append(x)
            y_train_list.append(y)

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    p20_file = test_dir / "P20_GHB_00015_0000348_full_features.npz"

    if not p20_file.exists():
        raise RuntimeError(f"File not found: {p20_file}")

    x_test = load_features_X_only(p20_file)

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
    threshold_values = np.linspace(0.0, 1.0, 101)
    y_proba = model.predict_proba(x_test)[:, 1]
    print("y_proba stats:",
          "min", float(y_proba.min()),
          "p50", float(np.median(y_proba)),
          "p90", float(np.quantile(y_proba, 0.90)),
          "max", float(y_proba.max()))
    for th in [0.1, 0.5, 0.9]:
        y_pred = (y_proba >= th).astype(np.uint8)
        print(th, "hit_ratio", float(y_pred.mean()))
    threshold_hits=[]
    for thres in threshold_values:
        y_pred = (y_proba >= thres).astype(np.uint8)
        pos_idx = np.where(y_pred == 1)[0]

        hit_ratio=len(pos_idx)/len(y_pred)
        threshold_hits.append((thres,hit_ratio))

    thr_arr = np.array([t for t, r in threshold_hits], dtype=float)
    hits_arr = np.array([r for t, r in threshold_hits], dtype=float)

    out_hit_ratio = Path("../data/p20_hit_ratio_curve.npz")

    np.savez(
        out_hit_ratio,
        thresholds=thr_arr.astype(np.float32),
        hit_ratio=hits_arr.astype(np.float32),
    )

    print(f"Saved hit-ratio curve to {out_hit_ratio}")

    plt.figure()
    plt.plot(thr_arr, hits_arr)
    plt.xlabel("Threshold")
    plt.ylabel("Hit Ratio (pos/total windows)")
    plt.title("Hit Ratio vs Threshold")
    plt.grid(True)
    plt.show()

        # sfreq = 256
        # step = 0.25  # in sec
        # step_sample = int(round(step * sfreq))
        # centers_in_sec = []
        # for index in pos_idx:
        #     start_sample = index * step_sample
        #     center_sample = int(start_sample + step_sample)
        #     center_sec = center_sample / sfreq
        #     centers_in_sec.append(center_sec)
        #
        # print(f"{len(pos_idx)}/{len(x_test)}")
        # print(centers_in_sec)



if __name__ == "__main__":
    main()
