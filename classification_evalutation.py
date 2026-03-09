import re
from operator import truediv
from pathlib import Path
import numpy as np
import mne
import matplotlib.pyplot as plt
import csv

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


def sec_to_hmsms(sec: float) -> str:
    # format seconds -> HH:MM:SS.mmm (for easy human reading)
    ms = int(round(sec * 1000))
    s, ms = divmod(ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def pd_intervals_from_mne_annotations(ann) -> np.ndarray:
    """
    Returns array of shape (K,2): [[start_sec, stop_sec], ...]
    Assumes PD_START/PD_STOP are ordered in time.
    Ignores '*' and '-' or any other labels.
    """
    events = sorted(zip(ann.onset, ann.description), key=lambda x: x[0])

    intervals = []
    current_start = None

    for t, desc in events:
        desc = desc.strip().upper()
        if desc == "PD_START":
            current_start = t
        elif desc == "PD_STOP":
            if current_start is not None and t > current_start:
                intervals.append((float(current_start), float(t)))
            current_start = None

    return np.array(intervals, dtype=float)


def label_windows_by_overlap(
        n_windows: int,
        win_sec: float,
        hop_sec: float,
        intervals: np.ndarray,
        *,
        min_overlap_sec: float = 0.0
) -> np.ndarray:
    """
    intervals: array (K,2) with [start, stop] in seconds.
    Returns y_true (n_windows,) uint8
    """
    starts = np.arange(n_windows, dtype=float) * hop_sec
    ends = starts + win_sec

    y = np.zeros(n_windows, dtype=np.uint8)
    if intervals.size == 0:
        return y

    for k in range(intervals.shape[0]):
        a, b = intervals[k]
        # overlap length for all windows with this interval
        overlap = np.minimum(ends, b) - np.maximum(starts, a)
        y |= (overlap > min_overlap_sec).astype(np.uint8)

    return y


def mask_to_intervals(mask: np.ndarray, win_sec: float, hop_sec: float):
    """
    Convert boolean/0-1 mask over windows to contiguous time intervals (start, duration).
    Windows i cover [i*hop, i*hop+win]. We merge consecutive positives into one interval.
    Returns list of (start_sec, duration_sec).
    """
    mask = mask.astype(bool)
    if mask.size == 0 or not mask.any():
        return []

    idx = np.flatnonzero(mask)
    # split where index jumps > 1
    splits = np.where(np.diff(idx) > 1)[0] + 1
    runs = np.split(idx, splits)

    intervals = []
    for run in runs:
        i0 = int(run[0])
        i1 = int(run[-1])
        start = i0 * hop_sec
        end = i1 * hop_sec + win_sec
        intervals.append((start, end - start))
    return intervals


def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(np.uint8)
    out = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, y_proba) if len(np.unique(y_true)) == 2 else np.nan,
    }
    return out


# -------------------------------------------------
# Main pipeline
# -------------------------------------------------

def main():
    in_dir = Path("../../data/freq_time_features_cache_basic")
    test_dir = Path("../testP70/features_cache_basic")
    files = sorted(in_dir.glob("*_features.npz"))
    if not files:
        raise RuntimeError("No *_features.npz files found")

    # group files by patient (Pxx)
    raw = mne.io.read_raw_edf("../../../data/extra_data/P70_GHB_M1679_0000078.edf", preload=False, verbose="ERROR")
    sfreq = float(raw.info["sfreq"])
    ann = raw.annotations
    print(ann)

    pos_ann_onsets = np.array([on for on, desc in zip(ann.onset, ann.description) if "*" in desc], dtype=float)
    pos_ann_onsets.sort()

    win_sec = 0.5
    hop_sec = 0.25

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
    # Use all patients, except p70, for training and test on p70 full eeg segmented into sliding windows
    # -------------------------------------------------

    test_pid = "P70"
    x_train_list, y_train_list = [], []
    for pid, flist in patient_files.items():
        if pid == test_pid:
            continue
        for f in flist:
            x, y = load_features(f)
            x_train_list.append(x)
            y_train_list.append(y)

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    p70_file = test_dir / "P70_GHB_M1679_0000078_full_features.npz"

    if not p70_file.exists():
        raise RuntimeError(f"File not found: {p70_file}")

    x_test = load_features_X_only(p70_file)

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
    th = 0.30
    y_pred = (y_proba >= th).astype(np.uint8)
    intervals = pd_intervals_from_mne_annotations(raw.annotations)
    y_true = label_windows_by_overlap(
        n_windows=len(y_proba),
        win_sec=win_sec,
        hop_sec=hop_sec,
        intervals=intervals,
        min_overlap_sec=0.0
    )
    # ---------------------------------------------
    # Hard examples:
    # - outside intervals (negative) with highest probability
    # - inside intervals (positive) with lowest probability
    # ---------------------------------------------
    top_k = 50  # πόσα θέλεις να κρατήσεις από κάθε κατηγορία

    centers_sec = np.arange(len(y_proba), dtype=float) * hop_sec + 0.5 * win_sec

    pos_idx = np.where(y_true == 1)[0]  # μέσα στα intervals
    neg_idx = np.where(y_true == 0)[0]  # έξω από τα intervals

    # αρνητικά με μεγαλύτερη πιθανότητα (δηλ. false-alarm-like δύσκολα negatives)
    neg_sorted = neg_idx[np.argsort(y_proba[neg_idx])[::-1]]

    # θετικά με μικρότερη πιθανότητα (δηλ. missed-like δύσκολα positives)
    pos_sorted = pos_idx[np.argsort(y_proba[pos_idx])]

    neg_top = neg_sorted[:top_k]
    pos_top = pos_sorted[:top_k]

    hard_rows = []

    # negative examples: outside intervals, highest probability
    for i in neg_top:
        csec = centers_sec[i]
        hard_rows.append([
            "negative",
            int(i),
            f"{csec:.6f}",
            sec_to_hmsms(csec),
            f"{float(y_proba[i]):.8f}",
        ])

    # positive examples: inside intervals, lowest probability
    for i in pos_top:
        csec = centers_sec[i]
        hard_rows.append([
            "positive",
            int(i),
            f"{csec:.6f}",
            sec_to_hmsms(csec),
            f"{float(y_proba[i]):.8f}",
        ])

    # προαιρετικά: sort όλου του csv κατά πιθανότητα
    # hard_rows.sort(key=lambda row: float(row[4]), reverse=True)

    csv_hard_out = Path(f"../../../data/{test_pid}_hard_windows.csv")

    with csv_hard_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "label",  # positive / negative
            "window_index",
            "center_sec",
            "center_hmsms",
            "probability",
        ])
        w.writerows(hard_rows)

    print(f"Saved hard windows to: {csv_hard_out.resolve()}")

    # ---- confusion masks per window ----
    y_true_b = y_true.astype(bool)
    y_pred_b = y_pred.astype(bool)

    tp = (y_true_b & y_pred_b).astype(np.uint8)
    fp = ((~y_true_b) & y_pred_b).astype(np.uint8)
    fn = (y_true_b & (~y_pred_b)).astype(np.uint8)

    tp_intervals = mask_to_intervals(tp, win_sec, hop_sec)
    fp_intervals = mask_to_intervals(fp, win_sec, hop_sec)
    fn_intervals = mask_to_intervals(fn, win_sec, hop_sec)

    T = len(y_proba) * hop_sec + win_sec

    fig, ax = plt.subplots(figsize=(16, 4))

    row_h = 0.8
    y0 = 0.2

    # GT row (optional, αν θες να υπάρχει πάντα reference)
    gt_intervals = mask_to_intervals(y_true, win_sec, hop_sec)
    # GT row (reference)
    if gt_intervals:
        ax.broken_barh(
            gt_intervals,
            (y0 + 2.0, row_h),
            facecolors="tab:blue",
            label="GT (y_true=1)"
        )

    # TP / FP / FN
    if tp_intervals:
        ax.broken_barh(
            tp_intervals,
            (y0 + 1.0, row_h),
            facecolors="tab:green",
            label="TP (pred=1, true=1)"
        )

    if fp_intervals:
        ax.broken_barh(
            fp_intervals,
            (y0 + 0.0, row_h),
            facecolors="tab:red",
            label="FP (pred=1, true=0)"
        )

    if fn_intervals:
        ax.broken_barh(
            fn_intervals,
            (y0 + 3.0, row_h),
            facecolors="tab:orange",
            label="FN (pred=0, true=1)"
        )

    ax.set_xlim(0, T)
    ax.set_ylim(0, 4.2)
    ax.set_yticks([y0 + 0.4, y0 + 1.4, y0 + 2.4, y0 + 3.4])
    ax.set_yticklabels(["FP", "TP", "GT", "FN"])
    ax.set_xlabel("Time (sec)")
    ax.set_title(f"Timeline vs threshold={th:.2f}")
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()

    print("GT positives ratio:", float(y_true.mean()))
    print(compute_metrics(y_true, y_proba, threshold=0.3))

    print("y_proba stats:",
          "min", float(y_proba.min()),
          "p50", float(np.median(y_proba)),
          "p90", float(np.quantile(y_proba, 0.90)),
          "max", float(y_proba.max()))
    for th in [0.1, 0.5, 0.9]:
        y_pred = (y_proba >= th).astype(np.uint8)
        print(th, "hit_ratio", float(y_pred.mean()))
    threshold_hits = []
    for thres in threshold_values:
        y_pred = (y_proba >= thres).astype(np.uint8)
        pos_idx = np.where(y_pred == 1)[0]

        hit_ratio = len(pos_idx) / len(y_pred)
        threshold_hits.append((thres, hit_ratio))

    thr_arr = np.array([t for t, r in threshold_hits], dtype=float)
    hits_arr = np.array([r for t, r in threshold_hits], dtype=float)

    out_hit_ratio = Path(f"../../../data/{test_pid}_hit_ratio_curve.npz")

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

    csv_out = Path(f"../../../data/{test_pid}_window_timestamps_and_proba.csv")

    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "window_index",
            "start_sec", "end_sec", "center_sec",
            "start_hmsms", "end_hmsms", "center_hmsms",
            "start_sample", "end_sample", "center_sample",
            "proba_y_eq_1",
        ])

        for i, p in enumerate(y_proba):
            start_sec = i * hop_sec
            end_sec = start_sec + win_sec
            center_sec = start_sec + 0.5 * win_sec

            start_sample = int(round(start_sec * sfreq))
            end_sample = int(round(end_sec * sfreq))
            center_sample = int(round(center_sec * sfreq))

            w.writerow([
                i,
                f"{start_sec:.6f}", f"{end_sec:.6f}", f"{center_sec:.6f}",
                sec_to_hmsms(start_sec), sec_to_hmsms(end_sec), sec_to_hmsms(center_sec),
                start_sample, end_sample, center_sample,
                f"{float(p):.8f}",
            ])

    print(f"Saved window timestamps + probabilities to: {csv_out.resolve()}")

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
