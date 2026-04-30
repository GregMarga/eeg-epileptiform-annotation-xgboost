"""
Build a Gantt-style timeline plot from:
  - The 5-shot predictions CSV  (predicted_label per sliding window)
  - The original EDF            (PD_START / PD_STOP annotations as ground truth)

Four rows are drawn:
  GT  : doctor's PD intervals (positives)
  TP  : pred=1 inside a PD interval
  FP  : pred=1 outside any PD interval
  FN  : pred=0 inside a PD interval
"""

from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
import mne

# -------------------------------------------------
# Config
# -------------------------------------------------

EDF_PATH = Path("../../../data/extra_data/P70_GHB_M1679_0000078.edf")
CSV_PATH = Path("../../../data/P70_5shot_predictions_labram.csv")
OUT_PNG = Path("../../../data/P70_timeline_labram.png")

# Sliding window timing — must match the script that produced the CSV
WIN_SEC = 0.5
HOP_SEC = 0.25

# Minimum window-vs-interval overlap to call the window "inside" a PD interval.
# 0.0 = any overlap counts as positive.
MIN_OVERLAP_SEC = 0.0


# -------------------------------------------------
# CSV loading
# -------------------------------------------------

def load_predictions(csv_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (window_index, predicted_label, center_sec) as parallel arrays,
    sorted by window_index."""
    indices, preds, centers = [], [], []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            indices.append(int(row["window_index"]))
            preds.append(int(row["predicted_label"]))
            centers.append(float(row["center_sec"]))

    order = np.argsort(indices)
    return (
        np.array(indices, dtype=int)[order],
        np.array(preds, dtype=np.uint8)[order],
        np.array(centers, dtype=float)[order],
    )


# -------------------------------------------------
# Ground truth from PD_START / PD_STOP
# -------------------------------------------------

def pd_intervals_from_annotations(ann: mne.Annotations) -> np.ndarray:
    """
    Pair PD_START / PD_STOP annotations into intervals.
    Returns array of shape (K, 2): [[start_sec, stop_sec], ...].
    """
    events = sorted(zip(ann.onset, ann.description), key=lambda x: x[0])

    intervals = []
    current_start = None
    for t, desc in events:
        d = desc.strip().upper()
        if d == "PD_START":
            current_start = float(t)
        elif d == "PD_STOP":
            if current_start is not None and t > current_start:
                intervals.append((current_start, float(t)))
            current_start = None

    return np.array(intervals, dtype=float) if intervals else np.zeros((0, 2))


def label_windows_by_overlap(
        n_windows: int,
        win_sec: float,
        hop_sec: float,
        intervals: np.ndarray,
        min_overlap_sec: float = 0.0,
) -> np.ndarray:
    """For each window, 1 if it overlaps any PD interval by more than min_overlap_sec."""
    starts = np.arange(n_windows, dtype=float) * hop_sec
    ends = starts + win_sec

    y = np.zeros(n_windows, dtype=np.uint8)
    if intervals.size == 0:
        return y

    for a, b in intervals:
        overlap = np.minimum(ends, b) - np.maximum(starts, a)
        y |= (overlap > min_overlap_sec).astype(np.uint8)

    return y


# -------------------------------------------------
# Window mask -> contiguous time intervals
# -------------------------------------------------

def mask_to_intervals(mask: np.ndarray, win_sec: float, hop_sec: float):
    """Convert a binary window mask to (start_sec, duration_sec) tuples,
    merging consecutive positive windows into a single interval."""
    mask = mask.astype(bool)
    if mask.size == 0 or not mask.any():
        return []

    idx = np.flatnonzero(mask)
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


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print(f"Reading EDF: {EDF_PATH}")
    raw = mne.io.read_raw_edf(EDF_PATH, preload=False, verbose="ERROR")

    print(f"Reading predictions CSV: {CSV_PATH}")
    win_idx, y_pred, centers_sec = load_predictions(CSV_PATH)
    n_windows = len(win_idx)
    print(f"  Loaded {n_windows} windows")

    # Build GT from EDF annotations
    pd_intervals = pd_intervals_from_annotations(raw.annotations)
    print(f"  PD intervals from annotations: {len(pd_intervals)}")

    y_true = label_windows_by_overlap(
        n_windows=n_windows,
        win_sec=WIN_SEC,
        hop_sec=HOP_SEC,
        intervals=pd_intervals,
        min_overlap_sec=MIN_OVERLAP_SEC,
    )

    # Confusion masks per window
    y_true_b = y_true.astype(bool)
    y_pred_b = y_pred.astype(bool)

    tp_mask = (y_true_b & y_pred_b)
    fp_mask = (~y_true_b & y_pred_b)
    fn_mask = (y_true_b & ~y_pred_b)

    # Counts
    n_tp = int(tp_mask.sum())
    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    n_pos_pred = int(y_pred_b.sum())
    n_pos_true = int(y_true_b.sum())

    print(f"\nWindow-level counts:")
    print(f"  GT positives:  {n_pos_true} / {n_windows} ({100.0 * n_pos_true / n_windows:.2f}%)")
    print(f"  Predicted=1:   {n_pos_pred} / {n_windows} ({100.0 * n_pos_pred / n_windows:.2f}%)")
    print(f"  TP={n_tp}  FP={n_fp}  FN={n_fn}")

    if n_pos_true > 0:
        recall = n_tp / n_pos_true
        print(f"  Recall      = TP / GT_pos     = {recall:.4f}")
    if n_pos_pred > 0:
        precision = n_tp / n_pos_pred
        print(f"  Precision   = TP / Pred_pos   = {precision:.4f}")

    # Convert masks to contiguous broken_barh intervals
    gt_intervals = [(float(a), float(b - a)) for a, b in pd_intervals]
    tp_intervals = mask_to_intervals(tp_mask, WIN_SEC, HOP_SEC)
    fp_intervals = mask_to_intervals(fp_mask, WIN_SEC, HOP_SEC)
    fn_intervals = mask_to_intervals(fn_mask, WIN_SEC, HOP_SEC)

    # Plot
    T = n_windows * HOP_SEC + WIN_SEC

    fig, ax = plt.subplots(figsize=(16, 4))

    row_h = 0.8
    y0 = 0.2

    if fn_intervals:
        ax.broken_barh(
            fn_intervals, (y0 + 3.0, row_h),
            facecolors="tab:orange",
            label=f"FN (pred=0, true=1) — {n_fn}",
        )
    if gt_intervals:
        ax.broken_barh(
            gt_intervals, (y0 + 2.0, row_h),
            facecolors="tab:blue",
            label=f"GT (y_true=1) — {n_pos_true}",
        )
    if tp_intervals:
        ax.broken_barh(
            tp_intervals, (y0 + 1.0, row_h),
            facecolors="tab:green",
            label=f"TP (pred=1, true=1) — {n_tp}",
        )
    if fp_intervals:
        ax.broken_barh(
            fp_intervals, (y0 + 0.0, row_h),
            facecolors="tab:red",
            label=f"FP (pred=1, true=0) — {n_fp}",
        )

    ax.set_xlim(0, T)
    ax.set_ylim(0, 4.2)
    ax.set_yticks([y0 + 0.4, y0 + 1.4, y0 + 2.4, y0 + 3.4])
    ax.set_yticklabels(["FP", "TP", "GT", "FN"])
    ax.set_xlabel("Time (sec)")
    ax.set_title(
        f"P70 combined— 5-shot prototypical predictions vs PD ground truth"
    )
    ax.grid(True, axis="x", alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PNG, dpi=150)
    print(f"\nSaved plot to: {OUT_PNG.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()
