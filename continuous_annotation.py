from __future__ import annotations

from pathlib import Path

import numpy as np
import mne
from sklearn.neighbors import KNeighborsClassifier

# ============================================================
# CONFIG
# ============================================================

EDF_PATH = Path(r"../../../Data/extra_data/P70_GHB_M1679_0000078.edf")

EMB_PATH = Path(
    r"../../../Data/labram_embeddings/"
    r"P70_finegrained_continuous_embeddings/"
    r"P70_GHB_M1679_0000078_embeddings.npy"
)

OUT_EDF_PATH = Path(r"../../../Data/produced_data/P70_GHB_M1679_0000078_knn_annotated_negs.edf")

WINDOW_SEC = 1.0
OVERLAP = 0.75
STEP_SEC = WINDOW_SEC * (1.0 - OVERLAP)  # 0.25 s
K = 1

NEG_SUPPORT = [
    "0:20:30",
    "0:30:47.421",
    "0:30:50.007",
    "0:31:22.886",
    "14:46:45.23",
    "14:46:46.23"
]

POS_SUPPORT = [
    "1:29:47.660",
    "1:30:00.855",
    "1:31:03.868",
    "1:30:05.209",
    "1:30:41.903",
    "1:31:04.643",
]


# ============================================================
# HELPERS
# ============================================================

def parse_timestamp_to_seconds(ts: str) -> float:
    parts = ts.strip().split(":")
    if len(parts) == 3:
        h = int(parts[0])
        m = int(parts[1])
        s = float(parts[2])
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m = int(parts[0])
        s = float(parts[1])
        return m * 60 + s
    else:
        raise ValueError(f"Unsupported timestamp format: {ts}")


def window_center_time(window_idx: int, window_sec: float, step_sec: float) -> float:
    return (window_sec / 2.0) + window_idx * step_sec


def nearest_window_index(t_sec: float, window_sec: float, step_sec: float, n_windows: int) -> int:
    idx = int(round((t_sec - window_sec / 2.0) / step_sec))
    return int(np.clip(idx, 0, n_windows - 1))


def positive_windows_to_annotations(
        positive_mask: np.ndarray,
        window_sec: float,
        step_sec: float,
        label: str = "knn_pos",
        orig_time=None,
) -> mne.Annotations:
    onsets = []
    durations = []
    descriptions = []

    pos_idx = np.where(positive_mask)[0]

    for i in pos_idx:
        onsets.append(i * step_sec)
        durations.append(window_sec)
        descriptions.append(label)

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=orig_time,
    )


def keep_only_pd_annotations(annotations: mne.Annotations) -> mne.Annotations:
    if annotations is None or len(annotations) == 0:
        return mne.Annotations(onset=[], duration=[], description=[], orig_time=None)

    descriptions = np.array(annotations.description, dtype=object)
    normalized = np.array([str(d).strip().lower() for d in descriptions], dtype=object)
    keep_desc = {"pd_start", "pd_stop"}
    mask = np.array([d in keep_desc for d in normalized], dtype=bool)

    return mne.Annotations(
        onset=annotations.onset[mask],
        duration=annotations.duration[mask],
        description=descriptions[mask].tolist(),
        orig_time=annotations.orig_time,
    )


def pd_annotations_to_intervals(pd_ann: mne.Annotations) -> list[tuple[float, float]]:
    """
    Converts PD_START / PD_STOP annotations into intervals [(start, stop), ...].
    Assumes chronological order or sorts them by onset.
    """
    if pd_ann is None or len(pd_ann) == 0:
        return []

    items = sorted(
        zip(pd_ann.onset, pd_ann.description),
        key=lambda x: x[0]
    )

    intervals: list[tuple[float, float]] = []
    current_start = None

    for onset, desc in items:
        d = str(desc).strip().lower()

        if d == "pd_start":
            current_start = float(onset)

        elif d == "pd_stop":
            if current_start is not None and onset >= current_start:
                intervals.append((current_start, float(onset)))
                current_start = None

    return intervals


def interval_overlaps_any(window_start: float, window_stop: float, intervals: list[tuple[float, float]]) -> bool:
    """
    Returns True if [window_start, window_stop] overlaps any PD interval [a, b].
    """
    for a, b in intervals:
        if window_start < b and window_stop > a:
            return True
    return False
def windows_inside_pd_to_annotations(
    mask: np.ndarray,
    pd_intervals: list[tuple[float, float]],
    window_sec: float,
    step_sec: float,
    label: str = "knn_neg",
    orig_time=None,
) -> mne.Annotations:
    """
    Keep only windows whose mask is True AND that overlap a PD interval.
    """
    onsets = []
    durations = []
    descriptions = []

    idxs = np.where(mask)[0]

    for i in idxs:
        start = i * step_sec
        stop = start + window_sec

        if interval_overlaps_any(start, stop, pd_intervals):
            onsets.append(start)
            durations.append(window_sec)
            descriptions.append(label)

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=orig_time,
    )

def positive_windows_outside_pd_to_annotations(
        positive_mask: np.ndarray,
        pd_intervals: list[tuple[float, float]],
        window_sec: float,
        step_sec: float,
        label: str = "knn_pos",
        orig_time=None,
) -> mne.Annotations:
    """
    Keep only positive windows that do NOT overlap any PD interval.
    """
    onsets = []
    durations = []
    descriptions = []

    pos_idx = np.where(positive_mask)[0]

    for i in pos_idx:
        start = i * step_sec
        stop = start + window_sec

        if not interval_overlaps_any(start, stop, pd_intervals):
            onsets.append(start)
            durations.append(window_sec)
            descriptions.append(label)

    return mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=orig_time,
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("Loading embeddings...")
    X = np.load(EMB_PATH)
    print("Embeddings shape:", X.shape)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape={X.shape}")

    n_windows, emb_dim = X.shape
    print(f"n_windows={n_windows}, emb_dim={emb_dim}")

    print("\nMapping support timestamps to nearest window centers...")
    neg_support_sec = [parse_timestamp_to_seconds(x) for x in NEG_SUPPORT]
    pos_support_sec = [parse_timestamp_to_seconds(x) for x in POS_SUPPORT]

    neg_idx = [nearest_window_index(t, WINDOW_SEC, STEP_SEC, n_windows) for t in neg_support_sec]
    pos_idx = [nearest_window_index(t, WINDOW_SEC, STEP_SEC, n_windows) for t in pos_support_sec]

    print("Negative support:")
    for ts, idx in zip(NEG_SUPPORT, neg_idx):
        c = window_center_time(idx, WINDOW_SEC, STEP_SEC)
        print(f"  {ts} -> idx={idx}, center={c:.3f}s")

    print("Positive support:")
    for ts, idx in zip(POS_SUPPORT, pos_idx):
        c = window_center_time(idx, WINDOW_SEC, STEP_SEC)
        print(f"  {ts} -> idx={idx}, center={c:.3f}s")

    support_idx = np.array(neg_idx + pos_idx, dtype=int)
    y_support = np.array([0] * len(neg_idx) + [1] * len(pos_idx), dtype=int)
    X_support = X[support_idx]

    print("\nTraining kNN...")
    clf = KNeighborsClassifier(n_neighbors=K, metric="euclidean")
    clf.fit(X_support, y_support)

    print("Predicting all windows...")
    y_pred = clf.predict(X)

    neighbor_indices = clf.kneighbors(X, return_distance=False)
    neighbor_labels = y_support[neighbor_indices]
    pos_frac = neighbor_labels.mean(axis=1)

    print(f"Predicted positives (all): {int(y_pred.sum())} / {len(y_pred)}")
    print(f"Predicted hard negatives (all): {int((pos_frac == 0).sum())} / {len(y_pred)}")
    print(f"Predicted hard positives (all): {int((pos_frac == 1).sum())} / {len(y_pred)}")

    print("\nLoading EDF...")
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True, verbose="ERROR")

    print("\nKeeping only PD_START / PD_STOP from original EDF annotations...")
    pd_ann = keep_only_pd_annotations(raw.annotations)
    print(f"Original PD annotations kept: {len(pd_ann)}")

    pd_intervals = pd_annotations_to_intervals(pd_ann)
    print(f"PD intervals found: {len(pd_intervals)}")

    print("\nCreating window-level kNN annotations outside PD intervals only...")
    print("\nCreating window-level kNN negative annotations inside PD intervals only...")
    hard_negative_mask = (pos_frac == 0.0)

    new_ann = windows_inside_pd_to_annotations(
        mask=hard_negative_mask,
        pd_intervals=pd_intervals,
        window_sec=WINDOW_SEC,
        step_sec=STEP_SEC,
        label="knn_neg",
        orig_time=pd_ann.orig_time,
    )
    print(f"New annotations created outside PD intervals: {len(new_ann)}")

    combined_ann = pd_ann + new_ann
    raw.set_annotations(combined_ann)

    print("\nExporting new EDF...")
    OUT_EDF_PATH.parent.mkdir(parents=True, exist_ok=True)

    mne.export.export_raw(
        OUT_EDF_PATH,
        raw,
        fmt="edf",
        overwrite=True,
    )

    print("\nDone.")
    print("Saved EDF:", OUT_EDF_PATH)

    np.savez_compressed(
        OUT_EDF_PATH.with_suffix(".debug_knn.npz"),
        y_pred=y_pred.astype(np.uint8),
        pos_frac=pos_frac.astype(np.float32),
        support_idx=support_idx,
        support_labels=y_support,
        neg_idx=np.array(neg_idx, dtype=int),
        pos_idx=np.array(pos_idx, dtype=int),
        pd_intervals=np.array(pd_intervals, dtype=float) if len(pd_intervals) > 0 else np.empty((0, 2), dtype=float),
    )
    print("Saved debug predictions:", OUT_EDF_PATH.with_suffix(".debug_knn.npz"))


if __name__ == "__main__":
    main()
