"""
Read the few-shot predictions CSV and the original EDF, and produce a new
EDF that contains:
  - the original PD_START / PD_STOP annotations (kept untouched)
  - one annotation at the center of every POSITIVE predicted window,
    labelled with its probability

Also prints the number and percentage of positive windows.
"""

from pathlib import Path
import csv
import mne


# -------------------------------------------------
# Config
# -------------------------------------------------

RAW_EDF_PATH = Path("../../../data/extra_data/P70_GHB_M1679_0000078.edf")
CSV_PATH     = Path("../../../data/P70_5shot_predictions.csv")
OUT_EDF_PATH = Path("../../../data/P70_with_positive_predictions.edf")


# -------------------------------------------------
# CSV loading
# -------------------------------------------------

def load_positive_rows(csv_path: Path) -> tuple[list[dict], int]:
    """
    Return (positive_rows, total_rows).
    A row is "positive" when predicted_label == 1.
    """
    positives = []
    total = 0

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if int(row["predicted_label"]) == 1:
                positives.append({
                    "window_index": int(row["window_index"]),
                    "center_sec":   float(row["center_sec"]),
                    "probability":  float(row["probability"]),
                })

    return positives, total


# -------------------------------------------------
# Annotation helpers
# -------------------------------------------------

def keep_only_pd_annotations(raw: mne.io.BaseRaw) -> mne.Annotations:
    """Return only PD_START / PD_STOP from the original annotations."""
    keep_onset = []
    keep_duration = []
    keep_desc = []

    for onset, duration, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        desc_up = desc.strip().upper()
        if desc_up in {"PD_START", "PD_STOP"}:
            keep_onset.append(float(onset))
            keep_duration.append(float(duration))
            keep_desc.append(desc_up)

    orig_time = raw.annotations.orig_time
    if orig_time is None:
        orig_time = raw.info.get("meas_date", None)

    return mne.Annotations(
        onset=keep_onset,
        duration=keep_duration,
        description=keep_desc,
        orig_time=orig_time,
    )


def build_positive_annotations(
    pos_rows: list[dict],
    orig_time,
    duration: float = 0.0,
) -> mne.Annotations:
    """One point annotation per positive prediction, encoding the probability."""
    onsets = [r["center_sec"] for r in pos_rows]
    descriptions = [
        f"POSITIVE|p={r['probability']:.6f}"
        for r in pos_rows
    ]

    return mne.Annotations(
        onset=onsets,
        duration=[float(duration)] * len(onsets),
        description=descriptions,
        orig_time=orig_time,
    )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print(f"Reading EDF: {RAW_EDF_PATH}")
    raw = mne.io.read_raw_edf(RAW_EDF_PATH, preload=False, verbose="ERROR")

    print(f"Reading predictions CSV: {CSV_PATH}")
    pos_rows, total_rows = load_positive_rows(CSV_PATH)

    n_pos = len(pos_rows)
    pct_pos = (100.0 * n_pos / total_rows) if total_rows > 0 else 0.0

    print(f"\nPositive windows: {n_pos} / {total_rows} ({pct_pos:.2f}%)")

    if n_pos > 0:
        probs = [r["probability"] for r in pos_rows]
        print(f"Probability among positives: "
              f"min={min(probs):.4f}  median={sorted(probs)[len(probs)//2]:.4f}  "
              f"max={max(probs):.4f}")

    # Keep only the doctor's PD intervals from the original annotations,
    # then append our positive predictions on top.
    pd_annotations = keep_only_pd_annotations(raw)
    pos_annotations = build_positive_annotations(
        pos_rows=pos_rows,
        orig_time=pd_annotations.orig_time,
        duration=0.0,
    )

    raw_out = raw.copy()
    raw_out.set_annotations(pd_annotations + pos_annotations)

    OUT_EDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_out.export(OUT_EDF_PATH, fmt="edf", overwrite=True)

    print(f"\nKept PD annotations:        {len(pd_annotations)}")
    print(f"Added POSITIVE annotations: {len(pos_annotations)}")
    print(f"Saved EDF to: {OUT_EDF_PATH.resolve()}")


if __name__ == "__main__":
    main()