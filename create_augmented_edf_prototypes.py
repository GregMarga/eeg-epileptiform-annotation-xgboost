"""
Build a new EDF that contains:
  - only the PD_START / PD_STOP annotations from the original EDF
  - one annotation per positive prediction from the few-shot CSV,
    with the predicted probability in its description.

Window onset is reconstructed from the CSV's `center_sec` column using
WIN_SEC (the window length used by the few-shot script that produced
the CSV). Make sure WIN_SEC matches that.
"""

from pathlib import Path
import csv
import mne
import numpy as np


# -------------------------------------------------
# Config
# -------------------------------------------------

RAW_EDF_PATH = Path("../../../data/extra_data/P70_GHB_M1679_0000078.edf")
CSV_PATH     = Path("../../../data/P70_5shot_predictions_combined.csv")
OUT_EDF_PATH = Path("../../../data/extra_data/P70_GHB_M1679_0000078_predictions_combined.edf")

# Window length used when producing the CSV. Must match the few-shot script.
# LaBraM: 1.0 ; original HC pipeline: 0.5
WIN_SEC = 1.0

# Annotation kinds to keep from the original EDF
KEEP_DESCRIPTIONS = {"PD_START", "PD_STOP"}


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def read_positive_predictions(csv_path: Path):
    """Yield (onset_sec, duration_sec, probability) for each positive row."""
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if int(row["predicted_label"]) != 1:
                continue
            center_sec = float(row["center_sec"])
            proba = float(row["probability"])
            onset = center_sec - 0.5 * WIN_SEC
            yield onset, WIN_SEC, proba


def filter_pd_annotations(annotations: mne.Annotations) -> mne.Annotations:
    """Keep only PD_START / PD_STOP from the original annotations."""
    keep_mask = np.array([
        d in KEEP_DESCRIPTIONS for d in annotations.description
    ], dtype=bool)

    return mne.Annotations(
        onset=annotations.onset[keep_mask],
        duration=annotations.duration[keep_mask],
        description=annotations.description[keep_mask],
        orig_time=annotations.orig_time,
    )


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print(f"Loading raw EDF: {RAW_EDF_PATH}")
    raw = mne.io.read_raw_edf(RAW_EDF_PATH, preload=True, verbose="ERROR")

    # 1) Keep only PD_START / PD_STOP from the original annotations
    pd_anns = filter_pd_annotations(raw.annotations)
    print(f"  Original annotations: {len(raw.annotations)}")
    print(f"  Kept PD annotations:  {len(pd_anns)}")

    # 2) Build positive-prediction annotations from CSV
    onsets = []
    durations = []
    descriptions = []
    for onset, duration, proba in read_positive_predictions(CSV_PATH):
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(f"pred=1 p={proba:.4f}")

    print(f"  Positive predictions: {len(onsets)}")

    # 3) Combine
    all_onsets       = np.concatenate([pd_anns.onset,       np.array(onsets, dtype=float)])
    all_durations    = np.concatenate([pd_anns.duration,    np.array(durations, dtype=float)])
    all_descriptions = np.concatenate([pd_anns.description, np.array(descriptions, dtype=object)])

    # Sort by onset for cleanliness (optional but nicer when inspecting)
    order = np.argsort(all_onsets, kind="stable")
    combined = mne.Annotations(
        onset=all_onsets[order],
        duration=all_durations[order],
        description=all_descriptions[order],
        orig_time=raw.annotations.orig_time,
    )

    raw.set_annotations(combined)
    print(f"  Total annotations on output: {len(combined)}")

    # 4) Export to a new EDF. Requires `edfio` (or `pyedflib` on older MNE).
    OUT_EDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nExporting to: {OUT_EDF_PATH}")
    mne.export.export_raw(
        str(OUT_EDF_PATH),
        raw,
        fmt="edf",
        overwrite=True,
        verbose="ERROR",
    )
    print("Done.")


if __name__ == "__main__":
    main()