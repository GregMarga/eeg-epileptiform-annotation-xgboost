"""
Trivial baseline: a classifier that predicts "negative" for every window.

Computes the accuracy this baseline achieves on the 5 selected segments,
to put the model accuracies in context (class imbalance is heavy, so a
do-nothing classifier already scores high).

Uses the HC predictions CSV only as a source of window timestamps -- the
actual probabilities are ignored, since we override every prediction to 0.
"""

from pathlib import Path
import mne
import numpy as np
import pandas as pd


# -------------------------------------------------
# Config
# -------------------------------------------------

JEROEN_EDF = Path("../../../data/extra_data/P70_GHB_M1679_0000078_segments.edf")

# Pick one model's CSV for the window timing. Both HC and XGB use 0.5 s
# windows, LaBraM uses 1 s. Set WIN_SEC accordingly.
WINDOW_SOURCES = [
    {"name": "HC/XGB (0.5 s windows)",
     "csv": Path("../../../data/P70_5shot_predictions.csv"),
     "win_sec": 0.5},
    {"name": "LaBraM (1.0 s windows)",
     "csv": Path("../../../data/P70_5shot_predictions_labram.csv"),
     "win_sec": 1.0},
]

PEAK_LABEL = "*"

SEGMENTS = [
    ("Seg1 04:22", 15720.0, 15780.0),
    ("Seg2 12:48", 46080.0, 46140.0),
    ("Seg3 01:39",  5940.0,  6000.0),
    ("Seg4 14:03", 50580.0, 50640.0),
    ("Seg5 04:10", 15000.0, 15060.0),
]


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def load_windows(csv_path, win_sec):
    df = pd.read_csv(csv_path)
    df["onset"] = df["center_sec"] - 0.5 * win_sec
    df["end"]   = df["center_sec"] + 0.5 * win_sec
    return df[["center_sec", "onset", "end"]].sort_values("center_sec").reset_index(drop=True)


def load_peaks(edf_path, label=PEAK_LABEL):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    ann = raw.annotations
    mask = np.array([d == label for d in ann.description])
    return np.array(ann.onset[mask], dtype=float)


def evaluate_always_negative(df, peaks, t0, t1):
    """All predictions = 0. Compute TP, FP, TN, FN_per_peak."""
    sub = df[(df.center_sec >= t0) & (df.center_sec < t1)].copy()
    seg_peaks = peaks[(peaks >= t0) & (peaks < t1)]

    starts = sub["onset"].values
    ends   = sub["end"].values

    if len(seg_peaks) > 0 and len(sub) > 0:
        contains = (seg_peaks[:, None] >= starts[None, :]) & (seg_peaks[:, None] < ends[None, :])
        has_peak = contains.any(axis=0)
    else:
        has_peak = np.zeros(len(sub), dtype=bool)

    # All predictions are 0
    tp = 0
    fp = 0
    tn = int((~has_peak).sum())
    # FN = every peak (since none are predicted)
    fn = len(seg_peaks)

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_windows": len(sub), "n_peaks": len(seg_peaks)}


def compute_metrics(m):
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if prec and rec else 0.0
    acc  = (tp + tn) / m["n_windows"] if m["n_windows"] > 0 else float("nan")
    return prec, rec, f1, acc


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    peaks = load_peaks(JEROEN_EDF)
    n_peaks_total = sum(int(((peaks >= t0) & (peaks < t1)).sum()) for _, t0, t1 in SEGMENTS)
    print(f"Total peaks across 5 segments: {n_peaks_total}\n")

    for src in WINDOW_SOURCES:
        print(f"===== Trivial all-negative classifier — {src['name']} =====")
        df = load_windows(src["csv"], src["win_sec"])

        agg = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n_windows": 0, "n_peaks": 0}
        print(f"  {'Segment':<14} {'n_win':>6} {'n_peak':>7} {'TN':>4} {'Acc':>6}")
        for label, t0, t1 in SEGMENTS:
            m = evaluate_always_negative(df, peaks, t0, t1)
            _, _, _, acc = compute_metrics(m)
            print(f"  {label:<14} {m['n_windows']:>6} {m['n_peaks']:>7} "
                  f"{m['tn']:>4} {acc:>6.3f}")
            for k in agg:
                agg[k] += m[k]

        prec, rec, f1, acc = compute_metrics(agg)
        print(f"\n  Aggregate: n_windows={agg['n_windows']}  n_peaks={agg['n_peaks']}")
        print(f"             TP={agg['tp']}  FP={agg['fp']}  TN={agg['tn']}  FN={agg['fn']}")
        print(f"             Precision={prec}  Recall={rec:.3f}  F1={f1:.3f}  "
              f"Accuracy={acc:.3f}\n")


if __name__ == "__main__":
    main()