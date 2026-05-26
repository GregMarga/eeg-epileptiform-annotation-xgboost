"""
Compare HC few-shot, XGBoost, and LaBraM few-shot against Jeroen's
exhaustive annotations on the 5 selected segments.

Window-level evaluation:
  - TP = positive window that contains at least one peak inside [onset, end)
  - FP = positive window with no peak
  - TN = negative window with no peak
  - FN = peak that is not contained in any positive window  (counted per peak,
         so the same peak isn't counted twice when it falls in two overlapping
         windows)

Reports per-model:
  - Per-segment metrics at threshold 0.5
  - Aggregate metrics at threshold 0.5
  - Threshold sweep: precision, recall, F1 over 0.05..0.95
  - AUPRC (area under precision-recall curve via sklearn on raw window scores)
"""

from pathlib import Path
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


# -------------------------------------------------
# Config
# -------------------------------------------------

JEROEN_EDF = Path("../../../data/extra_data/P70_GHB_M1679_0000078_segments.edf")
OUT_DIR    = Path("../../../data/comparison_against_jeroen")

PEAK_LABEL = "*"
DEFAULT_THR = 0.5

MODELS = [
    {"name": "HC",     "csv": Path("../../../data/P70_5shot_predictions.csv"),
     "win_sec": 0.5, "prob_col": "probability"},
    {"name": "XGB",    "csv": Path("../../../data/P70_window_timestamps_and_proba.csv"),
     "win_sec": 0.5, "prob_col": "proba_y_eq_1"},
    {"name": "LaBraM", "csv": Path("../../../data/P70_5shot_predictions_labram.csv"),
     "win_sec": 1.0, "prob_col": "probability"},
]

SEGMENTS = [
    ("Seg1 04:22", 15720.0, 15780.0),
    ("Seg2 12:48", 46080.0, 46140.0),
    ("Seg3 01:39",  5940.0,  6000.0),
    ("Seg4 14:03", 50580.0, 50640.0),
    ("Seg5 04:10", 15000.0, 15060.0),
]

THRESHOLDS = np.linspace(0.05, 0.95, 19)


# -------------------------------------------------
# Loaders
# -------------------------------------------------

def load_model_predictions(model_cfg):
    df = pd.read_csv(model_cfg["csv"])
    win = model_cfg["win_sec"]
    df["onset"] = df["center_sec"] - 0.5 * win
    df["end"]   = df["center_sec"] + 0.5 * win
    df["proba"] = df[model_cfg["prob_col"]].astype(float)
    return df[["center_sec", "onset", "end", "proba"]].sort_values("center_sec").reset_index(drop=True)


def load_peaks(edf_path, label=PEAK_LABEL):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    ann = raw.annotations
    mask = np.array([d == label for d in ann.description])
    return np.array(ann.onset[mask], dtype=float)


# -------------------------------------------------
# Window-level metrics for a single segment / threshold
# -------------------------------------------------

def evaluate_window_segment(pred_df, peaks, t0, t1, threshold):
    """Returns dict with TP, FP, TN, FN, n_windows, n_peaks for one segment."""
    sub = pred_df[(pred_df.center_sec >= t0) & (pred_df.center_sec < t1)].copy()
    seg_peaks = peaks[(peaks >= t0) & (peaks < t1)]

    starts = sub["onset"].values
    ends   = sub["end"].values
    preds  = (sub["proba"].values >= threshold).astype(int)

    # Each window: does it contain any peak?
    if len(seg_peaks) > 0 and len(sub) > 0:
        contains = (seg_peaks[:, None] >= starts[None, :]) & (seg_peaks[:, None] < ends[None, :])
        has_peak = contains.any(axis=0)
    else:
        has_peak = np.zeros(len(sub), dtype=bool)

    tp = int(((preds == 1) & has_peak).sum())
    fp = int(((preds == 1) & ~has_peak).sum())
    tn = int(((preds == 0) & ~has_peak).sum())

    # FN counted per PEAK, so a peak missed across multiple overlapping windows
    # only counts once.
    if len(seg_peaks) > 0 and len(sub) > 0:
        pos_starts = starts[preds == 1]
        pos_ends   = ends[preds == 1]
        if len(pos_starts) > 0:
            covered = (
                (seg_peaks[:, None] >= pos_starts[None, :])
                & (seg_peaks[:, None] < pos_ends[None, :])
            ).any(axis=1)
        else:
            covered = np.zeros(len(seg_peaks), dtype=bool)
        fn = int((~covered).sum())
    else:
        fn = len(seg_peaks)

    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "n_windows": len(sub), "n_peaks": len(seg_peaks)}


def aggregate_metrics(pred_df, peaks, segments, threshold):
    agg = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n_windows": 0, "n_peaks": 0}
    per_seg = []
    for label, t0, t1 in segments:
        m = evaluate_window_segment(pred_df, peaks, t0, t1, threshold)
        per_seg.append((label, m))
        for k in agg:
            agg[k] += m[k]
    return agg, per_seg


def compute_pr_f1_acc(m):
    tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rec  = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1   = 2*prec*rec/(prec+rec) if prec and rec and not np.isnan(prec+rec) else float("nan")
    acc  = (tp + tn) / m["n_windows"] if m["n_windows"] > 0 else float("nan")
    return prec, rec, f1, acc


# -------------------------------------------------
# AUPRC on raw window-level probabilities
# -------------------------------------------------

def window_level_auprc(pred_df, peaks, segments):
    """Pool window scores + binary labels (peak inside window) across all
    segments, then sklearn average_precision_score. This is the standard
    window-level AUPRC and avoids event-merging artifacts."""
    all_y    = []
    all_scor = []
    for _, t0, t1 in segments:
        sub = pred_df[(pred_df.center_sec >= t0) & (pred_df.center_sec < t1)]
        seg_peaks = peaks[(peaks >= t0) & (peaks < t1)]
        starts = sub["onset"].values
        ends   = sub["end"].values
        scores = sub["proba"].values
        if len(seg_peaks) > 0 and len(sub) > 0:
            has_peak = (
                (seg_peaks[:, None] >= starts[None, :])
                & (seg_peaks[:, None] < ends[None, :])
            ).any(axis=0)
        else:
            has_peak = np.zeros(len(sub), dtype=bool)
        all_y.append(has_peak.astype(int))
        all_scor.append(scores)
    y    = np.concatenate(all_y)
    scor = np.concatenate(all_scor)
    if y.sum() == 0:
        return float("nan"), y, scor
    return float(average_precision_score(y, scor)), y, scor


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print(f"Loading peaks: {JEROEN_EDF}")
    peaks = load_peaks(JEROEN_EDF)
    n_peaks_total = sum(int(((peaks >= t0) & (peaks < t1)).sum()) for _, t0, t1 in SEGMENTS)
    print(f"  Total peaks in 5 segments: {n_peaks_total}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    pr_curves = {}

    for cfg in MODELS:
        name = cfg["name"]
        print(f"\n===== Model: {name} (window={cfg['win_sec']}s) =====")
        pred_df = load_model_predictions(cfg)

        # Per-segment + aggregate at thr=0.5
        agg, per_seg = aggregate_metrics(pred_df, peaks, SEGMENTS, DEFAULT_THR)
        print(f"  -- Per-segment metrics @ thr={DEFAULT_THR} --")
        for label, m in per_seg:
            p, r, f1, acc = compute_pr_f1_acc(m)
            print(f"    {label}:  TP={m['tp']:>3}  FP={m['fp']:>3}  "
                  f"TN={m['tn']:>3}  FN={m['fn']:>3}  "
                  f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}  Acc={acc:.3f}")
        p, r, f1, acc = compute_pr_f1_acc(agg)
        print(f"  -- Aggregate @ thr={DEFAULT_THR} --")
        print(f"    TP={agg['tp']}  FP={agg['fp']}  TN={agg['tn']}  FN={agg['fn']}")
        print(f"    Precision={p:.3f}  Recall={r:.3f}  F1={f1:.3f}  Accuracy={acc:.3f}")

        # AUPRC over raw window scores (uses sklearn, no manual sweep)
        ap, y_pool, score_pool = window_level_auprc(pred_df, peaks, SEGMENTS)
        print(f"  AUPRC (window-level, sklearn): {ap:.3f}")

        # PR curve from sklearn for plotting
        precisions, recalls, _ = precision_recall_curve(y_pool, score_pool)
        pr_curves[name] = (precisions, recalls, ap)

        # Threshold sweep: P/R/F1 at each threshold (for the user to inspect)
        sweep_rows = []
        for thr in THRESHOLDS:
            m_thr, _ = aggregate_metrics(pred_df, peaks, SEGMENTS, thr)
            p_t, r_t, f1_t, acc_t = compute_pr_f1_acc(m_thr)
            sweep_rows.append({"threshold": thr, "tp": m_thr["tp"], "fp": m_thr["fp"],
                               "fn": m_thr["fn"], "precision": p_t, "recall": r_t,
                               "f1": f1_t, "accuracy": acc_t})
        sweep_df = pd.DataFrame(sweep_rows)
        sweep_df.to_csv(OUT_DIR / f"threshold_sweep_{name}.csv", index=False)

        # Best F1 in the sweep
        best = sweep_df.loc[sweep_df["f1"].idxmax()]
        print(f"  Best F1 in sweep: thr={best['threshold']:.2f}  "
              f"P={best['precision']:.3f}  R={best['recall']:.3f}  "
              f"F1={best['f1']:.3f}  Acc={best['accuracy']:.3f}")

        summary_rows.append({
            "model":     name,
            "win_sec":   cfg["win_sec"],
            "P@0.5":     p, "R@0.5": r, "F1@0.5": f1, "Acc@0.5": acc,
            "AUPRC":     ap,
            "best_thr":  best["threshold"],
            "F1@best":   best["f1"],
            "Acc@best":  best["accuracy"],
        })

    # ----- Comparison summary -----
    summary = pd.DataFrame(summary_rows)
    print("\n" + "=" * 100)
    print("MODEL COMPARISON (window-level, aggregated over 5 segments)")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    summary.to_csv(OUT_DIR / "comparison_summary.csv", index=False)

    # ----- PR curve plot -----
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, (precs, recs, ap) in pr_curves.items():
        ax.plot(recs, precs, label=f"{name} (AUPRC={ap:.3f})")
    # Random baseline = positive prevalence
    n_windows_total = sum(
        int(((load_model_predictions(MODELS[0]).center_sec >= t0) &
             (load_model_predictions(MODELS[0]).center_sec < t1)).sum())
        for _, t0, t1 in SEGMENTS
    )
    # Approximation: peaks usually fall in 2 windows (50% overlap), so positive
    # prevalence ≈ 2 * n_peaks / n_windows. Use this only as a rough baseline.
    base = min(1.0, 2 * n_peaks_total / max(n_windows_total, 1))
    ax.axhline(base, color="gray", linestyle="--", alpha=0.5,
               label=f"random baseline ≈ {base:.3f}")
    ax.set_xlabel("Recall (window-level)")
    ax.set_ylabel("Precision (window-level)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_title("Window-level PR curves vs Jeroen's exhaustive annotations\n"
                 "(5 segments × 60 s, ~5 minutes total)")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "pr_curves.png", dpi=150)
    print(f"\nSaved: {OUT_DIR/'pr_curves.png'}")
    print(f"Saved: {OUT_DIR/'comparison_summary.csv'}")


if __name__ == "__main__":
    main()