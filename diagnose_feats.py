"""
diagnose_preprocessing_separability.py
======================================
Why do two P58 test sets built with the SAME feature code, SAME labeling rule
and evaluated with the SAME model give balanced-accuracy 0.53 vs 0.71?

The only thing that differs is the PREPROCESSING that produced the windows:

  features_1s_labeled_1_40hz  -> full-recording filter 0.5-40Hz, 80Hz,
                                 static bad-channel report, 24h average reference
  features_1s_labeled         -> per-segment crop+pad, 0.1-75Hz + notch, 200Hz,
                                 per-segment PyPREP, per-segment average reference

This script isolates the preprocessing effect by measuring, per feature, how well
each RAW feature separates POSITIVE from NEGATIVE windows -- independent of any
model, threshold or z-score. Two scale-invariant metrics are used:

  * single-feature ROC AUC : rank-based separability (0.5 = chance)
  * Cohen's d              : standardized mean difference (pos - neg)

Both are invariant to feature scaling, so differences reflect ONLY how the
preprocessing shapes the pos/neg contrast -- not units or normalization.

The training distribution is included as the reference the model actually learned.
Whichever P58 variant matches the training separability profile is the one whose
preprocessing is consistent with the training features.
"""

from pathlib import Path
import re
import numpy as np
from sklearn.metrics import roc_auc_score

# -------------------------------------------------
# Paths -- adjust if needed
# -------------------------------------------------
TRAIN_DIR = Path("../../../data/80hz_freq_time_features_pyprep_1s")

# The two P58 variants. Labels are just display names; the folder decides content.
P58_VARIANTS = {
    "P58 [features_1s_labeled]":        Path("../../../data/evaluation_recordings/features_1s_labeled"),
    "P58 [features_1s_labeled_1_40hz]": Path("../../../data/evaluation_recordings/features_1s_labeled_1_40hz"),
}

EVAL_PATIENT_ID = "P58"
IGNORE_LABEL    = -1
SUFFIX          = "_features.npz"

FEATURE_NAMES_16 = [
    "zero_cross", "maxima", "minima", "rms", "skew", "kurt_excess",
    "total_power_1_40", "peak_freq_1_40",
    "mean_band_delta", "mean_band_theta", "mean_band_alpha", "mean_band_beta",
    "norm_band_delta", "norm_band_theta", "norm_band_alpha", "norm_band_beta",
]


# -------------------------------------------------
# Loading -- identical channel-averaging to the XGBoost eval loader
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_handcrafted_file(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float64)
    n_channels = len(z["ch_names"])
    n_samples  = X.shape[0]
    X = X.reshape(n_samples, n_channels, 16).mean(axis=1)   # -> (N, 16)
    y = z["y"].astype(np.int64).ravel()
    keep = y != IGNORE_LABEL
    return X[keep], y[keep].astype(np.uint8)


def load_dir(directory: Path, only_patient: str | None = None,
             exclude_patient: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    files = sorted(directory.glob(f"*{SUFFIX}"))
    if not files:
        raise RuntimeError(f"No *{SUFFIX} files in {directory}")
    xs, ys = [], []
    for f in files:
        pid = patient_from_filename(f.name)
        if only_patient and pid != only_patient:
            continue
        if exclude_patient and pid == exclude_patient:
            continue
        x, y = load_handcrafted_file(f)
        xs.append(x); ys.append(y)
    if not xs:
        raise RuntimeError(f"No matching files in {directory} "
                           f"(only={only_patient}, exclude={exclude_patient})")
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# -------------------------------------------------
# Separability metrics (per feature, scale-invariant)
# -------------------------------------------------

def single_feature_auc(x: np.ndarray, y: np.ndarray) -> float:
    """Raw-feature ROC AUC for pos vs neg. NaNs dropped. Returns nan if degenerate."""
    m = np.isfinite(x)
    xf, yf = x[m], y[m]
    if len(np.unique(yf)) < 2:
        return float("nan")
    return float(roc_auc_score(yf, xf))


def cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    """Standardized mean difference (pos - neg), pooled SD. NaNs ignored."""
    xp = x[(y == 1) & np.isfinite(x)]
    xn = x[(y == 0) & np.isfinite(x)]
    if len(xp) < 2 or len(xn) < 2:
        return float("nan")
    mp, mn = xp.mean(), xn.mean()
    vp, vn = xp.var(ddof=1), xn.var(ddof=1)
    npos, nneg = len(xp), len(xn)
    pooled = np.sqrt(((npos - 1) * vp + (nneg - 1) * vn) / (npos + nneg - 2))
    if pooled == 0:
        return float("nan")
    return float((mp - mn) / pooled)


def separability_profile(X: np.ndarray, y: np.ndarray) -> dict[str, tuple[float, float]]:
    """Per feature -> (auc, cohens_d)."""
    prof = {}
    for j, name in enumerate(FEATURE_NAMES_16):
        prof[name] = (single_feature_auc(X[:, j], y), cohens_d(X[:, j], y))
    return prof


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print("Loading datasets...\n")

    # Training reference: all patients EXCEPT P58, pooled.
    Xtr, ytr = load_dir(TRAIN_DIR, exclude_patient=EVAL_PATIENT_ID)
    print(f"train (all except {EVAL_PATIENT_ID}): "
          f"windows={len(ytr)}  pos={int(ytr.sum())}  neg={int((ytr==0).sum())}")

    profiles = {"TRAIN": separability_profile(Xtr, ytr)}
    class_balance = {"TRAIN": (int(ytr.sum()), int((ytr == 0).sum()))}

    for label, path in P58_VARIANTS.items():
        try:
            X, y = load_dir(path, only_patient=EVAL_PATIENT_ID)
        except RuntimeError as e:
            print(f"  [SKIP] {label}: {e}")
            continue
        print(f"{label}: windows={len(y)}  pos={int(y.sum())}  neg={int((y==0).sum())}")
        profiles[label] = separability_profile(X, y)
        class_balance[label] = (int(y.sum()), int((y == 0).sum()))

    cols = list(profiles.keys())

    # ---- AUC table (signed: >0.5 means pos has higher values) ----
    print("\n" + "=" * 100)
    print("Single-feature ROC AUC  (0.5 = chance; distance from 0.5 = separability)")
    print("=" * 100)
    header = f"{'feature':<20}" + "".join(f"{c:>26}" for c in cols)
    print(header)
    print("-" * len(header))
    for name in FEATURE_NAMES_16:
        row = f"{name:<20}"
        for c in cols:
            auc, _ = profiles[c][name]
            sep = abs(auc - 0.5) if np.isfinite(auc) else float("nan")
            row += f"{auc:>18.3f} (|{sep:.2f}|)" if np.isfinite(auc) else f"{'nan':>26}"
        print(row)

    # ---- Cohen's d table ----
    print("\n" + "=" * 100)
    print("Cohen's d  (pos - neg, standardized; |d|>0.8 large, 0.5 medium, 0.2 small)")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    for name in FEATURE_NAMES_16:
        row = f"{name:<20}"
        for c in cols:
            _, d = profiles[c][name]
            row += f"{d:>26.3f}" if np.isfinite(d) else f"{'nan':>26}"
        print(row)

    # ---- focused summary on the two dominant model features ----
    print("\n" + "=" * 100)
    print("FOCUS: the two features the XGBoost model relies on most")
    print("=" * 100)
    for feat in ("total_power_1_40", "mean_band_theta"):
        print(f"\n  {feat}")
        for c in cols:
            auc, d = profiles[c][feat]
            print(f"    {c:<34} AUC={auc:.3f}  |sep|={abs(auc-0.5):.3f}  d={d:+.3f}")

    # ---- mean separability across all features ----
    print("\n" + "=" * 100)
    print("Aggregate separability across all 16 features")
    print("=" * 100)
    for c in cols:
        seps = [abs(profiles[c][n][0] - 0.5) for n in FEATURE_NAMES_16
                if np.isfinite(profiles[c][n][0])]
        ds   = [abs(profiles[c][n][1]) for n in FEATURE_NAMES_16
                if np.isfinite(profiles[c][n][1])]
        pos, neg = class_balance[c]
        print(f"  {c:<34} mean|AUC-0.5|={np.mean(seps):.3f}  "
              f"mean|d|={np.mean(ds):.3f}  (pos={pos} neg={neg})")

    print("\nInterpretation:")
    print("  The P58 variant whose separability profile (especially for")
    print("  total_power_1_40 and mean_band_theta) is CLOSER to TRAIN is the one")
    print("  whose preprocessing is consistent with the training features.")
    print("  A near-0.5 AUC / near-0 d for total_power in one variant means that")
    print("  preprocessing has collapsed the pos/neg power contrast -- explaining")
    print("  why the model floods that variant with false positives.")


if __name__ == "__main__":
    main()