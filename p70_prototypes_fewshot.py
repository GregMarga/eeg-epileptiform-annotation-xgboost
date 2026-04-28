"""
Few-shot calibration on P70 using its OWN labeled windows.

Setup:
  - Support set: 5 positives + 5 negatives randomly sampled (seed=42)
                 from the P70 labeled features file
  - Query set:   all ~239k sliding-window features of P70
  - Features:    all 16 handcrafted features (averaged across 19 channels)
  - Classifier:  prototypical network (Euclidean distance to class centroids)
  - Normalization: StandardScaler fit on the SUPPORT set only,
                   then applied to support and query consistently

Output:
  CSV with one row per query window:
    window_index, predicted_label, center_sec, probability
"""

from pathlib import Path
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# Config
# -------------------------------------------------

LABELED_P70 = Path("../../../data/80hz_freq_time_features_pyprep/P70_GHB_M1679_0000078_fixed_features.npz")
SLIDING_P70 = Path("../testP70/features_cache/P70_GHB_M1679_0000078_fixed_full_features.npz")

CSV_OUTPUT = Path("../../../data/P70_5shot_predictions.csv")

# Sliding window timing (must match the script that produced the sliding cache)
WIN_MS  = 500.0
HOP_MS  = 250.0

# Few-shot config
N_SHOT = 5
RANDOM_SEED = 42
THRESHOLD = 0.5            # for binarizing the probability into a predicted label
DISTANCE = "euclidean"     # "euclidean" or "cosine"


# -------------------------------------------------
# Feature loading
# -------------------------------------------------

def load_labeled_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load labeled features file. Reduce per-channel features to 16 by
    averaging across the 19 channels (same as the few-shot script)."""
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)              # (N, 19*16) flattened
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # (N, 16)
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_sliding_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load unlabeled sliding features file.
    Returns (X_sliding, ch_names)."""
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # (N, 16)
    return X, z["ch_names"]


# -------------------------------------------------
# Few-shot prototypical classifier
# -------------------------------------------------

def sample_support(X: np.ndarray, y: np.ndarray, n_shot: int, rng) -> tuple[np.ndarray, np.ndarray]:
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < n_shot:
        raise ValueError(f"Not enough positives in P70 labeled: {len(pos_idx)} < {n_shot}")
    if len(neg_idx) < n_shot:
        raise ValueError(f"Not enough negatives in P70 labeled: {len(neg_idx)} < {n_shot}")

    chosen_pos = rng.choice(pos_idx, size=n_shot, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_shot, replace=False)
    support_idx = np.concatenate([chosen_pos, chosen_neg])

    return X[support_idx], y[support_idx]


def prototypical_predict_proba(
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    distance: str = "euclidean",
) -> np.ndarray:
    """
    Fit StandardScaler on the support set ONLY, transform both support
    and query with the SAME stats. This ensures both sets live in the
    same normalized feature space.
    """
    scaler = StandardScaler()
    X_support_s = scaler.fit_transform(X_support)
    X_query_s = scaler.transform(X_query)

    proto_pos = X_support_s[y_support == 1].mean(axis=0)
    proto_neg = X_support_s[y_support == 0].mean(axis=0)

    if distance == "euclidean":
        d_pos = np.linalg.norm(X_query_s - proto_pos, axis=1)
        d_neg = np.linalg.norm(X_query_s - proto_neg, axis=1)
    elif distance == "cosine":
        def cosine_dist(X, proto):
            num = X @ proto
            den = np.linalg.norm(X, axis=1) * np.linalg.norm(proto) + 1e-8
            return 1 - num / den
        d_pos = cosine_dist(X_query_s, proto_pos)
        d_neg = cosine_dist(X_query_s, proto_neg)
    else:
        raise ValueError(f"Unknown distance: {distance}")

    # Softmax over (-d_pos, -d_neg) → P(y=1)
    logit_pos = -d_pos
    logit_neg = -d_neg
    m = np.maximum(logit_pos, logit_neg)
    exp_pos = np.exp(logit_pos - m)
    exp_neg = np.exp(logit_neg - m)
    return exp_pos / (exp_pos + exp_neg)


# -------------------------------------------------
# Window timing
# -------------------------------------------------

def compute_window_centers_sec(n_windows: int, win_ms: float, hop_ms: float) -> np.ndarray:
    """Window i covers [i*hop, i*hop + win] in seconds.
    Center = i*hop + win/2."""
    hop_s = hop_ms / 1000.0
    win_s = win_ms / 1000.0
    return np.arange(n_windows, dtype=float) * hop_s + 0.5 * win_s


def sec_to_hmsms(sec: float) -> str:
    ms_total = int(round(sec * 1000))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    # 1) Load labeled P70 (support source)
    print(f"Loading labeled P70 from: {LABELED_P70}")
    X_labeled, y_labeled = load_labeled_features(LABELED_P70)
    print(f"  Labeled windows: {len(y_labeled)} "
          f"(pos={int(y_labeled.sum())}, neg={int((y_labeled == 0).sum())})")

    # 2) Sample 5+5 support
    rng = np.random.default_rng(RANDOM_SEED)
    X_sup, y_sup = sample_support(X_labeled, y_labeled, N_SHOT, rng)
    print(f"  Support set: {len(y_sup)} windows "
          f"(pos={int(y_sup.sum())}, neg={int((y_sup == 0).sum())})")

    # 3) Load sliding query set
    print(f"\nLoading sliding P70 from: {SLIDING_P70}")
    X_query, ch_names = load_sliding_features(SLIDING_P70)
    print(f"  Query windows: {len(X_query)}")

    # 4) Prototypical predict
    print(f"\nClassifying with prototypical network ({DISTANCE}, normalized)...")
    y_proba = prototypical_predict_proba(X_sup, y_sup, X_query, distance=DISTANCE)
    y_pred = (y_proba >= THRESHOLD).astype(np.uint8)

    print(f"  Probability stats: min={y_proba.min():.4f} "
          f"median={np.median(y_proba):.4f} max={y_proba.max():.4f}")
    print(f"  Predicted positives at thr={THRESHOLD}: "
          f"{int(y_pred.sum())} / {len(y_pred)} "
          f"({100.0 * y_pred.mean():.2f}%)")

    # 5) Compute window centers
    centers_sec = compute_window_centers_sec(len(y_proba), WIN_MS, HOP_MS)

    # 6) Save CSV
    CSV_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving predictions to: {CSV_OUTPUT}")

    with CSV_OUTPUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "window_index",
            "predicted_label",
            "center_sec",
            "center_hmsms",
            "probability",
        ])
        for i in range(len(y_proba)):
            w.writerow([
                i,
                int(y_pred[i]),
                f"{centers_sec[i]:.6f}",
                sec_to_hmsms(centers_sec[i]),
                f"{float(y_proba[i]):.8f}",
            ])

    print(f"Done. {len(y_proba)} rows written.")


if __name__ == "__main__":
    main()