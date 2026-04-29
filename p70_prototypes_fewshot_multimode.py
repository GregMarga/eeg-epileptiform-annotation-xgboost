"""
Few-shot calibration on P70 with selectable feature mode.

Three feature modes (set FEATURE_MODE below):
  - "handcrafted":  16 HC features averaged over channels
  - "labram":       LaBraM embeddings per window
  - "combined":     concat of (z-scored HC) and (z-scored LaBraM)

LaBraM files are plain .npy arrays (no labels stored). Labels for the
labeled LaBraM set are taken from the labeled HC .npz, which assumes the
two labeled files were produced from the same windows in the same order.
A length check enforces that.

Output:
  CSV named with the feature mode, e.g. P70_5shot_predictions_combined.csv
  Columns: window_index, predicted_label, center_sec, center_hmsms, probability
"""

from pathlib import Path
import csv
import numpy as np
from sklearn.preprocessing import StandardScaler


# -------------------------------------------------
# Config
# -------------------------------------------------

# Pick one: "handcrafted", "labram", "combined"
FEATURE_MODE = "labram"

# Handcrafted feature paths (per-channel, will be averaged over channels)
LABELED_HC = Path("../../../data/80hz_freq_time_features_pyprep/P70_GHB_M1679_0000078_fixed_features.npz")
SLIDING_HC = Path("../testP70/features_cache/P70_GHB_M1679_0000078_fixed_full_features.npz")

# LaBraM embedding paths
# labeled side is .npz (may carry y); sliding side is plain .npy
LABELED_LABRAM = Path("../../../data/labram_classification_1s/P70_GHB_M1679_0000078_fixed_embeddings_labeled.npz")
SLIDING_LABRAM = Path("../../../data/labram_sliding_embs_1s_75overlap/embeddings/P70_GHB_M1679_0000078_fixed_embeddings.npy")

CSV_DIR = Path("../../../data")

# Sliding window timing (must match the script that produced the sliding cache)
WIN_MS  = 500.0
HOP_MS  = 250.0

# Few-shot config
N_SHOT = 5
RANDOM_SEED = 42
THRESHOLD = 0.5
DISTANCE = "euclidean"     # "euclidean" or "cosine"


# -------------------------------------------------
# Generic .npy / .npz loader
# -------------------------------------------------

def _load_array(path: Path, key: str = "X") -> np.ndarray:
    """Load an array regardless of whether the file is .npy or .npz.

    For .npz: returns the value under `key` if present, else the first array.
    For .npy: returns the array directly.
    """
    obj = np.load(path, allow_pickle=True)
    if hasattr(obj, "files"):  # NpzFile
        if key in obj.files:
            return np.asarray(obj[key])
        return np.asarray(obj[obj.files[0]])
    return np.asarray(obj)


def _flatten_labram(X: np.ndarray) -> np.ndarray:
    """Ensure LaBraM embeddings are 2D: (N, D).

    If the array is 3D — e.g. (N, n_patches, D) or (N, n_channels, D) — mean
    pool over the middle dimension. If higher rank, mean pool over all dims
    except the first and the last.
    """
    if X.ndim == 2:
        return X
    if X.ndim == 3:
        return X.mean(axis=1)
    # Generic fallback: keep N and D, average the rest.
    return X.reshape(X.shape[0], -1, X.shape[-1]).mean(axis=1)


# -------------------------------------------------
# Feature loading
# -------------------------------------------------

def load_handcrafted_labeled(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Labeled HC: reduce per-channel features to 16 by averaging over channels."""
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # (N, 16)
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_handcrafted_sliding(path: Path) -> np.ndarray:
    """Unlabeled HC sliding: same channel-averaging as the labeled side."""
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # (N, 16)
    return X


def load_labram_labeled(path: Path) -> np.ndarray:
    """Labeled LaBraM (.npy): one embedding per window, labels come from HC."""
    X = _load_array(path).astype(np.float32)
    X = _flatten_labram(X)
    return X


def load_labram_sliding(path: Path) -> np.ndarray:
    """Sliding LaBraM (.npy): one embedding per window, no labels."""
    X = _load_array(path).astype(np.float32)
    X = _flatten_labram(X)
    return X


def load_features(mode: str):
    """Returns (X_labeled_blocks, y_labeled, X_query_blocks).

    Labels always come from the HC labeled file. For the LaBraM and combined
    modes we check that the LaBraM labeled length matches y so the assumed
    one-to-one window correspondence holds.
    """
    if mode == "handcrafted":
        X_lab, y_lab = load_handcrafted_labeled(LABELED_HC)
        X_qry = load_handcrafted_sliding(SLIDING_HC)
        return [X_lab], y_lab, [X_qry]

    if mode == "labram":
        # We still need labels — load them from HC labeled.
        _, y_lab = load_handcrafted_labeled(LABELED_HC)
        X_lab = load_labram_labeled(LABELED_LABRAM)
        if len(X_lab) != len(y_lab):
            raise ValueError(
                f"LaBraM labeled rows ({len(X_lab)}) != HC labels ({len(y_lab)}). "
                f"The two labeled files are not aligned window-to-window."
            )
        X_qry = load_labram_sliding(SLIDING_LABRAM)
        return [X_lab], y_lab, [X_qry]

    if mode == "combined":
        X_lab_hc, y_lab = load_handcrafted_labeled(LABELED_HC)
        X_lab_lb = load_labram_labeled(LABELED_LABRAM)
        if len(X_lab_lb) != len(y_lab):
            raise ValueError(
                f"LaBraM labeled rows ({len(X_lab_lb)}) != HC labels ({len(y_lab)}). "
                f"The two labeled files are not aligned window-to-window."
            )

        X_qry_hc = load_handcrafted_sliding(SLIDING_HC)
        X_qry_lb = load_labram_sliding(SLIDING_LABRAM)
        if len(X_qry_hc) != len(X_qry_lb):
            raise ValueError(
                f"Sliding HC ({len(X_qry_hc)}) and LaBraM ({len(X_qry_lb)}) "
                f"lengths differ — sliding sets are not aligned."
            )
        return [X_lab_hc, X_lab_lb], y_lab, [X_qry_hc, X_qry_lb]

    raise ValueError(f"Unknown FEATURE_MODE: {mode}")


# -------------------------------------------------
# Few-shot prototypical classifier
# -------------------------------------------------

def sample_support(y: np.ndarray, n_shot: int, rng) -> np.ndarray:
    """Returns indices into the labeled set for the support windows.
    Same indices are used across modalities so the support set stays aligned."""
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < n_shot:
        raise ValueError(f"Not enough positives: {len(pos_idx)} < {n_shot}")
    if len(neg_idx) < n_shot:
        raise ValueError(f"Not enough negatives: {len(neg_idx)} < {n_shot}")

    chosen_pos = rng.choice(pos_idx, size=n_shot, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_shot, replace=False)
    return np.concatenate([chosen_pos, chosen_neg])


def normalize_support_query(
    X_support_blocks: list[np.ndarray],
    X_query_blocks: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Per-modality StandardScaler fit on support, applied to both."""
    sup_parts = []
    qry_parts = []
    for X_sup_block, X_qry_block in zip(X_support_blocks, X_query_blocks):
        scaler = StandardScaler()
        sup_parts.append(scaler.fit_transform(X_sup_block))
        qry_parts.append(scaler.transform(X_qry_block))

    return np.concatenate(sup_parts, axis=1), np.concatenate(qry_parts, axis=1)


def prototypical_predict_proba(
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    distance: str = "euclidean",
) -> np.ndarray:
    proto_pos = X_support[y_support == 1].mean(axis=0)
    proto_neg = X_support[y_support == 0].mean(axis=0)

    if distance == "euclidean":
        d_pos = np.linalg.norm(X_query - proto_pos, axis=1)
        d_neg = np.linalg.norm(X_query - proto_neg, axis=1)
    elif distance == "cosine":
        def cosine_dist(X, proto):
            num = X @ proto
            den = np.linalg.norm(X, axis=1) * np.linalg.norm(proto) + 1e-8
            return 1 - num / den
        d_pos = cosine_dist(X_query, proto_pos)
        d_neg = cosine_dist(X_query, proto_neg)
    else:
        raise ValueError(f"Unknown distance: {distance}")

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
    print(f"Feature mode: {FEATURE_MODE}")

    # 1) Load all required modalities
    X_labeled_blocks, y_labeled, X_query_blocks = load_features(FEATURE_MODE)
    print(f"  Labeled windows: {len(y_labeled)} "
          f"(pos={int(y_labeled.sum())}, neg={int((y_labeled == 0).sum())})")
    for i, blk in enumerate(X_labeled_blocks):
        print(f"  Labeled block {i}: shape={blk.shape}")
    for i, blk in enumerate(X_query_blocks):
        print(f"  Query   block {i}: shape={blk.shape}")

    # 2) Pick the same support indices across modalities
    rng = np.random.default_rng(RANDOM_SEED)
    sup_idx = sample_support(y_labeled, N_SHOT, rng)
    y_sup = y_labeled[sup_idx]
    X_sup_blocks = [blk[sup_idx] for blk in X_labeled_blocks]
    print(f"  Support set: {len(y_sup)} windows "
          f"(pos={int(y_sup.sum())}, neg={int((y_sup == 0).sum())})")

    # 3) Normalize per modality, then concat
    X_sup, X_qry = normalize_support_query(X_sup_blocks, X_query_blocks)
    print(f"  Support features: {X_sup.shape}")
    print(f"  Query   features: {X_qry.shape}")

    # 4) Prototypical predict
    print(f"\nClassifying with prototypical network ({DISTANCE})...")
    y_proba = prototypical_predict_proba(X_sup, y_sup, X_qry, distance=DISTANCE)
    y_pred = (y_proba >= THRESHOLD).astype(np.uint8)

    print(f"  Probability stats: min={y_proba.min():.4f} "
          f"median={np.median(y_proba):.4f} max={y_proba.max():.4f}")
    print(f"  Predicted positives at thr={THRESHOLD}: "
          f"{int(y_pred.sum())} / {len(y_pred)} "
          f"({100.0 * y_pred.mean():.2f}%)")

    # 5) Compute window centers
    centers_sec = compute_window_centers_sec(len(y_proba), WIN_MS, HOP_MS)

    # 6) Save CSV with mode in the filename
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"P70_{N_SHOT}shot_predictions_{FEATURE_MODE}.csv"
    print(f"\nSaving predictions to: {csv_path}")

    with csv_path.open("w", newline="") as f:
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