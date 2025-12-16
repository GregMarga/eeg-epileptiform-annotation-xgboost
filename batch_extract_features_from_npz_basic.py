from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis


# -----------------------------
# Feature primitives (1D)
# -----------------------------

def zero_crossings(x: np.ndarray) -> int:
    return int(np.sum((x[:-1] * x[1:]) < 0))

def count_local_extrema(x: np.ndarray):
    dx = np.diff(x)
    maxima = np.sum((dx[:-1] > 0) & (dx[1:] < 0))
    minima = np.sum((dx[:-1] < 0) & (dx[1:] > 0))
    return int(maxima), int(minima)

def rms_amplitude(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x)))


# -----------------------------
# Per-window feature extraction
# -----------------------------

def extract_features_for_one_window(window_2d: np.ndarray, fs: float) -> np.ndarray:
    """
    window_2d: (n_channels, n_samples)
    returns: (n_channels * 6,) feature vector
    """
    n_ch = window_2d.shape[0]
    feats = []

    for ch in range(n_ch):
        x = window_2d[ch].astype(float)
        x = x - np.mean(x)  #center around 0

        zc = zero_crossings(x)
        mx, mn = count_local_extrema(x)
        rms = rms_amplitude(x)
        sk = float(skew(x, bias=False))
        ku = float(kurtosis(x, fisher=True, bias=False))

        feats.extend([zc, mx, mn, rms, sk, ku])

    return np.asarray(feats, dtype=np.float32)


def feature_names(ch_names: np.ndarray) -> list[str]:
    per_ch = [
        "zero_cross",
        "maxima",
        "minima",
        "rms",
        "skew",
        "kurt_excess",
    ]
    names = []
    for ch in ch_names:
        for f in per_ch:
            names.append(f"{ch}_{f}")
    return names


# -----------------------------
# Batch over NPZ files
# -----------------------------

def process_npz_file(npz_path: Path, out_dir: Path, batch_windows: int = 512):
    z = np.load(npz_path, allow_pickle=True)

    windows = z["windows"]              # (n_windows, n_channels, win_len)
    labels = z["labels"]
    sfreq = float(z["sfreq"])
    ch_names = z["ch_names"]
    edf_name = z.get("edf_name", npz_path.name)

    n_windows, n_ch, _ = windows.shape
    fnames = feature_names(ch_names)
    n_feat = len(fnames)

    X = np.empty((n_windows, n_feat), dtype=np.float32)

    for start in range(0, n_windows, batch_windows):
        end = min(start + batch_windows, n_windows)
        for i in range(start, end):
            X[i] = extract_features_for_one_window(windows[i], sfreq)
        print(f"  {npz_path.name}: features [{end}/{n_windows}]")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / npz_path.name.replace("_windows.npz", "_features.npz")

    np.savez_compressed(
        out_path,
        X=X,
        y=labels.astype(np.uint8),
        sfreq=sfreq,
        ch_names=np.array(ch_names, dtype=object),
        feature_names=np.array(fnames, dtype=object),
        source_npz=npz_path.name,
        source_edf=str(edf_name),
    )

    print(
        f"Saved: {out_path} | X={X.shape} positives={int(labels.sum())}/{len(labels)}"
    )


def main():
    in_dir = Path("../../data/windows_cache")
    out_dir = Path("../../data/features_cache_basic")

    npz_files = sorted(in_dir.glob("*_windows.npz"))
    if not npz_files:
        print(f"No *_windows.npz files found in {in_dir}")
        return

    print(f"Found {len(npz_files)} window files.")

    for i, p in enumerate(npz_files, start=1):
        print(f"\n[{i}/{len(npz_files)}] Processing {p.name}")
        process_npz_file(p, out_dir, batch_windows=512)

    print("\nDone.")


if __name__ == "__main__":
    main()
