from pathlib import Path
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch


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
# Feature primitives (1D) - frequency domain (NO HF)
# -----------------------------

def compute_welch_psd_1d(
    x: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      f:   (n_freqs,)
      psd: (n_freqs,)  power/Hz
    """
    x = np.asarray(x, dtype=np.float64)

    if nperseg is None:
        nperseg = min(256, x.shape[-1])
    if noverlap is None:
        noverlap = nperseg // 2

    f, psd = welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend="constant",
        scaling="density",
    )
    return f, psd


def _band_mask(f: np.ndarray, fmin: float, fmax: float) -> np.ndarray:
    if fmin >= fmax:
        raise ValueError(f"Invalid band [{fmin}, {fmax}]")
    return (f >= fmin) & (f <= fmax)


def bandpower_trapz(f: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    m = _band_mask(f, fmin, fmax)
    if not np.any(m):
        return 0.0

    f_band = f[m]
    psd_band = psd[m]

    if len(f_band) == 1:
        # approximate bandpower from one bin
        df = f[1] - f[0] if len(f) > 1 else 0.0
        return float(psd_band[0] * df)

    return float(np.trapezoid(psd_band, f_band))


def mean_psd_in_band(f: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    m = _band_mask(f, fmin, fmax)
    if not np.any(m):
        return 0.0
    return float(np.mean(psd[m]))


def peak_frequency_in_band(f: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    m = _band_mask(f, fmin, fmax)
    if not np.any(m):
        return float("nan")
    idx = int(np.argmax(psd[m]))
    return float(f[m][idx])


def freq_features_1d(
    x: np.ndarray,
    fs: float,
    *,
    total_range: tuple[float, float] = (1.0, 40.0),
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> list[float]:
    """
    Frequency features for ONE channel x (NO HF).

    Output order (fixed):
      total_power_1_40
      peak_freq_1_40
      mean_band_delta, mean_band_theta, mean_band_alpha, mean_band_beta
      norm_band_delta, norm_band_theta, norm_band_alpha, norm_band_beta
    """
    bands = {
        "delta": (1.0, 3.0),
        "theta": (4.0, 8.0),
        "alpha": (9.0, 13.0),
        "beta":  (14.0, 20.0),
    }

    f, psd = compute_welch_psd_1d(x, fs=fs, nperseg=nperseg, noverlap=noverlap)

    tr0, tr1 = total_range
    total_p = bandpower_trapz(f, psd, tr0, tr1)
    peak_f = peak_frequency_in_band(f, psd, tr0, tr1)

    mean_feats = []
    norm_feats = []
    eps = 1e-12

    for name in ("delta", "theta", "alpha", "beta"):
        fmin, fmax = bands[name]
        bp = bandpower_trapz(f, psd, fmin, fmax)
        mp = mean_psd_in_band(f, psd, fmin, fmax)
        mean_feats.append(mp)
        norm_feats.append(bp / (total_p + eps))
    return [total_p, peak_f, *mean_feats, *norm_feats]


# -----------------------------
# Per-window feature extraction
# -----------------------------

def extract_features_for_one_window(window_2d: np.ndarray, fs: float) -> np.ndarray:
    """
    window_2d: (n_channels, n_samples)
    returns: (n_channels * (6 + 10)) feature vector
      time: 6
      freq: 10  (2 + 4 mean + 4 norm)
    """
    n_ch = window_2d.shape[0]
    feats = []

    for ch in range(n_ch):
        x = window_2d[ch].astype(float)
        x = x - np.mean(x)  # center around 0

        # time-domain
        zc = zero_crossings(x)
        mx, mn = count_local_extrema(x)
        rms = rms_amplitude(x)
        sk = float(skew(x, bias=False))
        ku = float(kurtosis(x, fisher=True, bias=False))

        # freq-domain (NO HF)
        ffeats = freq_features_1d(x, fs)

        feats.extend([zc, mx, mn, rms, sk, ku, *ffeats])

    return np.asarray(feats, dtype=np.float32)


def feature_names(ch_names: np.ndarray) -> list[str]:
    per_ch_time = [
        "zero_cross",
        "maxima",
        "minima",
        "rms",
        "skew",
        "kurt_excess",
    ]

    per_ch_freq = [
        "total_power_1_40",
        "peak_freq_1_40",
        "mean_band_delta",
        "mean_band_theta",
        "mean_band_alpha",
        "mean_band_beta",
        "norm_band_delta",
        "norm_band_theta",
        "norm_band_alpha",
        "norm_band_beta",
    ]

    names = []
    for ch in ch_names:
        for f in per_ch_time:
            names.append(f"{ch}_{f}")
        for f in per_ch_freq:
            names.append(f"{ch}_{f}")
    return names


# -----------------------------
# Batch over NPZ files
# -----------------------------

def process_npz_file(npz_path: Path, out_dir: Path, batch_windows: int = 512):
    z = np.load(npz_path, allow_pickle=True)

    windows = z["windows"]              # (n_windows, n_channels, win_len)
    labels = z["labels"]
    center_sec = z["center_sec"]
    center_hmsms = z["center_hmsms"]
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
        center_sec=np.array(center_sec, dtype=np.float64),
        center_hmsms=np.array(center_hmsms, dtype=object),
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
    in_dir = Path("../../../Data/windows_cache")
    out_dir = Path("../data/freq_time_features_cache_basic")

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