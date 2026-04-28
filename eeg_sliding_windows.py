from pathlib import Path
import mne
import numpy as np
from mne.channels import make_standard_montage
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from tqdm import tqdm


# -----------------------------
# Bad channels for this specific recording (from PyPREP report)
# Only deviation / correlation / ransac are interpolated; hf_noise is ignored.
# Report for P70_GHB_M1679_0000078_fixed.edf:
#   hf_noise    : ['Fp2']
#   correlation : ['Fp2', 'F8']
# Channels to interpolate: Fp2 (correlation) + F8 (correlation) = ['Fp2', 'F8']
# -----------------------------
EDF_NAME = "P70_GHB_M1679_0000078_fixed"
EDF_PATH = f"../../../data/{EDF_NAME}.edf"
BAD_CHANNELS_TO_INTERPOLATE = ["Fp2", "F8"]

TARGET_SFREQ = 80.0   # downsample target — matches the trained model's pipeline


# -----------------------------
# Sliding windows
# -----------------------------

def create_sliding_epochs(
        data,
        sfreq,
        step_size=250,  # in ms
        window=500,  # in ms
):
    sample_window = int(round(window * sfreq / 1000))
    last_start = data.shape[-1] - sample_window
    step = int(round(step_size * sfreq / 1000))

    epochs = []
    for start in range(0, last_start + 1, step):
        epoch = data[:, start:start + sample_window]
        epochs.append(epoch)
    return np.array(epochs)


def create_sliding_windows_from_eeg(
        l_freq: float = 0.5,
        h_freq: float = 40.0,
):
    raw = mne.io.read_raw_edf(EDF_PATH, preload=True)

    # Clean channel names
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())

    # Standard 10-20 montage (needed for spatial interpolation)
    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    # Bandpass filter
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose="ERROR",
    )

    # Downsample to target sfreq (80 Hz) for speed and to match the
    # training pipeline. Done AFTER the lowpass filter so we are above
    # the Nyquist limit for the new rate.
    raw.resample(TARGET_SFREQ, verbose="ERROR")

    # Hardcoded bad channels from PyPREP report — interpolate only those
    # in deviation / correlation / ransac (hf_noise alone is skipped).
    bad_chs = [ch for ch in BAD_CHANNELS_TO_INTERPOLATE if ch in raw.ch_names]
    if bad_chs:
        raw.info["bads"] = bad_chs
        raw.interpolate_bads(reset_bads=True)
        print(f"Interpolated bad channels: {bad_chs}")
    else:
        print("No bad channels to interpolate.")

    # Average reference AFTER interpolation, so the bad channels do not
    # contaminate the reference signal.
    raw.set_eeg_reference("average", verbose="ERROR")

    data = raw.get_data()
    sfreq = float(raw.info["sfreq"])
    ch_names = np.array(raw.ch_names, dtype=object)
    print(f"Data shape: {data.shape}, sfreq={sfreq}")

    epochs = create_sliding_epochs(data=data, sfreq=sfreq)
    return (
        epochs.astype(np.float32),
        sfreq,
        ch_names,
        np.array(bad_chs, dtype=object),
    )


# -----------------------------
# Time-domain feature primitives
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
# Frequency-domain feature primitives
# -----------------------------

def compute_welch_psd_1d(
        x: np.ndarray,
        fs: float,
        nperseg: int | None = None,
        noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)

    if nperseg is None:
        nperseg = min(256, x.shape[-1])
    if noverlap is None:
        noverlap = nperseg // 2

    f, psd = welch(
        x, fs=fs,
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
    return float(np.trapezoid(psd[m], f[m]))


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


def extract_features_for_one_window(window_2d: np.ndarray, fs: float) -> np.ndarray:
    """
    window_2d: (n_channels, n_samples)
    returns: (n_channels * (6 + 10)) feature vector
    """
    n_ch = window_2d.shape[0]
    feats = []

    for ch in range(n_ch):
        x = window_2d[ch].astype(float)
        x = x - np.mean(x)

        # time-domain (6)
        zc = zero_crossings(x)
        mx, mn = count_local_extrema(x)
        rms = rms_amplitude(x)
        sk = float(skew(x, bias=False))
        ku = float(kurtosis(x, fisher=True, bias=False))

        # freq-domain (10)
        ffeats = freq_features_1d(x, fs)

        feats.extend([zc, mx, mn, rms, sk, ku, *ffeats])

    return np.asarray(feats, dtype=np.float32)


def process_np_windows_one_patient(windows, sfreq: float, ch_names, out_dir: Path):
    n_windows = windows.shape[0]
    fnames = feature_names(ch_names)
    n_feat = len(fnames)

    X = np.empty((n_windows, n_feat), dtype=np.float32)

    for i in tqdm(range(n_windows), desc=f"Extracting features ({EDF_NAME})"):
        X[i] = extract_features_for_one_window(windows[i], sfreq)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EDF_NAME}_full_features.npz"

    np.savez_compressed(
        out_path,
        X=X,
        sfreq=float(sfreq),
        ch_names=np.array(ch_names, dtype=object),
        feature_names=np.array(fnames, dtype=object),
        source_edf=str(EDF_NAME),
    )

    print(f"Saved: {out_path} | X={X.shape}")


def main():
    epochs, sfreq, ch_names, bad_chs = create_sliding_windows_from_eeg(
        l_freq=0.5, h_freq=40.0,
    )

    windows_out = Path("../testP70/windows_cache")
    windows_out.mkdir(parents=True, exist_ok=True)

    windows_path = windows_out / f"{EDF_NAME}_full_windows.npz"

    np.savez_compressed(
        windows_path,
        windows=epochs.astype(np.float32),
        sfreq=float(sfreq),
        ch_names=np.array(ch_names, dtype=object),
        bad_chs=np.array(bad_chs, dtype=object),
        source_edf=EDF_NAME,
    )

    print(f"Saved windows: {windows_path} | windows={epochs.shape}")
    print(f"Created sliding windows: {len(epochs)}")

    # process_np_windows_one_patient(
    #     windows=epochs,
    #     sfreq=sfreq,
    #     ch_names=ch_names,
    #     out_dir=Path("../testP70/features_cache"),
    # )


if __name__ == "__main__":
    main()