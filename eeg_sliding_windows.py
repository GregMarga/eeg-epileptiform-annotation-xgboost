from pathlib import Path
import mne
import numpy as np
from mne.channels import make_standard_montage
from scipy.stats import skew, kurtosis


def detect_bad_channels_by_std(
        raw: mne.io.BaseRaw,
        high_factor: float = 8.0,
        low_factor: float = 10.0,
):
    # find outlier (bad) channels based on std

    data = raw.get_data()  # shape: (n_channels, n_samples)
    stds = data.std(axis=1)  # std per channel
    median_std = np.median(stds)

    high_threshold = median_std * high_factor
    low_threshold = median_std / low_factor

    bad_high = np.where(stds > high_threshold)[0]  # too noisy eg electrode pop artifacts
    bad_low = np.where(stds < low_threshold)[0]  # dead channel

    bad_idx = np.unique(np.concatenate([bad_high, bad_low]))
    bad_channels = [raw.ch_names[i] for i in bad_idx]

    return bad_channels, stds


def mark_and_interpolate_bad_channels(
        raw: mne.io.BaseRaw,
        high_factor: float = 8.0,
        low_factor: float = 10.0,
        plot: bool = False,
):
    bad_channels, stds = detect_bad_channels_by_std(
        raw,
        high_factor=high_factor,
        low_factor=low_factor,
    )

    # print("STD per channel:")
    # for name, s in zip(raw.ch_names, stds):
    #     print(f"{name}: {s:.3e}")
    # print("Detected bad channels:", bad_channels)

    # Mark as bad
    raw.info['bads'] = bad_channels

    # Spatial interpolation
    if len(bad_channels) > 0:
        raw.interpolate_bads(reset_bads=True)
        # print("Interpolated bad channels.")
    # else:
    #     print("No bad channels detected, nothing to interpolate.")

    if plot:
        raw.plot(scalings='auto', block=True)

    return bad_channels


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
        high_factor: float = 8.0,
        low_factor: float = 10.0,
):
    data_dir = Path("../../data")
    out_dir = data_dir / "windows_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(data_dir.glob("*.edf"))

    raw = mne.io.read_raw_edf(
        "../../../data/P20_GHB_00015_0000348.edf",
        preload=True
    )

    # clean channel names
    raw.rename_channels(lambda ch: ch.replace('EEG ', '').strip())

    # montage
    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    # average reference
    raw.set_eeg_reference('average', verbose="ERROR")

    # filter
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        fir_design='firwin',
        phase='zero',
        verbose="ERROR",
    )

    # bad channels detection + interpolation (your existing function)
    bad_chs = mark_and_interpolate_bad_channels(
        raw,
        high_factor=high_factor,
        low_factor=low_factor,
        plot=False,
    )

    data = raw.get_data()
    sfreq = float(raw.info['sfreq'])
    ch_names = np.array(raw.ch_names, dtype=object)
    print(data.shape)

    epochs = create_sliding_epochs(data=data, sfreq=sfreq)
    return (
        epochs.astype(np.float32),
        sfreq,
        ch_names,
        np.array(bad_chs, dtype=object),
    )


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


def extract_features_for_one_window(window_2d: np.ndarray, fs: float) -> np.ndarray:
    """
    window_2d: (n_channels, n_samples)
    returns: (n_channels * 6) feature vector
    """
    n_ch = window_2d.shape[0]
    feats = []

    for ch in range(n_ch):
        x = window_2d[ch].astype(float)
        x = x - np.mean(x)  # center around 0

        zc = zero_crossings(x)
        mx, mn = count_local_extrema(x)
        rms = rms_amplitude(x)
        sk = float(skew(x, bias=False))
        ku = float(kurtosis(x, fisher=True, bias=False))

        feats.extend([zc, mx, mn, rms, sk, ku])

    return np.asarray(feats, dtype=np.float32)


def process_np_windows_one_patient(windows, sfreq: float, ch_names, out_dir: Path):
    edf_name = "P20_GHB_00015_0000348"

    n_windows, n_ch, _ = windows.shape
    fnames = feature_names(ch_names)
    n_feat = len(fnames)

    X = np.empty((n_windows, n_feat), dtype=np.float32)

    for i in range(n_windows):
        X[i] = extract_features_for_one_window(windows[i], sfreq)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{edf_name}_full_features.npz"

    np.savez_compressed(
        out_path,
        X=X,
        sfreq=float(sfreq),
        ch_names=np.array(ch_names, dtype=object),
        feature_names=np.array(fnames, dtype=object),
        source_edf=str(edf_name),
    )

    print(f"Saved: {out_path} | X={X.shape}")


def main():
    out_dir = Path("../../test/features_cache_basic")

    epochs, sfreq, ch_names, bad_chs = create_sliding_windows_from_eeg(
        l_freq=0.5, h_freq=40.0, high_factor=8.0, low_factor=10.0,
    )

    print(f"Created sliding windows: {len(epochs)}")

    process_np_windows_one_patient(
        windows=epochs,
        sfreq=sfreq,
        ch_names=ch_names,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
