import mne
import numpy as np
from mne.channels import make_standard_montage
from pathlib import Path


def create_sliding_windows(
        data,
        sfreq,
        window_sec=1.0,
        stride_sec=0.25,
):
    """Create continuous sliding windows of fixed length with given stride.

    Parameters
    ----------
    data : np.ndarray
        Shape (n_channels, n_samples).
    sfreq : float
        Sampling frequency in Hz.
    window_sec : float
        Window length in seconds.
    stride_sec : float
        Step between consecutive windows in seconds.
        E.g. stride=0.25 with window=1.0 -> 75% overlap.

    Returns
    -------
    epochs : np.ndarray
        Shape (n_windows, n_channels, n_samples_per_window).
    window_onsets : np.ndarray
        Sample index of each window's start.
    """
    window_samples = int(round(window_sec * sfreq))
    stride_samples = int(round(stride_sec * sfreq))

    n_samples = data.shape[1]

    # Vectorized window indexing
    starts = np.arange(0, n_samples - window_samples + 1, stride_samples)
    epochs = np.stack([data[:, s:s + window_samples] for s in starts], axis=0)

    return epochs, starts


def preprocess_for_labram(
        raw,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
):
    """LaBraM-matching preprocessing pipeline (no bad channel handling).

    Order:
      1. Pick EEG, set montage
      2. Bandpass (0.1 - 75 Hz) matching pre-training
      3. Notch at line frequency
      4. Average reference
      5. Resample to 200 Hz
    """
    raw.pick("eeg")
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())

    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    # Bandpass matching LaBraM pre-training (0.1 - 75 Hz)
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose="ERROR",
    )

    # Notch filter for line noise
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

    # Average reference
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")

    # Resample to 200 Hz (LaBraM target)
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    # Convert to μV
    data_uv = raw.get_data() * 1e6

    return raw, data_uv


def batch_create_sliding_windows_from_eeg(
        edf_path: str,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
        window_sec: float = 1.0,
        stride_sec: float = 0.25,
):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    raw, data_normalized = preprocess_for_labram(
        raw,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=notch_freq,
        target_sfreq=target_sfreq,
    )

    sfreq = float(raw.info["sfreq"])
    ch_names = np.array(raw.ch_names, dtype=object)

    print(f"  Data shape (channels, samples): {data_normalized.shape}")
    print(f"  Sampling frequency: {sfreq}")
    print(f"  Window: {window_sec}s ({int(window_sec*sfreq)} samples)")
    print(f"  Stride: {stride_sec}s ({int(stride_sec*sfreq)} samples) "
          f"-> {int((1 - stride_sec/window_sec)*100)}% overlap")

    epochs, window_onsets = create_sliding_windows(
        data=data_normalized,
        sfreq=sfreq,
        window_sec=window_sec,
        stride_sec=stride_sec,
    )

    return (
        epochs.astype(np.float32),
        ch_names,
        window_onsets,
        sfreq,
    )


def main():
    data_dir = Path("../../../data")
    out_dir = data_dir / "labram_sliding_windows_1s_75overlap"
    out_dir.mkdir(parents=True, exist_ok=True)

    target_stem = "P70_GHB_M1679_0000078_fixed"
    edf_path = data_dir / f"{target_stem}.edf"

    if not edf_path.exists():
        raise FileNotFoundError(f"Could not find {edf_path}")

    print(f"Processing: {edf_path.name}")

    (
        windows,
        ch_names,
        window_onsets,
        sfreq,
    ) = batch_create_sliding_windows_from_eeg(
        edf_path=str(edf_path),
        l_freq=0.1,
        h_freq=75.0,
        notch_freq=50.0,
        target_sfreq=200.0,
        window_sec=1.0,
        stride_sec=0.25,
    )

    out_path = out_dir / f"{edf_path.stem}_windows.npz"
    np.savez_compressed(
        out_path,
        windows=windows,
        ch_names=ch_names,
        edf_name=edf_path.name,
        window_onsets=window_onsets.astype(np.int64),
        sfreq=sfreq,
        window_sec=1.0,
        stride_sec=0.25,
    )

    print(f"  Saved: {out_path}")
    print(f"  windows shape: {windows.shape}")
    print(f"  total windows: {len(windows)}")

    print("Done.")


if __name__ == "__main__":
    main()