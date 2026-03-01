import mne
import numpy as np
from mne.channels import make_standard_montage
from pathlib import Path
from tqdm import tqdm

def create_sliding_epochs(
        data,
        sfreq,
        step_size=750,  # 75% of 1s -> 25% overlap
        window=1000,    # 1 second
):
    sample_window = int(round(window * sfreq / 1000))
    last_start = data.shape[-1] - sample_window
    step = int(round(step_size * sfreq / 1000))

    epochs = []
    for start in tqdm(range(0, last_start + 1, step),
                      desc="Windows",
                      leave=False):
        epoch = data[:, start:start + sample_window]
        epochs.append(epoch)

    return np.array(epochs)


def create_sliding_windows_from_eeg(
        edf_path: str,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
):
    data_dir = Path("../../../data")
    out_dir = data_dir / "labram_windows_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    # 1) Keep only EEG channels (remove irrelevant channels)
    raw.pick("eeg")

    # Clean channel names
    raw.rename_channels(lambda ch: ch.replace('EEG ', '').strip())

    # Set standard 10-20 montage
    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    # 2) Bandpass 0.1–75 Hz
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        fir_design='firwin',
        phase='zero',
        verbose="ERROR",
    )

    # 3) Notch 50 Hz
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

    # 4) Resample to 200 Hz
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    # 5) Convert from Volts to microvolts (μV)
    data_uv = raw.get_data() * 1e6

    sfreq = float(raw.info['sfreq'])  # should now be 200 Hz
    ch_names = np.array(raw.ch_names, dtype=object)

    print("Data shape (channels, samples):", data_uv.shape)
    print("Sampling frequency:", sfreq)

    epochs = create_sliding_epochs(data=data_uv, sfreq=sfreq)

    return (
        epochs.astype(np.float32),
        ch_names,
    )
