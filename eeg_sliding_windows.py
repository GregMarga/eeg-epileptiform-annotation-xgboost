from pathlib import Path
import mne
import numpy as np
from mne.channels import make_standard_montage


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
    sample_window = int(window * sfreq / 1000)
    last_start = data.shape[-1] - sample_window
    step = int(step_size * sfreq / 1000)

    epochs=[]

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

    epochs=create_sliding_epochs(data=data,sfreq=sfreq)
    return (
        epochs.astype(np.float32),
        sfreq,
        ch_names,
        np.array(bad_chs, dtype=object),
    )

create_sliding_windows_from_eeg(
    l_freq=0.5, h_freq=40.0, high_factor=8.0, low_factor=10.0, )
