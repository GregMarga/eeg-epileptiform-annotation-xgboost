import mne
import numpy as np
from mne.channels import make_standard_montage
from pathlib import Path


# ---------------------------
# 1. Detect bad channels
# ---------------------------


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

# Τυπικό longitudinal (double banana) montage
LONGITUDINAL_PAIRS = [
    # Lateral left
    ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
    # Lateral right
    ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
    # Parasagittal left
    ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
    # Parasagittal right
    ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    # Midline
    ("Fz", "Cz"), ("Cz", "Pz"),
]


def apply_longitudinal_montage(
        raw: mne.io.BaseRaw,
        pairs: list[tuple[str, str]] = LONGITUDINAL_PAIRS,
) -> tuple[np.ndarray, list[str]]:
    """
    Υπολογίζει bipolar channels από τα ζεύγη.
    Επιστρέφει:
        bipolar_data : (n_pairs, n_samples)
        bipolar_names: ["Fp1-F7", ...]
    """
    data = raw.get_data()
    ch_names_lower = {ch.lower(): i for i, ch in enumerate(raw.ch_names)}

    bipolar_data = []
    bipolar_names = []

    for anode, cathode in pairs:
        a_idx = ch_names_lower.get(anode.lower())
        c_idx = ch_names_lower.get(cathode.lower())

        if a_idx is None or c_idx is None:
            print(f"  Skipping {anode}-{cathode}: channel not found")
            continue

        bipolar_data.append(data[a_idx] - data[c_idx])
        bipolar_names.append(f"{anode}-{cathode}")

    return np.array(bipolar_data, dtype=np.float32), bipolar_names


def sec_to_hmsms(sec: float) -> str:
    ms_total = int(round(sec * 1000))
    s, ms = divmod(ms_total, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def create_annotation_centered_epochs(
        data,
        annotations,
        sfreq,
        positive_label='*',
        negative_label='-',
        window=250  # in ms
):
    annotation_onsets_samples = np.round(annotations.onset * sfreq).astype(int)
    annotation_onsets_sec = np.asarray(annotations.onset, dtype=float)
    descriptions = np.array(annotations.description)

    labels = []
    epochs = []
    center_sec_list = []
    center_hmsms_list = []

    margin = int(sfreq * window / 1000)
    n_samples = data.shape[1]

    for i in range(len(annotations)):
        epoch_center_idx = int(annotation_onsets_samples[i])
        center_sec = float(annotation_onsets_sec[i])

        start = epoch_center_idx - margin
        stop = epoch_center_idx + margin

        if start < 0 or stop > n_samples:
            continue

        if descriptions[i] == positive_label:
            labels.append(1)
            epochs.append(data[:, start:stop])
            center_sec_list.append(center_sec)
            center_hmsms_list.append(sec_to_hmsms(center_sec))

        elif descriptions[i] == negative_label:
            labels.append(0)
            epochs.append(data[:, start:stop])
            center_sec_list.append(center_sec)
            center_hmsms_list.append(sec_to_hmsms(center_sec))

    return (
        np.array(epochs),
        np.array(labels),
        np.array(center_sec_list, dtype=np.float64),
        np.array(center_hmsms_list, dtype=object),
    )


def preprocess_edf_to_windows(
        edf_path: str,
        l_freq: float = 0.5,
        h_freq: float = 40.0,
        high_factor: float = 8.0,
        low_factor: float = 10.0,
        positive_label: str = '*',
        negative_label: str = '-',
        montage_pairs: list[tuple[str, str]] = LONGITUDINAL_PAIRS,
):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.rename_channels(lambda ch: ch.replace('EEG ', '').strip())

    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    # ΔΕΝ κάνεις average reference εδώ —
    # το bipolar montage είναι το ίδιο reference

    raw.filter(
        l_freq=l_freq, h_freq=h_freq,
        method='fir', fir_design='firwin',
        phase='zero', verbose="ERROR",
    )

    raw.resample(80, verbose="ERROR")

    bad_chs = mark_and_interpolate_bad_channels(  # πρώτα bad channels
        raw, high_factor=high_factor,
        low_factor=low_factor, plot=False,
    )

    raw.set_eeg_reference('average', verbose="ERROR")

    data = raw.get_data()
    ch_names = np.array(raw.ch_names, dtype=object)  # <-- 19 κανάλια
    sfreq = float(raw.info['sfreq'])
    annotations = raw.annotations

    epochs, labels, center_sec, center_hmsms = create_annotation_centered_epochs(
        data=data,
        sfreq=sfreq,
        annotations=annotations,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    return (
        epochs.astype(np.float32),
        labels.astype(np.uint8),
        center_sec.astype(np.float64),
        center_hmsms.astype(object),
        sfreq,
        ch_names,
        np.array(bad_chs, dtype=object),
    )


