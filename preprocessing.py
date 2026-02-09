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



def create_annotation_centered_epochs(
        data,
        annotations,
        sfreq,
        positive_label='*',
        negative_label='-',
        window=250  # in ms
):
    annotation_onsets = np.round(annotations.onset * sfreq).astype(int)
    onsets_sec = np.array(annotations.onset, dtype=float)
    descriptions = np.array(annotations.description, dtype=object)

    labels = []
    epochs = []

    kept_ann_idx = []
    kept_onset_sec = []
    kept_desc = []

    margin = int(sfreq * window / 1000)
    n_samples = data.shape[1]

    for i in range(len(annotations)):
        epoch_center_idx = int(annotation_onsets[i])
        start = epoch_center_idx - margin
        stop = epoch_center_idx + margin

        if start < 0 or stop > n_samples:
            continue

        if descriptions[i] == positive_label:
            lab = 1
        elif descriptions[i] == negative_label:
            lab = 0
        else:
            continue  # ignore other labels

        epochs.append(data[:, start:stop])
        labels.append(lab)

        kept_ann_idx.append(i)
        kept_onset_sec.append(onsets_sec[i])
        kept_desc.append(descriptions[i])

    return (
        np.array(epochs, dtype=np.float32),
        np.array(labels, dtype=np.uint8),
        np.array(kept_ann_idx, dtype=np.int32),
        np.array(kept_onset_sec, dtype=np.float64),
        np.array(kept_desc, dtype=object),
    )


def preprocess_edf_to_windows(
        edf_path: str,
        l_freq: float = 0.5,
        h_freq: float = 40.0,
        high_factor: float = 8.0,
        low_factor: float = 10.0,
        positive_label: str = '*',
        negative_label: str = '-',
):

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

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
    annotations=raw.annotations

    epochs, labels, kept_ann_idx, kept_onset_sec, kept_desc = create_annotation_centered_epochs(
        data=data,
        sfreq=sfreq,
        annotations=annotations,
        positive_label=positive_label,
        negative_label=negative_label,
    )

    ch_names = np.array(raw.ch_names, dtype=object)

    return (
        epochs,
        labels,
        sfreq,
        ch_names,
        np.array(bad_chs, dtype=object),
        kept_ann_idx,
        kept_onset_sec,
        kept_desc,
    )

## check
# BASE = Path(__file__).resolve().parents[2]  # Thesis
# edf_path = BASE / "Data" / "p28_GHB_00000_0002249_0001.edf"
#
# epochs, labels, sfreq, ch_names, bad_chs = preprocess_edf_to_windows(
#     edf_path=str(edf_path),
#     l_freq=0.5,
#     h_freq=40.0,
#     positive_label='*',
#     negative_label='-',
# )
#
# print("Epochs shape:", epochs.shape)   # (n_epochs, n_channels, n_samples)
# print("Labels shape:", labels.shape)   # (n_epochs,)
# print("Sampling freq:", sfreq)
# print("Channels:", ch_names)
# print("Bad channels:", bad_chs)
#
#
# print("positives:", labels.sum())
# print("negatives:", (labels == 0).sum())
#
# print("unique labels:", np.unique(labels))
#
# assert epochs.ndim == 3
# assert len(epochs) == len(labels)
# assert epochs.shape[1] == len(ch_names)

