import mne
import numpy as np
from mne.channels import make_standard_montage
from pathlib import Path
import re

def find_real_annotation_center(
        data,
        annotation_onset_point,
        sfreq
):
    margin = int(sfreq * 0.25)
    slice_data = (data[:, annotation_onset_point - margin:annotation_onset_point + margin])
    max_idx = np.argmax(slice_data)
    _, t = np.unravel_index(max_idx, slice_data.shape)
    center = annotation_onset_point - margin + t
    return center


def create_annotation_centered_epochs(
        data,
        annotations,
        sfreq,
        positive_label='*',
        negative_label='-',
        window=500  # in ms
):
    annotation_onsets = np.round(annotations.onset * sfreq).astype(int)
    descriptions = np.array(annotations.description)

    labels = []
    epochs = []

    margin = int(sfreq * window / 1000)
    n_samples = data.shape[1]
    for i in range(len(annotations)):
        real_center_idx = find_real_annotation_center(data, annotation_onsets[i], sfreq)
        epoch_center_idx = int(real_center_idx)
        start = epoch_center_idx - margin
        stop = epoch_center_idx + margin

        if start < 0 or stop > n_samples:
            continue

        if descriptions[i] == positive_label:
            labels.append(1)
            epoch = data[:, start:stop]
            epochs.append(epoch)
        elif descriptions[i] == negative_label:
            labels.append(0)
            epoch = data[:, start:stop]
            epochs.append(epoch)

    return np.array(epochs), np.array(labels)


def batch_create_annotation_windows_from_eeg(
        edf_path: str,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
):
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
    annotations = raw.annotations

    print("Data shape (channels, samples):", data_uv.shape)
    print("Sampling frequency:", sfreq)

    epochs, labels = create_annotation_centered_epochs(data=data_uv, annotations=annotations, sfreq=sfreq)

    return (
        epochs.astype(np.float32),
        ch_names,
        labels
    )
def patient_id_from_filename(filename: str):
    m = re.match(r"^P(\d+)_", filename)
    return int(m.group(1)) if m else None

def main():
    data_dir = Path("../../../data")
    out_dir = data_dir / "labram_labeled_windows_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(data_dir.glob("*.edf"))

    selected = []
    for p in edf_files:
        pid = patient_id_from_filename(p.name)
        if pid is None:
            continue
        selected.append(p)

    print(f"Found {len(selected)} EDF files")

    for i, edf_path in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {edf_path.name}")

        windows, ch_names,labels = batch_create_annotation_windows_from_eeg(
            edf_path=str(edf_path),
            l_freq=0.1,
            h_freq=75.0,
        )

        out_path = out_dir / f"{edf_path.stem}_windows.npz"
        np.savez_compressed(
            out_path,
            windows=windows,
            ch_names=ch_names,
            edf_name=edf_path.name,
            labels=labels.astype(np.uint8)
        )

        print(f"  Saved: {out_path} | windows={windows.shape} ")

    print("Done.")

if __name__ == "__main__":
    main()