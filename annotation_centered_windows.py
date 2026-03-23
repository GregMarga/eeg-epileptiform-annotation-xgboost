import mne
import numpy as np
from mne.channels import make_standard_montage
from pathlib import Path
import re


def create_annotation_centered_epochs(
        data,
        annotations,
        sfreq,
        positive_label='*',
        negative_label='-',
        window=500,   # half-window in ms -> total window = 1 sec
):
    annotation_onsets = np.round(annotations.onset * sfreq).astype(int)
    descriptions = np.array(annotations.description)

    labels = []
    epochs = []

    margin = int(sfreq * window / 1000)
    n_samples = data.shape[1]

    for i in range(len(annotations)):
        center_idx = annotation_onsets[i]
        desc = descriptions[i]

        start = center_idx - margin
        stop = center_idx + margin

        if start < 0 or stop > n_samples:
            continue

        if desc == positive_label:
            labels.append(1)
            epochs.append(data[:, start:stop])
        elif desc == negative_label:
            labels.append(0)
            epochs.append(data[:, start:stop])

    return np.array(epochs), np.array(labels)


def preprocess_for_labram(
        raw,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
):
    raw.pick("eeg")
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())

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

    # Notch filter
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

    # Resample to target sampling frequency
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    # Convert signal from volts to microvolts
    data_uv = raw.get_data() * 1e6

    return raw, data_uv


def batch_create_annotation_windows_from_eeg(
        edf_path: str,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    raw, data_uv = preprocess_for_labram(
        raw,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=notch_freq,
        target_sfreq=target_sfreq,
    )

    sfreq = float(raw.info["sfreq"])
    ch_names = np.array(raw.ch_names, dtype=object)
    annotations = raw.annotations

    print("Data shape (channels, samples):", data_uv.shape)
    print("Sampling frequency:", sfreq)

    epochs, labels = create_annotation_centered_epochs(
        data=data_uv,
        annotations=annotations,
        sfreq=sfreq,
    )

    return (
        epochs.astype(np.float32),
        ch_names,
        labels,
    )


def patient_id_from_filename(filename: str):
    m = re.match(r"^P(\d+)_", filename)
    return int(m.group(1)) if m else None


def main():
    data_dir = Path("../../../data")
    out_dir = data_dir / "corrected_labram_labeled_windows_cache"
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

        windows, ch_names, labels = batch_create_annotation_windows_from_eeg(
            edf_path=str(edf_path),
            l_freq=0.1,
            h_freq=75.0,
            notch_freq=50.0,
            target_sfreq=200.0,
        )

        out_path = out_dir / f"{edf_path.stem}_windows.npz"
        np.savez_compressed(
            out_path,
            windows=windows,
            ch_names=ch_names,
            edf_name=edf_path.name,
            labels=labels.astype(np.uint8)
        )
        print(f"  Saved: {out_path} | windows={windows.shape}")

    print("Done.")


if __name__ == "__main__":
    main()