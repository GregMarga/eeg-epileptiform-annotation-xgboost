import mne
import numpy as np
from mne.channels import make_standard_montage
from pyprep import NoisyChannels
from pathlib import Path
import re


def create_annotation_centered_epochs(
        data,
        annotations,
        sfreq,
        positive_label='*',
        negative_label='-',
        window=500,
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


def detect_and_interpolate_bad_channels_pyprep(
        raw: mne.io.BaseRaw,
        random_state: int = 42,
        verbose: bool = True,
):
    """Detect bad channels using PyPREP on broadband data.

    Should be called BEFORE the final low-pass filter so that the
    high-frequency noise criterion has signal in the >50 Hz band.
    """
    nc = NoisyChannels(raw, random_state=random_state, do_detrend=False)
    nc.find_all_bads(ransac=True, channel_wise=False)

    bads_by_criterion = {
        "nan":          nc.bad_by_nan,
        "flat":         nc.bad_by_flat,
        "deviation":    nc.bad_by_deviation,
        "hf_noise":     nc.bad_by_hf_noise,
        "correlation":  nc.bad_by_correlation,
        "ransac":       nc.bad_by_ransac,
        "dropout":      nc.bad_by_dropout,
    }

    all_bads = nc.get_bads()

    if verbose:
        print(f"  PyPREP bad channel summary:")
        for criterion, bads in bads_by_criterion.items():
            if bads:
                print(f"    {criterion:12s}: {bads}")
        print(f"  Total unique bad channels: {all_bads}")

    raw.info['bads'] = all_bads

    if len(all_bads) > 0:
        raw.interpolate_bads(reset_bads=True)

    return all_bads, bads_by_criterion


def preprocess_for_labram(
        raw,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
        prep_hp_freq: float = 1.0,
        use_pyprep: bool = True,
):
    """LaBraM-matching preprocessing pipeline.

    Frequencies match the LaBraM pre-training (0.1-75 Hz, 200 Hz fs)
    so the input distribution stays close to what the model was trained on.

    Order:
      1. Pick EEG, set montage
      2. Light high-pass (1 Hz) + notch on broadband data
      3. PyPREP bad channel detection on broadband data
      4. Interpolate bad channels
      5. Final bandpass (0.1 - 75 Hz) matching pre-training
      6. Average reference (after bad channel removal)
      7. Resample to 200 Hz
    """
    raw.pick("eeg")
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())

    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    # Light high-pass to remove DC drift, no low-pass yet
    # (keeps broadband content for PyPREP's HF noise criterion)
    raw.filter(
        l_freq=prep_hp_freq,
        h_freq=None,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose="ERROR",
    )

    # Notch filter for line noise
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

    # Bad channel detection on broadband data
    if use_pyprep:
        bad_channels, bads_by_criterion = detect_and_interpolate_bad_channels_pyprep(raw)
    else:
        bad_channels, bads_by_criterion = [], {}

    # Final bandpass matching LaBraM pre-training (0.1 - 75 Hz)
    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose="ERROR",
    )

    # Average reference (after bad channel interpolation)
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")

    # Resample to 200 Hz (LaBraM target)
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    # Convert to μV
    data_uv = raw.get_data() * 1e6

    return raw, data_uv, bad_channels, bads_by_criterion


def batch_create_annotation_windows_from_eeg(
        edf_path: str,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
        prep_hp_freq: float = 1.0,
        use_pyprep: bool = True,
):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    raw, data_normalized, bad_channels, bads_by_criterion = preprocess_for_labram(
        raw,
        l_freq=l_freq,
        h_freq=h_freq,
        notch_freq=notch_freq,
        target_sfreq=target_sfreq,
        prep_hp_freq=prep_hp_freq,
        use_pyprep=use_pyprep,
    )

    sfreq = float(raw.info["sfreq"])
    ch_names = np.array(raw.ch_names, dtype=object)
    annotations = raw.annotations

    print(f"  Data shape (channels, samples): {data_normalized.shape}")
    print(f"  Sampling frequency: {sfreq}")

    epochs, labels = create_annotation_centered_epochs(
        data=data_normalized,
        annotations=annotations,
        sfreq=sfreq,
    )

    return (
        epochs.astype(np.float32),
        ch_names,
        labels,
        bad_channels,
        bads_by_criterion,
    )


def patient_id_from_filename(filename: str):
    m = re.match(r"^P(\d+)_", filename)
    return int(m.group(1)) if m else None


def main():
    data_dir = Path("../../../data")
    out_dir = data_dir / "labram_labeled_windows_1s"
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

        windows, ch_names, labels, bad_channels, bads_by_criterion = batch_create_annotation_windows_from_eeg(
            edf_path=str(edf_path),
            l_freq=0.1,
            h_freq=75.0,
            notch_freq=50.0,
            target_sfreq=200.0,
            prep_hp_freq=1.0,
            use_pyprep=True,
        )

        out_path = out_dir / f"{edf_path.stem}_windows.npz"
        np.savez_compressed(
            out_path,
            windows=windows,
            ch_names=ch_names,
            edf_name=edf_path.name,
            labels=labels.astype(np.uint8),
            bad_channels=np.array(bad_channels, dtype=object),
        )
        print(f"  Saved: {out_path} | windows={windows.shape}")

    print("Done.")


if __name__ == "__main__":
    main()