import mne
import numpy as np
from mne.channels import make_standard_montage
from pathlib import Path
import re
import csv


def find_real_annotation_center(
        data,
        annotation_onset_point,
        sfreq
):
    margin = int(sfreq * 0.25)
    start = max(0, annotation_onset_point - margin)
    stop = min(data.shape[1], annotation_onset_point + margin)

    slice_data = data[:, start:stop]
    min_idx = np.argmax(slice_data)   # negative peak
    _, t = np.unravel_index(min_idx, slice_data.shape)
    center = start + t
    return center


def create_annotation_centered_epochs(
        data,
        annotations,
        sfreq,
        positive_label='*',
        negative_label='-',
        window=500,   # epoch half-window in ms
        plot_window_sec=10.0,
        edf_name=None,
):
    annotation_onsets = np.round(annotations.onset * sfreq).astype(int)
    descriptions = np.array(annotations.description)

    labels = []
    epochs = []
    shift_rows = []

    margin = int(sfreq * window / 1000)
    n_samples = data.shape[1]
    half_plot_window_sec = plot_window_sec / 2.0

    for i in range(len(annotations)):
        original_center_idx = annotation_onsets[i]
        desc = descriptions[i]

        # recenter only positive annotations
        if desc == positive_label:
            final_center_idx = find_real_annotation_center(
                data, original_center_idx, sfreq
            )
        else:
            final_center_idx = original_center_idx

        shift_samples = final_center_idx - original_center_idx
        shift_ms = (shift_samples / sfreq) * 1000.0

        orig_sec = original_center_idx / sfreq
        new_sec = final_center_idx / sfreq

        plot_start_sec = max(0.0, orig_sec - half_plot_window_sec)
        plot_end_sec = plot_start_sec + plot_window_sec

        total_duration_sec = n_samples / sfreq
        if plot_end_sec > total_duration_sec:
            plot_end_sec = total_duration_sec
            plot_start_sec = max(0.0, plot_end_sec - plot_window_sec)

        shift_rows.append({
            "edf_name": edf_name if edf_name is not None else "",
            "annotation_idx": i,
            "description": str(desc),
            "orig_sample": int(original_center_idx),
            "new_sample": int(final_center_idx),
            "orig_sec": float(orig_sec),
            "new_sec": float(new_sec),
            "shift_samples": int(shift_samples),
            "shift_ms": float(shift_ms),
            "plot_start_sec": float(plot_start_sec),
            "plot_end_sec": float(plot_end_sec),
        })

        orig_val = np.min(data[:, original_center_idx])
        shift_val = np.min(data[:, final_center_idx])

        if desc == positive_label and abs(shift_ms) > 50:
            print(
                f"[CENTER SHIFT] annotation #{i} "
                f"orig_time={orig_sec:.3f}s "
                f"shifted_time={new_sec:.3f}s "
                f"shift={shift_ms:.1f} ms | "
                f"orig_val={orig_val:.2f} uV "
                f"shifted_val={shift_val:.2f} uV"
            )

        epoch_center_idx = int(final_center_idx)
        start = epoch_center_idx - margin
        stop = epoch_center_idx + margin

        if start < 0 or stop > n_samples:
            continue

        if desc == positive_label:
            labels.append(1)
            epoch = data[:, start:stop]
            epochs.append(epoch)
        elif desc == negative_label:
            labels.append(0)
            epoch = data[:, start:stop]
            epochs.append(epoch)

    return np.array(epochs), np.array(labels), shift_rows


def batch_create_annotation_windows_from_eeg(
        edf_path: str,
        l_freq: float = 0.1,
        h_freq: float = 75.0,
        notch_freq: float = 50.0,
        target_sfreq: float = 200.0,
):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    raw.pick("eeg")
    raw.rename_channels(lambda ch: ch.replace('EEG ', '').strip())

    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing='ignore')

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method='fir',
        fir_design='firwin',
        phase='zero',
        verbose="ERROR",
    )

    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    data_uv = raw.get_data() * 1e6

    sfreq = float(raw.info['sfreq'])
    ch_names = np.array(raw.ch_names, dtype=object)
    annotations = raw.annotations

    print("Data shape (channels, samples):", data_uv.shape)
    print("Sampling frequency:", sfreq)

    epochs, labels, shift_rows = create_annotation_centered_epochs(
        data=data_uv,
        annotations=annotations,
        sfreq=sfreq,
        edf_name=Path(edf_path).name,
    )

    return (
        epochs.astype(np.float32),
        ch_names,
        labels,
        shift_rows
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

    fieldnames = [
        "edf_name",
        "annotation_idx",
        "description",
        "orig_sample",
        "new_sample",
        "orig_sec",
        "new_sec",
        "shift_samples",
        "shift_ms",
        "plot_start_sec",
        "plot_end_sec",
    ]

    for i, edf_path in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {edf_path.name}")

        windows, ch_names, labels, shift_rows = batch_create_annotation_windows_from_eeg(
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
        print(f"  Saved: {out_path} | windows={windows.shape}")

        csv_path = out_dir / f"{edf_path.stem}_annotation_center_shifts.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(shift_rows)

        print(f"  Saved shift CSV: {csv_path}")

    print("Done.")


if __name__ == "__main__":
    main()