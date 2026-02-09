# batch_make_windows.py
from pathlib import Path
import re
import numpy as np
import preprocessing  # imports preprocess_edf_to_windows from preprocessing.py


def patient_id_from_filename(filename: str):
    m = re.match(r"^P(\d+)_", filename)
    return int(m.group(1)) if m else None


def main():
    data_dir = Path("../data")
    out_dir = data_dir / "windows_cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(data_dir.glob("*.edf"))

    # "before P28" => keep Pxx < 28
    selected = []
    for p in edf_files:    ## ayto na to allaksw meta eksairei to arxeio 28
        pid = patient_id_from_filename(p.name)
        if pid is None:
            continue
        if pid != 28:
            selected.append(p)

    print(f"Found {len(selected)} EDF files with Pxx != 28")

    for i, edf_path in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {edf_path.name}")

        windows, labels, sfreq, ch_names, bad_chs, kept_ann_idx, kept_onset_sec, kept_desc = preprocessing.preprocess_edf_to_windows(
            edf_path=str(edf_path),
            l_freq=0.5,
            h_freq=40.0,
            high_factor=8.0,
            low_factor=10.0,
            positive_label="*",
            negative_label="-",
        )

        out_path = out_dir / f"{edf_path.stem}_windows.npz"
        np.savez_compressed(
            out_path,
            windows=windows,
            labels=labels,
            sfreq=sfreq,
            ch_names=ch_names,
            bad_chs=bad_chs,
            edf_name=edf_path.name,
            kept_ann_idx=kept_ann_idx,
            kept_onset_sec=kept_onset_sec,
            kept_desc=kept_desc,
        )

        print(f"  Saved: {out_path} | windows={windows.shape} positives={int(labels.sum())}/{len(labels)}")

    print("Done.")


if __name__ == "__main__":
    main()
