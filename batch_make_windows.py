# batch_make_windows.py
from pathlib import Path
import re
import numpy as np
import preprocessing  # imports preprocess_edf_to_windows + parse_channel_detection_report


REPORT_PATH = Path("../../../data/channel_detection_details.txt")


def patient_id_from_filename(filename: str):
    m = re.match(r"^P(\d+)_", filename)
    return int(m.group(1)) if m else None


def main():
    data_dir = Path("../../../data")
    out_dir = data_dir / "windows_cache_80hz_pyprep"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Parse the PyPREP report once; preprocess_edf_to_windows needs it for every file
    report = preprocessing.parse_channel_detection_report(REPORT_PATH)
    print(f"Loaded PyPREP report with {len(report)} entries")

    edf_files = sorted(data_dir.glob("*.edf"))

    # Drop patient P28 (TODO: revisit if we want to include them later)
    selected = []
    for p in edf_files:
        pid = patient_id_from_filename(p.name)
        if pid is None:
            continue
        if pid != 28:
            selected.append(p)

    print(f"Found {len(selected)} EDF files with Pxx != 28")

    for i, edf_path in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {edf_path.name}")

        windows, labels, center_sec, center_hmsms, sfreq, ch_names, bad_chs = preprocessing.preprocess_edf_to_windows(
            edf_path=str(edf_path),
            report=report,
            l_freq=0.5,
            h_freq=40.0,
            positive_label="*",
            negative_label="-",
        )

        out_path = out_dir / f"{edf_path.stem}_windows.npz"
        np.savez_compressed(
            out_path,
            windows=windows,
            labels=labels,
            center_sec=center_sec,
            center_hmsms=center_hmsms,
            sfreq=sfreq,
            ch_names=ch_names,
            bad_chs=bad_chs,
            edf_name=edf_path.name,
        )

        print(f"  Saved: {out_path} | windows={windows.shape} positives={int(labels.sum())}/{len(labels)}")

    print("Done.")


if __name__ == "__main__":
    main()