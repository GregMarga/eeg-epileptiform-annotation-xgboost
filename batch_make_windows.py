# batch_make_windows.py
from pathlib import Path
import re
import numpy as np
import preprocessing  # imports preprocess_edf_to_windows from preprocessing.py


def patient_id_from_filename(filename: str):
    m = re.match(r"^P(\d+)_", filename)
    return int(m.group(1)) if m else None

def main():
    data_dir = Path("../../../data")
    out_dir = data_dir / "labram_windows_cache"
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

        windows, ch_names = preprocessing.create_sliding_windows_from_eeg(
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
        )

        print(f"  Saved: {out_path} | windows={windows.shape} ")

    print("Done.")

if __name__ == "__main__":
    main()
