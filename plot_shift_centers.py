import csv
from pathlib import Path
import mne


def plot_shift_from_csv_row(
    edf_path,
    row,
    l_freq=0.1,
    h_freq=75.0,
    notch_freq=50.0,
    target_sfreq=200.0,
):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
    raw.pick("eeg")
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())

    raw.filter(
        l_freq=l_freq,
        h_freq=h_freq,
        method="fir",
        fir_design="firwin",
        phase="zero",
        verbose="ERROR",
    )
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    orig_sec = float(row["orig_sec"])
    new_sec = float(row["new_sec"])

    duration = 0.5
    plot_start_sec = max(0.0, orig_sec - duration / 2)

    ann = mne.Annotations(
        onset=[orig_sec, new_sec],
        duration=[0.0, 0.0],
        description=["ORIG", "SHIFTED"],
    )

    raw.set_annotations(ann)

    raw.plot(
        start=plot_start_sec,
        duration=duration,
        scalings={"eeg": 50e-6},  # 50 μV
        block=True,
    )


def main():
    data_dir = Path("../../../data")
    csv_path = (
        data_dir
        / "labram_labeled_windows_cache"
        / "P20_GHB_00015_0000348_annotation_center_shifts.csv"
    )

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows):

        edf_path = data_dir / row["edf_name"]

        print(f"\nShowing annotation {i+1}/{len(rows)}")
        print("shift (ms):", row["shift_ms"])

        plot_shift_from_csv_row(edf_path, row)


if __name__ == "__main__":
    main()