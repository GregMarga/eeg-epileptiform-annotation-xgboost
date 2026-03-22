import os
import glob
import pandas as pd
import mne

# ======================
# CONFIG (EDIT THESE)
# ======================
CSV_DIR = r"C:\Users\gregm\KU Leuven\Thesis\Models\data\lopo_hard_errors"
EDF_DIR = r"C:\Users\gregm\KU Leuven\Thesis\data"

WINDOW_SEC = 10
HALF_WINDOW = WINDOW_SEC / 2
SCALING = 100e-6  # 50 μV


# ======================
def time_to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ======================
def process_all():
    csv_files = glob.glob(os.path.join(CSV_DIR, "*_hard_errors.csv"))

    if not csv_files:
        print("No CSV files found.")
        return

    edf_cache = {}

    for csv_file in csv_files:
        print(f"\nProcessing: {csv_file}")
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            recording = row["recording_file"]
            t_center = time_to_seconds(row["window_center_time"])

            # Ground truth label
            gt_label = "GT Positive" if row["y_true"] == 1 else "GT Negative"

            edf_path = os.path.join(EDF_DIR, recording)

            if not os.path.exists(edf_path):
                print(f"Missing EDF: {edf_path}")
                continue

            # ======================
            # Load EDF (cache)
            # ======================
            if edf_path not in edf_cache:
                print(f"Loading EDF: {recording}")
                edf_cache[edf_path] = mne.io.read_raw_edf(
                    edf_path, preload=True, verbose=False
                )

            raw = edf_cache[edf_path]

            # ======================
            # Crop window
            # ======================
            tmin = max(0, t_center - HALF_WINDOW)
            tmax = t_center + HALF_WINDOW

            raw_segment = raw.copy().crop(tmin=tmin, tmax=tmax)

            # ======================
            # Title
            # ======================
            title = f"{recording} | {gt_label}"

            # ======================
            # Plot
            # ======================
            raw_segment.plot(
                duration=WINDOW_SEC,
                scalings=dict(eeg=SCALING),
                title=title,
                show=True,
                block=True
            )


# ======================
# ENTRY POINT
# ======================
if __name__ == "__main__":
    print("Starting hard error visualization...")
    process_all()
    print("Done.")