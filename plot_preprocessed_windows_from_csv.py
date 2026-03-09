from pathlib import Path
import pandas as pd
import preprocessing


def main():
    edf_path = Path("../../../data/P70_GHB_M1679_0000078.edf")
    csv_path = Path("../../../data/P70_hard_windows.csv")

    hop_sec = 0.25
    window_sec = 0.5
    context_sec = 10.0

    raw, bad_chs = preprocessing.preprocess_raw_edf(
        edf_path=str(edf_path),
        l_freq=0.5,
        h_freq=40.0,
        high_factor=8.0,
        low_factor=10.0,
    )

    print("Bad channels:", bad_chs)

    df = pd.read_csv(csv_path)
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    pos_df = df[df["label"] == "positive"].head(1)
    neg_df = df[df["label"] == "negative"].head(10)
    selected_df = pd.concat([pos_df, neg_df], ignore_index=True)

    for _, row in selected_df.iterrows():
        idx = int(row["window_index"])
        prob = float(row["probability"])
        label = row["label"]
        center_txt = row["center_hmsms"]

        center_sec = idx * hop_sec + 0.5 * window_sec
        tmin = max(0.0, center_sec - context_sec / 2)
        tmax = center_sec + context_sec / 2

        seg = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=False)

        seg.plot(
            duration=context_sec,
            n_channels=len(seg.ch_names),
            scalings={"eeg": 50e-6},
            title=f"idx={idx} | {label} | center={center_txt} | proba={prob:.6f}",
            block=True,
        )


if __name__ == "__main__":
    main()