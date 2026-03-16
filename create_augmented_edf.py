from pathlib import Path
import csv
import mne


def load_false_negative_rows(csv_path: Path):
    rows = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label", "").strip().lower()
            if label == "false_negative":
                rows.append({
                    "center_sec": float(row["center_sec"]),
                    "probability": float(row["probability"]),
                })
    return rows


def keep_only_pd_annotations(raw: mne.io.BaseRaw) -> mne.Annotations:
    keep_onset = []
    keep_duration = []
    keep_desc = []

    for onset, duration, desc in zip(
        raw.annotations.onset,
        raw.annotations.duration,
        raw.annotations.description,
    ):
        desc_up = desc.strip().upper()
        if desc_up in {"PD_START", "PD_STOP"}:
            keep_onset.append(float(onset))
            keep_duration.append(float(duration))
            keep_desc.append(desc_up)

    orig_time = raw.annotations.orig_time
    if orig_time is None:
        orig_time = raw.info.get("meas_date", None)

    return mne.Annotations(
        onset=keep_onset,
        duration=keep_duration,
        description=keep_desc,
        orig_time=orig_time,
    )


def build_false_negative_annotations(
    fn_rows,
    orig_time,
    duration: float = 0.0,
) -> mne.Annotations:
    onsets = [float(r["center_sec"]) for r in fn_rows]
    descriptions = [
        f"FALSE_NEGATIVE|p={float(r['probability']):.6f}"
        for r in fn_rows
    ]

    return mne.Annotations(
        onset=onsets,
        duration=[float(duration)] * len(onsets),
        description=descriptions,
        orig_time=orig_time,
    )


def main():
    raw_edf_path = Path("../../../data/extra_data/P70_GHB_M1679_0000078.edf")
    csv_path = Path("../../../data/P70_mislabeled_windows.csv")
    out_edf_path = Path("../../../data/P70_with_fn_annotations.edf")

    raw = mne.io.read_raw_edf(raw_edf_path, preload=False, verbose="ERROR")

    pd_annotations = keep_only_pd_annotations(raw)
    fn_rows = load_false_negative_rows(csv_path)

    fn_annotations = build_false_negative_annotations(
        fn_rows=fn_rows,
        orig_time=pd_annotations.orig_time,
        duration=0.0,
    )

    raw_out = raw.copy()
    raw_out.set_annotations(pd_annotations + fn_annotations)

    raw_out.export(out_edf_path, fmt="edf", overwrite=True)

    print(f"Kept PD annotations: {len(pd_annotations)}")
    print(f"Added FALSE_NEGATIVE annotations: {len(fn_annotations)}")
    print(f"Saved EDF to: {out_edf_path.resolve()}")


if __name__ == "__main__":
    main()
# from pathlib import Path
# import csv
# import mne
#
#
# def load_false_positive_rows(csv_path: Path):
#     rows = []
#     with csv_path.open("r", newline="") as f:
#         reader = csv.DictReader(f)
#         for row in reader:
#             label = row.get("label", "").strip().lower()
#             if label == "false_positive":
#                 rows.append({
#                     "center_sec": float(row["center_sec"]),
#                     "probability": float(row["probability"]),
#                 })
#     return rows
#
# def keep_only_pd_annotations(raw: mne.io.BaseRaw) -> mne.Annotations:
#     keep_onset = []
#     keep_duration = []
#     keep_desc = []
#
#     for onset, duration, desc in zip(
#         raw.annotations.onset,
#         raw.annotations.duration,
#         raw.annotations.description,
#     ):
#         desc_up = desc.strip().upper()
#         if desc_up in {"PD_START", "PD_STOP"}:
#             keep_onset.append(float(onset))
#             keep_duration.append(float(duration))
#             keep_desc.append(desc_up)
#
#     orig_time = raw.annotations.orig_time
#     if orig_time is None:
#         orig_time = raw.info.get("meas_date", None)
#
#     return mne.Annotations(
#         onset=keep_onset,
#         duration=keep_duration,
#         description=keep_desc,
#         orig_time=orig_time,
#     )
#
#
# def build_false_positive_annotations(
#     fp_rows,
#     orig_time,
#     duration: float = 0.0,
# ) -> mne.Annotations:
#     onsets = [float(r["center_sec"]) for r in fp_rows]
#     descriptions = [
#         f"FALSE_POSITIVE|p={float(r['probability']):.6f}"
#         for r in fp_rows
#     ]
#
#     return mne.Annotations(
#         onset=onsets,
#         duration=[float(duration)] * len(onsets),
#         description=descriptions,
#         orig_time=orig_time,
#     )
#
#
# def main():
#     raw_edf_path = Path("../../../data/extra_data/P70_GHB_M1679_0000078.edf")
#     csv_path = Path("../../../data/P70_mislabeled_windows.csv")
#     out_edf_path = Path("../../../data/P70_with_fp_annotations.edf")
#
#     raw = mne.io.read_raw_edf(raw_edf_path, preload=False, verbose="ERROR")
#
#     pd_annotations = keep_only_pd_annotations(raw)
#     fp_rows = load_false_positive_rows(csv_path)
#
#     fp_annotations = build_false_positive_annotations(
#         fp_rows=fp_rows,
#         orig_time=pd_annotations.orig_time,
#         duration=0.0,
#     )
#
#     raw_out = raw.copy()
#     raw_out.set_annotations(pd_annotations + fp_annotations)
#
#     raw_out.export(out_edf_path, fmt="edf", overwrite=True)
#
#     print(f"Kept PD annotations: {len(pd_annotations)}")
#     print(f"Added FALSE_POSITIVE annotations: {len(fp_annotations)}")
#     print(f"Saved EDF to: {out_edf_path.resolve()}")
#
#
# if __name__ == "__main__":
#     main()
