"""
extract_labram_annot_centered.py
================================
Annotation-CENTERED (POSITIVE-only) variant of the LaBraM sliding-window script.

Same LaBraM preprocessing as the sliding version (0.1-75 Hz + 50 Hz notch,
per-segment PyPREP, average reference, 200 Hz), so these annotation-centered
positive windows are directly comparable to the LaBraM sliding windows.

Instead of dense sliding windows, this extracts ONE 1s window CENTERED on each
'*' discharge mark. There are no '-' marks in these recordings, so this file
contains POSITIVES only -- the negatives for any pool come from the sliding
NON_PD windows. The window is cut from the full padded, preprocessed crop by
absolute time, so marks near a segment edge still get valid +/-0.5 s of context.

Output: raw windows (for later LaBraM embedding extraction) in a NEW folder.
"""

import re
import mne
import numpy as np
from pathlib import Path
from mne.channels import make_standard_montage
from pyprep import NoisyChannels


# ----------------------------- config -----------------------------

EDGE_PAD_SEC = 60.0

NOTE_PREFIX = "Note : "
PD_START, PD_STOP = "PD_START", "PD_STOP"
NON_PD_START, NON_PD_STOP = "NON_PD_START", "NON_PD_STOP"
PD_MARK = "*"

SEGMENT_STARTS = {PD_START: "PD", NON_PD_START: "NON_PD"}
SEGMENT_STOPS = {PD_STOP: "PD", NON_PD_STOP: "NON_PD"}

OLD_TO_NEW_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

WINDOW_SEC = 1.0
TARGET_SFREQ = 200.0


# ------------------------- annotation parsing -------------------------

def strip_prefix(desc: str) -> str:
    """Remove the exact 'Note : ' export prefix, then strip whitespace."""
    return desc.removeprefix(NOTE_PREFIX).strip()


def parse_segments(raw):
    """Return (segments, pd_marks) using EXACT name matching.

    segments : list of {type: 'PD'|'NON_PD', start, stop} in seconds
    pd_marks : np.ndarray of '*' onset times in seconds
    """
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = [strip_prefix(d) for d in raw.annotations.description]

    print("Unique descriptions:")
    for d in sorted(set(descs)):
        print(repr(d))

    order = np.argsort(onsets, kind="stable")
    onsets = onsets[order]
    descs = [descs[i] for i in order]

    segments, pd_marks = [], []
    open_start = None

    for t, d in zip(onsets, descs):
        if d == PD_MARK:
            pd_marks.append(t)
        elif d in SEGMENT_STARTS:
            if open_start is not None:
                print(f"  ! new START before previous STOP at {t:.2f}s — dropping previous")
            open_start = (SEGMENT_STARTS[d], t)
        elif d in SEGMENT_STOPS:
            if open_start is None:
                print(f"  ! STOP without START at {t:.2f}s ({d}) — skipping")
                continue
            seg_type, t0 = open_start
            if SEGMENT_STOPS[d] != seg_type:
                print(f"  ! START/STOP type mismatch near {t:.2f}s — skipping")
            else:
                segments.append({"type": seg_type, "start": float(t0), "stop": float(t)})
            open_start = None

    return segments, np.sort(np.array(pd_marks, dtype=float))


# ----------------------- montage / channel names -----------------------

def _clean_name(name: str) -> str:
    n = name.replace("EEG ", "").strip()
    n = re.sub(r"[-_ ]?(REF|LE|AVG)\Z", "", n, flags=re.IGNORECASE).strip()
    return OLD_TO_NEW_1020.get(n.upper(), n)


def clean_and_set_montage(raw, verbose: bool = True):
    mapping = {ch: _clean_name(ch) for ch in raw.ch_names}
    mapping = {k: v for k, v in mapping.items() if k != v}
    if mapping:
        raw.rename_channels(mapping)

    raw.set_montage(make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")

    no_pos = [ch["ch_name"] for ch in raw.info["chs"]
              if np.isnan(np.asarray(ch["loc"][:3], dtype=float)).any()]
    if no_pos:
        if verbose:
            print(f"  Dropping {len(no_pos)} channel(s) without montage positions: {no_pos}")
        raw.drop_channels(no_pos)

    return raw


# ----------------------- bad channel detection -----------------------

def detect_and_interpolate_bad_channels_pyprep(raw, random_state=42, verbose=True):
    nc = NoisyChannels(raw, random_state=random_state, do_detrend=False)
    nc.find_bad_by_deviation()
    nc.find_bad_by_correlation()
    nc.find_bad_by_ransac(channel_wise=False)

    all_bads = nc.get_bads()
    if verbose:
        print(f"  PyPREP bad channels: {all_bads}")

    raw.info["bads"] = all_bads
    if len(all_bads) > 0:
        raw.interpolate_bads(reset_bads=True)
    return all_bads


# ----------------------- preprocessing (LaBraM) -----------------------

def preprocess_for_labram(raw, l_freq=0.1, h_freq=75.0, notch_freq=50.0,
                          target_sfreq=200.0, prep_hp_freq=1.0,
                          use_pyprep=True, random_state=42):
    raw.pick("eeg")
    clean_and_set_montage(raw)

    raw.filter(l_freq=prep_hp_freq, h_freq=None, method="fir",
               fir_design="firwin", phase="zero", verbose="ERROR")
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

    if use_pyprep:
        bad_channels = detect_and_interpolate_bad_channels_pyprep(raw, random_state=random_state)
    else:
        bad_channels = []

    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir",
               fir_design="firwin", phase="zero", verbose="ERROR")
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    data_uv = raw.get_data() * 1e6
    return raw, data_uv, bad_channels


# ----------------------- annotation-centered windowing -----------------------

def extract_segment_annot_windows(raw_full, seg, marks, rec_end_sec,
                                  edge_pad_sec=EDGE_PAD_SEC, window_sec=WINDOW_SEC,
                                  target_sfreq=TARGET_SFREQ, use_pyprep=True,
                                  random_state=42, **prep_kwargs):
    """Crop ONE segment (with padding), preprocess, then cut a 1s window centered
    on each '*' mark inside the segment (positives only)."""
    seg_start, seg_stop = seg["start"], seg["stop"]
    seg_marks = [float(t) for t in marks if seg_start <= t < seg_stop]
    if not seg_marks:
        return np.empty((0,)), np.array([]), np.array([]), None, None, []

    crop_tmin = max(0.0, seg_start - edge_pad_sec)
    crop_tmax = min(rec_end_sec, seg_stop + edge_pad_sec)
    if (seg_start - crop_tmin) < edge_pad_sec or (crop_tmax - seg_stop) < edge_pad_sec:
        print("    (segment near recording boundary — reduced padding)")

    raw_seg = raw_full.copy().crop(tmin=crop_tmin, tmax=crop_tmax).load_data(verbose="ERROR")
    raw_seg, data_uv, bad_channels = preprocess_for_labram(
        raw_seg, target_sfreq=target_sfreq, use_pyprep=use_pyprep,
        random_state=random_state, **prep_kwargs)
    sf = float(raw_seg.info["sfreq"])

    margin = int(round(window_sec / 2 * sf))   # 0.5 s each side
    n = data_uv.shape[1]

    windows, onsets_abs, centers_abs = [], [], []
    for t in seg_marks:
        center = int(round((t - crop_tmin) * sf))
        a, b = center - margin, center + margin
        if a < 0 or b > n:
            print(f"    (mark at {t:.2f}s too close to crop edge — skipped)")
            continue
        windows.append(data_uv[:, a:b])
        onsets_abs.append(crop_tmin + a / sf)   # absolute window start
        centers_abs.append(t)

    if not windows:
        return np.empty((0,)), np.array([]), np.array([]), list(raw_seg.ch_names), sf, list(bad_channels)

    epochs = np.stack(windows, axis=0).astype(np.float32)
    return (epochs, np.asarray(onsets_abs, dtype=float),
            np.asarray(centers_abs, dtype=float),
            list(raw_seg.ch_names), sf, list(bad_channels))


# ----------------------------- main -----------------------------

def main():
    data_dir = Path("../../../data/evaluation_recordings")
    out_dir = data_dir / "labram_annot_centered_windows_1s"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = "P58_GHB_M1681_0000031"
    edf_path = data_dir / f"{stem}.edf"
    if not edf_path.exists():
        raise FileNotFoundError(edf_path)

    raw_full = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    rec_end = float(raw_full.times[-1])

    segments, pd_marks = parse_segments(raw_full)
    n_pd = sum(s["type"] == "PD" for s in segments)
    n_non = sum(s["type"] == "NON_PD" for s in segments)
    print(f"{edf_path.name}: {len(segments)} segments "
          f"({n_pd} PD, {n_non} NON_PD), {len(pd_marks)} '*' marks, "
          f"recording {rec_end / 3600:.2f} h")

    if len(pd_marks) == 0:
        uniq = sorted(set(raw_full.annotations.description))
        print("  ! No '*' marks parsed. Unique raw annotation descriptions present:")
        for u in uniq:
            print(f"      {u!r}")
        return

    all_w, all_on, all_cen, all_sid, all_stype = [], [], [], [], []
    bad_lists, bad_seg_ids = [], []
    ch_names, sfreq = None, None

    for i, seg in enumerate(segments):
        seg_marks_here = int(np.sum((pd_marks >= seg["start"]) & (pd_marks < seg["stop"])))
        print(f"[seg {i:02d}] {seg['type']:6s} {seg['start']:.1f}-{seg['stop']:.1f}s "
              f"({seg['stop'] - seg['start']:.1f}s) | '*' marks={seg_marks_here}")
        if seg_marks_here == 0:
            continue

        epochs, onsets_abs, centers_abs, ch_names, sfreq, bad_channels = \
            extract_segment_annot_windows(
                raw_full, seg, pd_marks, rec_end,
                edge_pad_sec=EDGE_PAD_SEC, window_sec=WINDOW_SEC,
                target_sfreq=TARGET_SFREQ, use_pyprep=True, random_state=42,
                l_freq=0.1, h_freq=75.0, notch_freq=50.0, prep_hp_freq=1.0,
            )
        if len(epochs) == 0:
            continue
        all_w.append(epochs)
        all_on.append(onsets_abs)
        all_cen.append(centers_abs)
        all_sid.append(np.full(len(epochs), i, dtype=np.int64))
        all_stype.append(np.array([seg["type"]] * len(epochs), dtype=object))
        bad_lists.append(np.array(bad_channels, dtype=object))
        bad_seg_ids.append(i)
        print(f"    centered windows={len(epochs)}  bad={bad_channels}")

    if not all_w:
        print("  (no centered windows produced — nothing saved)")
        return

    windows = np.concatenate(all_w, axis=0)
    onsets_abs = np.concatenate(all_on, axis=0)
    centers_abs = np.concatenate(all_cen, axis=0)
    seg_id = np.concatenate(all_sid, axis=0)
    seg_type = np.concatenate(all_stype, axis=0)
    labels = np.ones(len(windows), dtype=np.int8)   # positives only

    out_path = out_dir / f"{stem}_windows.npz"
    np.savez_compressed(
        out_path,
        windows=windows,
        labels=labels,                     # all 1 (each window centered on a '*')
        ch_names=np.array(ch_names, dtype=object),
        window_onsets_sec=onsets_abs,      # absolute window start (center - 0.5s)
        center_sec=centers_abs,            # the '*' mark time each window is centered on
        segment_id=seg_id,
        segment_type=seg_type,
        pd_marks_sec=pd_marks,
        bad_channels=np.array(bad_lists, dtype=object),
        bad_channels_segment_id=np.array(bad_seg_ids, dtype=np.int64),
        edf_name=edf_path.name,
        sfreq=sfreq,
        window_sec=WINDOW_SEC,
    )
    print(f"Saved {out_path}  windows={windows.shape}  (POSITIVE only)")
    print("Done.")


if __name__ == "__main__":
    main()