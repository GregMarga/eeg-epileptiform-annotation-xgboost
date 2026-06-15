import re
import mne
import numpy as np
from pathlib import Path
from mne.channels import make_standard_montage
from pyprep import NoisyChannels


# ----------------------------- config -----------------------------

# Padding (seconds) added on EACH side of a segment BEFORE filtering,
# then trimmed off afterwards. Must comfortably exceed the FIR edge-effect
# length. The 0.1 Hz high-pass is the bottleneck (~30+ s filter), so ~15-16 s
# of each edge is contaminated. 60 s gives a safe margin. PyPREP detection also
# runs on this padded crop, so it benefits from the extra context.
EDGE_PAD_SEC = 60.0

# Exact annotation names (no fuzzy matching). The manual EDF export prefixes
# every description with this exact string; we strip exactly that, then match
# the remainder by exact equality.
NOTE_PREFIX = "Note : "

PD_START, PD_STOP = "PD_START", "PD_STOP"
NON_PD_START, NON_PD_STOP = "NON_PD_START", "NON_PD_STOP"
PD_MARK = "*"

SEGMENT_STARTS = {PD_START: "PD", NON_PD_START: "NON_PD"}
SEGMENT_STOPS = {PD_STOP: "PD", NON_PD_STOP: "NON_PD"}

# Old clinical 10-20 labels -> modern standard_1020 labels used by MNE.
OLD_TO_NEW_1020 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}


# ------------------------- annotation parsing -------------------------

def strip_prefix(desc: str) -> str:
    """Remove the exact 'Note : ' export prefix if present; otherwise unchanged."""
    return desc.removeprefix(NOTE_PREFIX)


def parse_segments(raw):
    """Return (segments, pd_marks) using EXACT name matching.

    segments : list of {type: 'PD'|'NON_PD', start, stop} in seconds
    pd_marks : np.ndarray of '*' onset times in seconds (metadata only)
    """
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = [strip_prefix(d) for d in raw.annotations.description]

    print("Unique descriptions:")
    for d in sorted(set(descs)):
        print(repr(d))

    order = np.argsort(onsets)
    onsets = onsets[order]
    descs = [descs[i] for i in order]

    segments, pd_marks = [], []
    open_start = None  # (type, onset)

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

    return segments, np.array(pd_marks, dtype=float)


# ----------------------- montage / channel names -----------------------

def _clean_name(name: str) -> str:
    """Normalize a channel name so it matches the standard_1020 montage.

    - drop the 'EEG ' prefix
    - strip a trailing reference tag ('-REF', '-Ref', '-LE', '-AVG', ...)
    - map old clinical labels (T3/T4/T5/T6) to modern ones (T7/T8/P7/P8)
    """
    n = name.replace("EEG ", "").strip()
    n = re.sub(r"[-_ ]?(REF|LE|AVG)\Z", "", n, flags=re.IGNORECASE).strip()
    return OLD_TO_NEW_1020.get(n.upper(), n)


def clean_and_set_montage(raw, verbose: bool = True):
    """Clean channel names, set the 10-20 montage, and DROP any channel that
    still has no position (RANSAC and interpolation both require positions)."""
    mapping = {ch: _clean_name(ch) for ch in raw.ch_names}
    # only rename where it actually changes, to avoid spurious collisions
    mapping = {k: v for k, v in mapping.items() if k != v}
    if mapping:
        raw.rename_channels(mapping)

    raw.set_montage(make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")

    no_pos = [ch["ch_name"] for ch in raw.info["chs"]
              if np.isnan(np.asarray(ch["loc"][:3], dtype=float)).any()]
    if no_pos:
        if verbose:
            print(f"  Dropping {len(no_pos)} channel(s) without montage positions "
                  f"(can't be used by RANSAC/interpolation): {no_pos}")
        raw.drop_channels(no_pos)

    return raw


# ----------------------- bad channel detection -----------------------

def detect_and_interpolate_bad_channels_pyprep(raw, random_state=42, verbose=True):
    """PyPREP detection using ONLY deviation (amplitude), correlation
    (+ dropout byproduct) and RANSAC. HF-noise is intentionally NOT used.

    NaN/flat is also skipped to match the requested criteria. If recordings can
    contain dead/NaN channels, add `nc.find_bad_by_nan_flat()` as the first call.
    """
    nc = NoisyChannels(raw, random_state=random_state, do_detrend=False)

    nc.find_bad_by_deviation()                   # amplitude / deviation
    nc.find_bad_by_correlation()                 # correlation (also sets bad_by_dropout)
    nc.find_bad_by_ransac(channel_wise=False)    # RANSAC
    # nc.find_bad_by_hf_noise()  <-- deliberately NOT called

    bads_by_criterion = {
        "deviation":   list(nc.bad_by_deviation),
        "correlation": list(nc.bad_by_correlation),
        "dropout":     list(nc.bad_by_dropout),
        "ransac":      list(nc.bad_by_ransac),
    }
    all_bads = nc.get_bads()

    if verbose:
        print("  PyPREP bad channels (deviation/correlation/ransac, no hf_noise):")
        for crit, bads in bads_by_criterion.items():
            if bads:
                print(f"    {crit:12s}: {bads}")
        print(f"  Total unique bad channels: {all_bads}")

    raw.info["bads"] = all_bads
    if len(all_bads) > 0:
        raw.interpolate_bads(reset_bads=True)

    return all_bads, bads_by_criterion


# ----------------------- windowing & preprocessing -----------------------

def create_sliding_windows(data, sfreq, window_sec=1.0, stride_sec=0.25):
    window_samples = int(round(window_sec * sfreq))
    stride_samples = int(round(stride_sec * sfreq))
    n_samples = data.shape[1]
    starts = np.arange(0, n_samples - window_samples + 1, stride_samples)
    if len(starts) == 0:
        return np.empty((0, data.shape[0], window_samples), dtype=data.dtype), starts
    epochs = np.stack([data[:, s:s + window_samples] for s in starts], axis=0)
    return epochs, starts


def preprocess_for_labram(raw, l_freq=0.1, h_freq=75.0, notch_freq=50.0,
                          target_sfreq=200.0, prep_hp_freq=1.0,
                          use_pyprep=True, random_state=42):
    """LaBraM-matching pipeline with PyPREP bad-channel handling."""
    raw.pick("eeg")
    clean_and_set_montage(raw)

    # Light high-pass (no low-pass yet) keeps broadband content for detection
    raw.filter(l_freq=prep_hp_freq, h_freq=None, method="fir",
               fir_design="firwin", phase="zero", verbose="ERROR")
    raw.notch_filter(freqs=[notch_freq], verbose="ERROR")

    if use_pyprep:
        bad_channels, bads_by_criterion = detect_and_interpolate_bad_channels_pyprep(
            raw, random_state=random_state)
    else:
        bad_channels, bads_by_criterion = [], {}

    # Final bandpass matching LaBraM pre-training
    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir",
               fir_design="firwin", phase="zero", verbose="ERROR")

    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    raw.resample(target_sfreq, npad="auto", verbose="ERROR")

    data_uv = raw.get_data() * 1e6
    return raw, data_uv, bad_channels, bads_by_criterion


def extract_segment_windows(raw_full, seg, rec_end_sec,
                            edge_pad_sec=EDGE_PAD_SEC,
                            window_sec=1.0, stride_sec=0.25,
                            target_sfreq=200.0, use_pyprep=True,
                            random_state=42, **prep_kwargs):
    """Crop ONE segment (with padding), preprocess (incl. PyPREP), trim, window."""
    seg_start, seg_stop = seg["start"], seg["stop"]

    crop_tmin = max(0.0, seg_start - edge_pad_sec)
    crop_tmax = min(rec_end_sec, seg_stop + edge_pad_sec)
    left_pad = seg_start - crop_tmin
    if left_pad < edge_pad_sec or (crop_tmax - seg_stop) < edge_pad_sec:
        print("    (segment near recording boundary — reduced padding, "
              "minor edge effects possible)")

    raw_seg = raw_full.copy().crop(tmin=crop_tmin, tmax=crop_tmax).load_data(verbose="ERROR")
    raw_seg, data_uv, bad_channels, _ = preprocess_for_labram(
        raw_seg, target_sfreq=target_sfreq, use_pyprep=use_pyprep,
        random_state=random_state, **prep_kwargs)
    sf = float(raw_seg.info["sfreq"])

    a = int(round(left_pad * sf))
    b = a + int(round((seg_stop - seg_start) * sf))
    data_core = data_uv[:, a:b]

    epochs, onsets_in_core = create_sliding_windows(data_core, sf, window_sec, stride_sec)
    onsets_abs = seg_start + onsets_in_core / sf

    return epochs.astype(np.float32), onsets_abs, list(raw_seg.ch_names), sf, list(bad_channels)


# ----------------------------- main -----------------------------

def main():
    data_dir = Path("../../../data/evaluation_recordings")
    out_dir = data_dir / "labram_sliding_windows_1s_75overlap"
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

    # Diagnostic: if no '*' found, show exactly what descriptions DO exist
    # (use repr so trailing spaces / odd glyphs are visible) instead of guessing.
    if len(pd_marks) == 0:
        uniq = sorted(set(raw_full.annotations.description))
        print("  ! No '*' marks parsed. Unique raw annotation descriptions present:")
        for u in uniq:
            print(f"      {u!r}")

    all_w, all_on, all_sid, all_stype = [], [], [], []
    bad_lists, bad_seg_ids = [], []
    ch_names, sfreq = None, None

    for i, seg in enumerate(segments):
        print(f"[seg {i:02d}] {seg['type']:6s} {seg['start']:.1f}-{seg['stop']:.1f}s "
              f"({seg['stop'] - seg['start']:.1f}s)")
        epochs, onsets_abs, ch_names, sfreq, bad_channels = extract_segment_windows(
            raw_full, seg, rec_end,
            edge_pad_sec=EDGE_PAD_SEC, window_sec=1.0, stride_sec=0.25,
            target_sfreq=200.0, use_pyprep=True, random_state=42,
            l_freq=0.1, h_freq=75.0, notch_freq=50.0, prep_hp_freq=1.0,
        )
        if len(epochs) == 0:
            print("    (segment shorter than one window — skipped)")
            continue
        all_w.append(epochs)
        all_on.append(onsets_abs)
        all_sid.append(np.full(len(epochs), i, dtype=np.int64))
        all_stype.append(np.array([seg["type"]] * len(epochs), dtype=object))
        bad_lists.append(np.array(bad_channels, dtype=object))
        bad_seg_ids.append(i)

    windows = np.concatenate(all_w, axis=0)
    onsets_abs = np.concatenate(all_on, axis=0)
    seg_id = np.concatenate(all_sid, axis=0)
    seg_type = np.concatenate(all_stype, axis=0)

    out_path = out_dir / f"{stem}_windows.npz"
    np.savez_compressed(
        out_path,
        windows=windows,
        ch_names=np.array(ch_names, dtype=object),
        window_onsets_sec=onsets_abs,
        segment_id=seg_id,
        segment_type=seg_type,
        pd_marks_sec=pd_marks,
        bad_channels=np.array(bad_lists, dtype=object),
        bad_channels_segment_id=np.array(bad_seg_ids, dtype=np.int64),
        edf_name=edf_path.name,
        sfreq=sfreq,
        window_sec=1.0,
        stride_sec=0.25,
    )
    print(f"Saved {out_path}  windows={windows.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
