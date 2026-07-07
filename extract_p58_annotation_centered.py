"""
extract_annot_centered_persegment.py
====================================
Annotation-CENTERED variant of extract_features_persegment_pyprep.py.

Same per-segment preprocessing (bandpass 0.5-40 Hz, per-segment PyPREP,
average reference, resample 80 Hz) as the sliding-window script, so pool and
test differ ONLY in windowing -- NOT in preprocessing.

Instead of dense sliding windows, this extracts ONE 1s window CENTERED on each
annotation:
  '*'  -> POSITIVE window
  '-'  -> NEGATIVE window
The window is taken from the full padded, preprocessed crop by absolute time, so
annotations near a segment edge still get valid +/-0.5 s of context from the pad.
The label comes directly from the annotation type (not from mark-in-window).

Output goes to the SAME folder as the sliding features, but with a distinct
filename ('{stem}_annot_centered.npz') that does NOT end in '_features.npz', so
a test loader globbing '*_features.npz' will not pick these up. segment_type is
stored per window so the active-learning pool can keep NON_PD negatives only.

This is the POOL dataset for the active-learning experiment (pool = annotation-
centered; test = sliding).
"""

import re
from pathlib import Path

import mne
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from mne.channels import make_standard_montage
from pyprep import NoisyChannels


# =================================================
# Paths
# =================================================
DATA_DIR = Path("../../../data/evaluation_recordings")
OUT_DIR  = DATA_DIR / "features_1s_labeled_persegment"   # SAME folder as sliding

# =================================================
# Window / band params
# =================================================
WINDOW_SEC = 1.0                 # centered window length (+/- 0.5 s around mark)
L_FREQ     = 0.5
H_FREQ     = 40.0
TARGET_SFREQ = 80.0
EDGE_PAD_SEC = 20.0

# =================================================
# Labeling / annotations
# =================================================
POSITIVE, NEGATIVE = 1, 0
NOTE_PREFIX = "Note : "
POS_MARK = "*"
NEG_MARK = "-"
PD, NON_PD = "PD", "NON_PD"

OLD_TO_NEW_10_20 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
STANDARD_1020_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]


# =================================================
# Annotation parsing: segments + centered events ('*'/'-')
# =================================================

def strip_prefix(desc: str) -> str:
    return desc.removeprefix(NOTE_PREFIX).strip()


def parse_segments_and_events(raw: mne.io.BaseRaw):
    """Return (segments, events).

    segments : list of (start_sec, stop_sec, seg_type)
    events   : list of (onset_sec, kind) with kind in {"POS", "NEG"} for '*'/'-'
    """
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = [strip_prefix(d) for d in raw.annotations.description]

    order = np.argsort(onsets, kind="stable")
    onsets = onsets[order]
    descs = [descs[i] for i in order]

    segments, events = [], []
    open_seg = None

    for t, d in zip(onsets, descs):
        if d == f"{PD}_START":
            open_seg = (PD, t)
        elif d == f"{PD}_STOP":
            if open_seg is not None and open_seg[0] == PD:
                segments.append((open_seg[1], t, PD))
            open_seg = None
        elif d == f"{NON_PD}_START":
            open_seg = (NON_PD, t)
        elif d == f"{NON_PD}_STOP":
            if open_seg is not None and open_seg[0] == NON_PD:
                segments.append((open_seg[1], t, NON_PD))
            open_seg = None
        elif d == POS_MARK:
            events.append((t, "POS"))
        elif d == NEG_MARK:
            events.append((t, "NEG"))

    if open_seg is not None:
        print(f"  [WARN] Unclosed {open_seg[0]} segment at {open_seg[1]:.2f}s — dropped")

    return segments, events


# =================================================
# Channel prep (identical to the sliding script)
# =================================================

def resolve_channel_set(raw_header: mne.io.BaseRaw) -> list[str]:
    names = [ch.replace("EEG ", "").strip() for ch in raw_header.ch_names]
    names = [OLD_TO_NEW_10_20.get(n, n) for n in names]
    present = set(names)
    eeg_present = [ch for ch in STANDARD_1020_CHANNELS if ch in present]
    if not eeg_present:
        raise RuntimeError("No standard 10-20 EEG channels found")
    return eeg_present


def prep_channels(raw: mne.io.BaseRaw, keep: list[str]):
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())
    raw.rename_channels({o: n for o, n in OLD_TO_NEW_10_20.items() if o in raw.ch_names})
    raw.pick([ch for ch in keep if ch in raw.ch_names])
    raw.reorder_channels(keep)
    raw.set_montage(make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")
    return raw


# =================================================
# Per-segment preprocessing (identical to the sliding script)
# =================================================

def detect_bad_channels_pyprep(raw, random_state=42):
    try:
        nc = NoisyChannels(raw, random_state=random_state, do_detrend=False)
        nc.find_bad_by_deviation()
        nc.find_bad_by_correlation()
        nc.find_bad_by_ransac(channel_wise=False)
        return nc.get_bads()
    except Exception as e:
        print(f"    [WARN] PyPREP failed ({e}) — no interpolation for this segment")
        return []


def preprocess_segment_crop(raw_seg, keep_channels, random_state=42):
    prep_channels(raw_seg, keep_channels)
    raw_seg.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="fir", fir_design="firwin",
                   phase="zero", verbose="ERROR")
    bads = detect_bad_channels_pyprep(raw_seg, random_state=random_state)
    bads = [b for b in bads if b in raw_seg.ch_names]
    raw_seg.info["bads"] = bads
    if bads:
        raw_seg.interpolate_bads(reset_bads=True)
    raw_seg.set_eeg_reference("average", verbose="ERROR")
    raw_seg.resample(TARGET_SFREQ, npad="auto", verbose="ERROR")

    data = raw_seg.get_data()
    if not np.isfinite(data).all():
        bad = [raw_seg.ch_names[i] for i in range(len(raw_seg.ch_names))
               if not np.isfinite(data[i]).all()]
        raise RuntimeError(f"Non-finite after preprocessing: {bad}")

    return data, float(raw_seg.info["sfreq"]), bads


def extract_segment_annot_windows(raw_full, seg, rec_end, keep_channels,
                                  events, random_state=42):
    """Extract one centered window per '*'/'-' event that falls inside the segment.

    Windows are cut from the FULL padded preprocessed crop by absolute time, so
    marks near the segment edge still get +/-0.5 s of valid (padded) context.
    """
    seg_start, seg_stop, seg_type = seg
    seg_events = [(t, kind) for (t, kind) in events if seg_start <= t < seg_stop]
    if not seg_events:
        return [], [], [], None, []

    crop_tmin = max(0.0, seg_start - EDGE_PAD_SEC)
    crop_tmax = min(rec_end, seg_stop + EDGE_PAD_SEC)
    if (seg_start - crop_tmin) < EDGE_PAD_SEC or (crop_tmax - seg_stop) < EDGE_PAD_SEC:
        print("    (near recording boundary — reduced padding)")

    raw_seg = raw_full.copy().crop(tmin=crop_tmin, tmax=crop_tmax).load_data(verbose="ERROR")
    data, sf, bads = preprocess_segment_crop(raw_seg, keep_channels, random_state)

    margin = int(round(WINDOW_SEC / 2 * sf))    # 0.5 s each side
    n = data.shape[1]

    windows, labels, centers = [], [], []
    for t, kind in seg_events:
        center = int(round((t - crop_tmin) * sf))
        a, b = center - margin, center + margin
        if a < 0 or b > n:
            print(f"    (event at {t:.2f}s too close to crop edge — skipped)")
            continue
        windows.append(data[:, a:b])
        labels.append(POSITIVE if kind == "POS" else NEGATIVE)
        centers.append(t)

    return windows, labels, centers, sf, bads


# =================================================
# Feature primitives (identical to the sliding script)
# =================================================

def zero_crossings(x):
    return int(np.sum((x[:-1] * x[1:]) < 0))


def count_local_extrema(x):
    dx = np.diff(x)
    maxima = np.sum((dx[:-1] > 0) & (dx[1:] < 0))
    minima = np.sum((dx[:-1] < 0) & (dx[1:] > 0))
    return int(maxima), int(minima)


def rms_amplitude(x):
    return float(np.sqrt(np.mean(x * x)))


def compute_welch_psd_1d(x, fs, nperseg=None, noverlap=None):
    x = np.asarray(x, dtype=np.float64)
    if nperseg is None:
        nperseg = min(256, x.shape[-1])
    if noverlap is None:
        noverlap = nperseg // 2
    f, psd = welch(x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                   detrend="constant", scaling="density")
    return f, psd


def _band_mask(f, fmin, fmax):
    if fmin >= fmax:
        raise ValueError(f"Invalid band [{fmin}, {fmax}]")
    return (f >= fmin) & (f <= fmax)


def bandpower_trapz(f, psd, fmin, fmax):
    m = _band_mask(f, fmin, fmax)
    if not np.any(m):
        return 0.0
    f_band, psd_band = f[m], psd[m]
    if len(f_band) == 1:
        df = f[1] - f[0] if len(f) > 1 else 0.0
        return float(psd_band[0] * df)
    return float(np.trapezoid(psd_band, f_band))


def mean_psd_in_band(f, psd, fmin, fmax):
    m = _band_mask(f, fmin, fmax)
    if not np.any(m):
        return 0.0
    return float(np.mean(psd[m]))


def peak_frequency_in_band(f, psd, fmin, fmax):
    m = _band_mask(f, fmin, fmax)
    if not np.any(m):
        return float("nan")
    idx = int(np.argmax(psd[m]))
    return float(f[m][idx])


def freq_features_1d(x, fs, *, total_range=(1.0, 40.0), nperseg=None, noverlap=None):
    bands = {"delta": (1.0, 3.0), "theta": (4.0, 8.0),
             "alpha": (9.0, 13.0), "beta": (14.0, 20.0)}
    f, psd = compute_welch_psd_1d(x, fs=fs, nperseg=nperseg, noverlap=noverlap)
    tr0, tr1 = total_range
    total_p = bandpower_trapz(f, psd, tr0, tr1)
    peak_f = peak_frequency_in_band(f, psd, tr0, tr1)
    mean_feats, norm_feats = [], []
    eps = 1e-12
    for name in ("delta", "theta", "alpha", "beta"):
        fmin, fmax = bands[name]
        bp = bandpower_trapz(f, psd, fmin, fmax)
        mp = mean_psd_in_band(f, psd, fmin, fmax)
        mean_feats.append(mp)
        norm_feats.append(bp / (total_p + eps))
    return [total_p, peak_f, *mean_feats, *norm_feats]


def extract_features_for_one_window(window_2d, fs):
    feats = []
    for ch in range(window_2d.shape[0]):
        x = window_2d[ch].astype(float)
        x = x - np.mean(x)
        zc = zero_crossings(x)
        mx, mn = count_local_extrema(x)
        rms = rms_amplitude(x)
        sk = float(skew(x, bias=False))
        ku = float(kurtosis(x, fisher=True, bias=False))
        ffeats = freq_features_1d(x, fs)
        feats.extend([zc, mx, mn, rms, sk, ku, *ffeats])
    return np.asarray(feats, dtype=np.float32)


def feature_names(ch_names):
    per_ch_time = ["zero_cross", "maxima", "minima", "rms", "skew", "kurt_excess"]
    per_ch_freq = ["total_power_1_40", "peak_freq_1_40",
                   "mean_band_delta", "mean_band_theta", "mean_band_alpha", "mean_band_beta",
                   "norm_band_delta", "norm_band_theta", "norm_band_alpha", "norm_band_beta"]
    names = []
    for ch in ch_names:
        for f in per_ch_time:
            names.append(f"{ch}_{f}")
        for f in per_ch_freq:
            names.append(f"{ch}_{f}")
    return names


# =================================================
# Per-file pipeline
# =================================================

def process_file(edf_path: Path, out_path: Path, batch_windows: int = 512):
    raw_full = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    rec_end = float(raw_full.times[-1])

    keep_channels = resolve_channel_set(raw_full)
    print(f"  channels ({len(keep_channels)}): {keep_channels}")

    segments, events = parse_segments_and_events(raw_full)
    if not segments:
        print("  (no PD/NON_PD segments — skipped)")
        return
    if not events:
        print("  (no '*'/'-' annotation events — skipped)")
        return

    all_windows, all_labels, all_types, all_segids, all_centers = [], [], [], [], []
    bad_lists, bad_seg_ids = [], []
    sfreq = None

    for seg_id, seg in enumerate(segments):
        seg_start, seg_stop, seg_type = seg
        try:
            windows, labels, centers, sf, bads = extract_segment_annot_windows(
                raw_full, seg, rec_end, keep_channels, events)
        except Exception as e:
            print(f"  [seg {seg_id:02d}] ERROR — skipped ({e})")
            continue
        if not windows:
            continue
        sfreq = sf
        all_windows.extend(windows)
        all_labels.extend(labels)
        all_types.extend([seg_type] * len(windows))
        all_segids.extend([seg_id] * len(windows))
        all_centers.extend(centers)
        bad_lists.append(np.array(bads, dtype=object))
        bad_seg_ids.append(seg_id)
        n_pos = sum(1 for l in labels if l == POSITIVE)
        n_neg = sum(1 for l in labels if l == NEGATIVE)
        print(f"  [seg {seg_id:02d}] {seg_type:6s} {seg_start:.1f}-{seg_stop:.1f}s | "
              f"events pos={n_pos} neg={n_neg} | bad={bads}")

    if not all_windows:
        print("  (no centered windows produced — skipped)")
        return

    windows = np.stack(all_windows)
    labels = np.asarray(all_labels, dtype=np.int8)
    centers = np.asarray(all_centers, dtype=np.float64)

    order = np.argsort(centers, kind="stable")
    windows = windows[order]
    labels = labels[order]
    centers = centers[order]
    seg_types_arr = np.asarray(all_types, dtype=object)[order]
    seg_ids_arr = np.asarray(all_segids, dtype=np.int64)[order]

    fnames = feature_names(keep_channels)
    X = np.empty((len(windows), len(fnames)), dtype=np.float32)
    for start in range(0, len(windows), batch_windows):
        end = min(start + batch_windows, len(windows))
        for i in range(start, end):
            X[i] = extract_features_for_one_window(windows[i], sfreq)
        print(f"  features [{end}/{len(windows)}]")

    n_pos = int((labels == POSITIVE).sum())
    n_neg = int((labels == NEGATIVE).sum())
    n_nonpd_neg = int(((labels == NEGATIVE) & (seg_types_arr == NON_PD)).sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=labels,
        labels=labels,
        ch_names=np.array(keep_channels, dtype=object),
        feature_names=np.array(fnames, dtype=object),
        center_sec=centers,
        segment_type=seg_types_arr,
        segment_id=seg_ids_arr,
        sfreq=sfreq,
        window_sec=WINDOW_SEC,
        bad_channels=np.array(bad_lists, dtype=object),
        bad_channels_segment_id=np.array(bad_seg_ids, dtype=np.int64),
        source_edf=edf_path.name,
    )
    print(f"  Saved: {out_path} | X={X.shape} | POSITIVE={n_pos} NEGATIVE={n_neg} "
          f"(NON_PD negs={n_nonpd_neg})")


def main():
    edf_files = sorted(DATA_DIR.glob("*.edf"))
    if not edf_files:
        raise RuntimeError(f"No *.edf files in {DATA_DIR}")

    print(f"Found {len(edf_files)} EDF files")
    print(f"Annotation-centered windows | band {L_FREQ}-{H_FREQ}Hz, {TARGET_SFREQ:.0f}Hz, "
          f"window {WINDOW_SEC}s, pad {EDGE_PAD_SEC}s")
    print("Per-segment PyPREP (deviation/correlation/ransac, no hf_noise).")
    print("'*' -> positive, '-' -> negative; label from annotation type.\n")

    for edf_path in edf_files:
        print(f"[{edf_path.stem}]")
        out_path = OUT_DIR / f"{edf_path.stem}_annot_centered.npz"
        try:
            process_file(edf_path, out_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()