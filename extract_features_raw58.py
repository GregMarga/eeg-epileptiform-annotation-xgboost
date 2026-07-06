"""

=====================================

Same handcrafted-feature band as the doc-12 script (0.5-40 Hz, 80 Hz), same
segment-restricted windowing, same 16-dim features and output format -- but the
static 24h bad-channel report is replaced by PER-SEGMENT PyPREP run from scratch.

Only ONE thing changes vs the doc-12 script: static report -> per-segment PyPREP.
If this recovers ~0.71 balanced accuracy on P58, it proves the false-positive
flood in features_1s_labeled_1_40hz came from the static report leaving some
segment-local bad channels uninterpolated (their broadband power leaked into
NEGATIVE windows and looked like discharges).

Pipeline per segment (band kept EXACTLY as doc 12 by band-passing BEFORE PyPREP):
  crop [start-pad, stop+pad]
    -> rename/map 10-20, pick standard EEG channels, montage
    -> bandpass 0.5-40 Hz               (single filter -> effective HP = 0.5 Hz)
    -> PyPREP detect (deviation/correlation/ransac, NO hf_noise) + interpolate
    -> average reference                (after interp, so bads don't contaminate)
    -> resample 80 Hz
    -> trim padding, keep core
    -> sliding windows fully inside the segment core
  then label from '*' marks + extract handcrafted features.
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
OUT_DIR  = DATA_DIR / "features_1s_labeled_persegment"

# =================================================
# Window / band params
# =================================================
WINDOW_SEC = 1.0
OVERLAP    = 0.75
STRIDE_SEC = WINDOW_SEC * (1.0 - OVERLAP)   # 0.25 s
L_FREQ     = 0.5
H_FREQ     = 40.0
TARGET_SFREQ = 80.0

# Padding on each side of a segment before filtering, trimmed off afterwards.
# 0.5 Hz high-pass FIR is ~6-7 s, so 20 s is a safe margin. PyPREP also runs on
# the padded crop, benefiting from the extra context.
EDGE_PAD_SEC = 20.0

# =================================================
# Labeling / annotations
# =================================================
POSITIVE, NEGATIVE = 1, 0
NOTE_PREFIX = "Note : "
PD_MARK = "*"
PD, NON_PD = "PD", "NON_PD"

OLD_TO_NEW_10_20 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}
STANDARD_1020_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]


# =================================================
# Annotation parsing (segments + marks) -- with .strip() and stable sort
# =================================================

def strip_prefix(desc: str) -> str:
    return desc.removeprefix(NOTE_PREFIX).strip()


def parse_segments_and_marks(raw: mne.io.BaseRaw):
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = [strip_prefix(d) for d in raw.annotations.description]

    order = np.argsort(onsets, kind="stable")
    onsets = onsets[order]
    descs = [descs[i] for i in order]

    segments = []
    marks = []
    open_seg = None  # (seg_type, start_time)

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
        elif d == PD_MARK:
            marks.append(t)

    if open_seg is not None:
        print(f"  [WARN] Unclosed {open_seg[0]} segment at {open_seg[1]:.2f}s — dropped")

    return segments, np.sort(np.asarray(marks, dtype=float))


# =================================================
# Channel prep -- determined ONCE from the full recording for consistency
# =================================================

def resolve_channel_set(raw_header: mne.io.BaseRaw) -> list[str]:
    """Return the fixed list of standard EEG channels this recording provides,
    in canonical order, so every segment yields identical feature dimensions."""
    names = [ch.replace("EEG ", "").strip() for ch in raw_header.ch_names]
    names = [OLD_TO_NEW_10_20.get(n, n) for n in names]
    present = set(names)
    eeg_present = [ch for ch in STANDARD_1020_CHANNELS if ch in present]
    if not eeg_present:
        raise RuntimeError("No standard 10-20 EEG channels found")
    return eeg_present


def prep_channels(raw: mne.io.BaseRaw, keep: list[str]):
    """Rename EEG prefix + old 10-20, pick the fixed set, set montage."""
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())
    raw.rename_channels({o: n for o, n in OLD_TO_NEW_10_20.items() if o in raw.ch_names})
    raw.pick([ch for ch in keep if ch in raw.ch_names])
    raw.reorder_channels(keep)
    raw.set_montage(make_standard_montage("standard_1020"),
                    match_case=False, on_missing="ignore")
    return raw


# =================================================
# Per-segment preprocessing (band-first, then PyPREP)
# =================================================

def detect_bad_channels_pyprep(raw, random_state=42):
    """deviation + correlation + ransac (NO hf_noise). Robust to failures."""
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
    """crop already loaded -> band-pass -> PyPREP interp -> avg ref -> resample."""
    prep_channels(raw_seg, keep_channels)

    # Band FIRST so the effective band is exactly 0.5-40 (matches doc 12).
    raw_seg.filter(l_freq=L_FREQ, h_freq=H_FREQ, method="fir", fir_design="firwin",
                   phase="zero", verbose="ERROR")

    # PyPREP on the band-passed signal at the ORIGINAL rate (better detection).
    bads = detect_bad_channels_pyprep(raw_seg, random_state=random_state)
    bads = [b for b in bads if b in raw_seg.ch_names]
    raw_seg.info["bads"] = bads
    if bads:
        raw_seg.interpolate_bads(reset_bads=True)

    # Average reference AFTER interpolation, then resample.
    raw_seg.set_eeg_reference("average", verbose="ERROR")
    raw_seg.resample(TARGET_SFREQ, npad="auto", verbose="ERROR")

    data = raw_seg.get_data()
    if not np.isfinite(data).all():
        bad = [raw_seg.ch_names[i] for i in range(len(raw_seg.ch_names))
               if not np.isfinite(data[i]).all()]
        raise RuntimeError(f"Non-finite after preprocessing: {bad}")

    return data, float(raw_seg.info["sfreq"]), bads


def extract_segment_windows(raw_full, seg, rec_end, keep_channels, random_state=42):
    seg_start, seg_stop, seg_type = seg
    crop_tmin = max(0.0, seg_start - EDGE_PAD_SEC)
    crop_tmax = min(rec_end, seg_stop + EDGE_PAD_SEC)
    left_pad = seg_start - crop_tmin
    if left_pad < EDGE_PAD_SEC or (crop_tmax - seg_stop) < EDGE_PAD_SEC:
        print("    (near recording boundary — reduced padding)")

    raw_seg = raw_full.copy().crop(tmin=crop_tmin, tmax=crop_tmax).load_data(verbose="ERROR")
    data, sf, bads = preprocess_segment_crop(raw_seg, keep_channels, random_state)

    # Core = the segment itself, padding removed (indices at target rate).
    a = int(round(left_pad * sf))
    b = a + int(round((seg_stop - seg_start) * sf))
    data_core = data[:, a:b]

    win_len = int(round(WINDOW_SEC * sf))
    stride  = int(round(STRIDE_SEC * sf))
    n = data_core.shape[1]

    windows, onsets = [], []
    s = 0
    while s + win_len <= n:
        windows.append(data_core[:, s:s + win_len])
        onsets.append(seg_start + s / sf)   # absolute onset time
        s += stride

    return windows, onsets, sf, bads


# =================================================
# Feature primitives (identical to doc 12)
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


def label_windows(onsets, marks, window_sec):
    onsets = np.asarray(onsets, dtype=float)
    marks = np.sort(np.asarray(marks, dtype=float))
    labels = np.full(len(onsets), NEGATIVE, dtype=np.int8)
    if marks.size == 0:
        return labels
    lo = np.searchsorted(marks, onsets, side="left")
    hi = np.searchsorted(marks, onsets + window_sec, side="right")
    labels[hi > lo] = POSITIVE
    return labels


# =================================================
# Per-file pipeline
# =================================================

def process_file(edf_path: Path, out_path: Path, batch_windows: int = 512):
    raw_full = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    rec_end = float(raw_full.times[-1])

    keep_channels = resolve_channel_set(raw_full)
    print(f"  channels ({len(keep_channels)}): {keep_channels}")

    segments, marks = parse_segments_and_marks(raw_full)
    if not segments:
        print("  (no PD/NON_PD segments — skipped)")
        return

    all_windows, all_onsets, all_types, all_segids = [], [], [], []
    bad_lists, bad_seg_ids = [], []
    sfreq = None

    for seg_id, seg in enumerate(segments):
        seg_start, seg_stop, seg_type = seg
        print(f"  [seg {seg_id:02d}] {seg_type:6s} {seg_start:.1f}-{seg_stop:.1f}s "
              f"({seg_stop - seg_start:.1f}s)")
        try:
            windows, onsets, sf, bads = extract_segment_windows(
                raw_full, seg, rec_end, keep_channels)
        except Exception as e:
            print(f"    ERROR in segment — skipped ({e})")
            continue
        if not windows:
            print("    (shorter than one window — skipped)")
            continue
        sfreq = sf
        all_windows.extend(windows)
        all_onsets.extend(onsets)
        all_types.extend([seg_type] * len(windows))
        all_segids.extend([seg_id] * len(windows))
        bad_lists.append(np.array(bads, dtype=object))
        bad_seg_ids.append(seg_id)
        print(f"    windows={len(windows)}  bad_channels={bads}")

    if not all_windows:
        print("  (no windows produced — skipped)")
        return

    windows = np.stack(all_windows)
    onsets = np.asarray(all_onsets, dtype=np.float64)
    order = np.argsort(onsets, kind="stable")
    windows = windows[order]
    onsets = onsets[order]
    seg_types_arr = np.asarray(all_types, dtype=object)[order]
    seg_ids_arr = np.asarray(all_segids, dtype=np.int64)[order]

    labels = label_windows(onsets, marks, WINDOW_SEC)

    fnames = feature_names(keep_channels)
    X = np.empty((len(windows), len(fnames)), dtype=np.float32)
    for start in range(0, len(windows), batch_windows):
        end = min(start + batch_windows, len(windows))
        for i in range(start, end):
            X[i] = extract_features_for_one_window(windows[i], sfreq)
        print(f"  features [{end}/{len(windows)}]")

    n_pos = int((labels == POSITIVE).sum())
    n_neg = int((labels == NEGATIVE).sum())
    n_pd_seg = sum(1 for _, _, t in segments if t == PD)
    n_nonpd_seg = sum(1 for _, _, t in segments if t == NON_PD)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=labels,
        labels=labels,
        ch_names=np.array(keep_channels, dtype=object),
        feature_names=np.array(fnames, dtype=object),
        window_onsets_sec=onsets,
        segment_type=seg_types_arr,
        segment_id=seg_ids_arr,
        pd_marks_sec=marks,
        sfreq=sfreq,
        window_sec=WINDOW_SEC,
        stride_sec=STRIDE_SEC,
        bad_channels=np.array(bad_lists, dtype=object),
        bad_channels_segment_id=np.array(bad_seg_ids, dtype=np.int64),
        source_edf=edf_path.name,
    )
    print(f"  Saved: {out_path} | X={X.shape} | POSITIVE={n_pos} NEGATIVE={n_neg} | "
          f"marks={len(marks)} | PD_segs={n_pd_seg} NON_PD_segs={n_nonpd_seg}")


def main():
    edf_files = sorted(DATA_DIR.glob("*.edf"))
    if not edf_files:
        raise RuntimeError(f"No *.edf files in {DATA_DIR}")

    print(f"Found {len(edf_files)} EDF files")
    print(f"Band {L_FREQ}-{H_FREQ}Hz, {TARGET_SFREQ:.0f}Hz, window {WINDOW_SEC}s "
          f"stride {STRIDE_SEC}s, pad {EDGE_PAD_SEC}s")
    print("Per-segment PyPREP (deviation/correlation/ransac, no hf_noise).\n")

    for edf_path in edf_files:
        print(f"[{edf_path.stem}]")
        out_path = OUT_DIR / f"{edf_path.stem}_features.npz"
        try:
            process_file(edf_path, out_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()