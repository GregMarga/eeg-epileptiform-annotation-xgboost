import re
import ast
from pathlib import Path

import mne
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from mne.channels import make_standard_montage


# =================================================
# Paths
# =================================================
DATA_DIR    = Path("../../../data/evaluation_recordings")
REPORT_PATH = Path("../../../data/channel_detection_details.txt")
OUT_DIR     = DATA_DIR / "features_1s_labeled_1_40hz"

# =================================================
# Window params
# =================================================
WINDOW_SEC  = 1.0
OVERLAP     = 0.75
STRIDE_SEC  = WINDOW_SEC * (1.0 - OVERLAP)  # 0.25 s

# =================================================
# Labeling
# =================================================
POSITIVE, NEGATIVE = 1, 0
NOTE_PREFIX = "Note : "
PD_MARK = "*"
PD, NON_PD = "PD", "NON_PD"


# =================================================
# Bad-channel report parsing (from the preprocessing script)
# =================================================
INTERPOLATE_CATEGORIES = ("deviation", "correlation", "ransac")


def _parse_channel_list(value: str) -> list[str]:
    cleaned = re.sub(r"np\.str_\(([^)]+)\)", r"\1", value.strip())
    try:
        return [str(c) for c in ast.literal_eval(cleaned)]
    except (ValueError, SyntaxError):
        return []


def parse_channel_detection_report(txt_path: Path) -> dict[str, dict[str, list[str]]]:
    text = Path(txt_path).read_text(encoding="utf-8", errors="replace")
    edf_header_re = re.compile(r"^\[\d+/\d+\]\s+(.+?)\.edf\s*$", re.MULTILINE)
    matches = list(edf_header_re.finditer(text))

    report: dict[str, dict[str, list[str]]] = {}
    for i, m in enumerate(matches):
        basename = m.group(1).strip()
        block_start = m.end()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[block_start:block_end]

        cats: dict[str, list[str]] = {
            "deviation": [], "hf_noise": [], "correlation": [], "ransac": [],
        }
        for cat in cats.keys():
            cat_re = re.compile(rf"^\s*{cat}\s*:\s*(\[.*?\])\s*$", re.MULTILINE)
            cm = cat_re.search(block)
            if cm:
                cats[cat] = _parse_channel_list(cm.group(1))
        report[basename] = cats
    return report


# Old 10-20 nomenclature -> current standard names.
OLD_TO_NEW_10_20 = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Real EEG channels we expect to keep; everything else (ECG, "Unspec ..."
# monitoring channels, etc.) is dropped before filtering/interpolation.
STANDARD_1020_CHANNELS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T7", "C3", "Cz", "C4", "T8",
    "P7", "P3", "Pz", "P4", "P8",
    "O1", "O2",
]


def channels_to_interpolate(report_entry: dict[str, list[str]],
                             rename_map: dict[str, str] | None = None) -> list[str]:
    to_interp: set[str] = set()
    for cat in INTERPOLATE_CATEGORIES:
        to_interp.update(report_entry.get(cat, []))
    if rename_map:
        to_interp = {rename_map.get(ch, ch) for ch in to_interp}
    return sorted(to_interp)


def mark_and_interpolate_bad_channels_from_report(
        raw: mne.io.BaseRaw,
        edf_basename: str,
        report: dict[str, dict[str, list[str]]],
) -> list[str]:
    if edf_basename not in report:
        print(f"  [WARN] No report entry for {edf_basename} — skipping interpolation")
        return []

    entry = report[edf_basename]
    # report was generated with old channel names in some recordings (T3/T4/T5/T6),
    # so map those to the renamed channels actually present in raw
    bad_channels = channels_to_interpolate(entry, rename_map=OLD_TO_NEW_10_20)
    bad_channels = [ch for ch in bad_channels if ch in raw.ch_names]

    if not bad_channels:
        return []

    raw.info["bads"] = bad_channels
    raw.interpolate_bads(reset_bads=True)
    return bad_channels


# =================================================
# Preprocessing (filter, resample, bad-channel interp, avg reference)
# Returns the FULL continuous recording — no epoching here, since we
# need the whole timeline to slide windows across PD/NON_PD segments.
# =================================================

def preprocess_raw(edf_path: Path, report: dict[str, dict[str, list[str]]],
                    l_freq: float = 0.5, h_freq: float = 40.0):
    edf_basename = edf_path.stem

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.rename_channels(lambda ch: ch.replace("EEG ", "").strip())
    raw.rename_channels({old: new for old, new in OLD_TO_NEW_10_20.items() if old in raw.ch_names})

    # Drop everything that isn't a real 10-20 EEG channel (ECG, "Unspec ..."
    # monitoring channels, etc.) — these have no montage position and/or
    # invalid values that break spline interpolation.
    eeg_present = [ch for ch in STANDARD_1020_CHANNELS if ch in raw.ch_names]
    dropped = sorted(set(raw.ch_names) - set(eeg_present))
    if dropped:
        print(f"  Dropping non-EEG / unrecognized channels: {dropped}")
    if not eeg_present:
        raise RuntimeError(f"No standard 10-20 EEG channels found in {edf_path.name}")
    raw.pick(eeg_present)

    montage = make_standard_montage("standard_1020")
    raw.set_montage(montage, match_case=False, on_missing="ignore")

    raw.filter(l_freq=l_freq, h_freq=h_freq, method="fir", fir_design="firwin",
               phase="zero", verbose="ERROR")
    raw.resample(80, verbose="ERROR")

    bad_chs = mark_and_interpolate_bad_channels_from_report(raw, edf_basename, report)

    # average reference AFTER interpolation, same as before
    raw.set_eeg_reference("average", verbose="ERROR")

    data = raw.get_data()
    if not np.isfinite(data).all():
        bad_data_chs = [raw.ch_names[i] for i in range(len(raw.ch_names))
                         if not np.isfinite(data[i]).all()]
        raise RuntimeError(f"Non-finite values remain after preprocessing in: {bad_data_chs}")

    return raw, bad_chs


# =================================================
# Segment (PD_START/PD_STOP, NON_PD_START/NON_PD_STOP) + mark parsing
# =================================================

def strip_prefix(desc: str) -> str:
    return desc.removeprefix(NOTE_PREFIX).strip()


def parse_segments_and_marks(raw: mne.io.BaseRaw):
    """
    Returns:
        segments: list of (start_sec, stop_sec, seg_type) for PD / NON_PD blocks
        marks:    sorted np.ndarray of '*' discharge onset times (sec)
    START/STOP pairs are matched sequentially in chronological order;
    an unmatched START (no following STOP of the same type) is dropped.
    """
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = [strip_prefix(d) for d in raw.annotations.description]

    order = np.argsort(onsets)
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
        print(f"  [WARN] Unclosed {open_seg[0]} segment starting at {open_seg[1]:.2f}s — dropped")

    return segments, np.sort(np.asarray(marks, dtype=float))


def generate_windows_for_segment(data, sfreq, seg_start, seg_stop, seg_type, seg_id):
    """
    Sliding windows of WINDOW_SEC with STRIDE_SEC stride, kept only if
    FULLY contained inside [seg_start, seg_stop] (strict containment,
    no partial overlap allowed at segment edges).
    """
    win_len = int(round(WINDOW_SEC * sfreq))
    stride = int(round(STRIDE_SEC * sfreq))
    n_samples = data.shape[1]

    start_sample = int(round(seg_start * sfreq))
    stop_sample = int(round(seg_stop * sfreq))

    windows, onsets, seg_types, seg_ids = [], [], [], []
    s = start_sample
    while s + win_len <= stop_sample and s + win_len <= n_samples:
        windows.append(data[:, s:s + win_len])
        onsets.append(s / sfreq)
        seg_types.append(seg_type)
        seg_ids.append(seg_id)
        s += stride

    return windows, onsets, seg_types, seg_ids


def label_windows(onsets, marks, window_sec):
    """POSITIVE if a '*' mark falls inside [onset, onset + window_sec], else NEGATIVE."""
    onsets = np.asarray(onsets, dtype=float)
    marks = np.sort(np.asarray(marks, dtype=float))

    labels = np.full(len(onsets), NEGATIVE, dtype=np.int8)
    if marks.size == 0:
        return labels

    starts = onsets
    ends = onsets + window_sec
    lo = np.searchsorted(marks, starts, side="left")
    hi = np.searchsorted(marks, ends, side="right")
    labels[hi > lo] = POSITIVE
    return labels


# =================================================
# Feature primitives (unchanged from the feature-extraction script)
# =================================================

def zero_crossings(x: np.ndarray) -> int:
    return int(np.sum((x[:-1] * x[1:]) < 0))


def count_local_extrema(x: np.ndarray):
    dx = np.diff(x)
    maxima = np.sum((dx[:-1] > 0) & (dx[1:] < 0))
    minima = np.sum((dx[:-1] < 0) & (dx[1:] > 0))
    return int(maxima), int(minima)


def rms_amplitude(x: np.ndarray) -> float:
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
    bands = {
        "delta": (1.0, 3.0), "theta": (4.0, 8.0),
        "alpha": (9.0, 13.0), "beta": (14.0, 20.0),
    }
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
    n_ch = window_2d.shape[0]
    feats = []
    for ch in range(n_ch):
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
    per_ch_freq = [
        "total_power_1_40", "peak_freq_1_40",
        "mean_band_delta", "mean_band_theta", "mean_band_alpha", "mean_band_beta",
        "norm_band_delta", "norm_band_theta", "norm_band_alpha", "norm_band_beta",
    ]
    names = []
    for ch in ch_names:
        for f in per_ch_time:
            names.append(f"{ch}_{f}")
        for f in per_ch_freq:
            names.append(f"{ch}_{f}")
    return names


# =================================================
# Per-file pipeline: preprocess -> segment-restricted windows -> features
# =================================================

def process_file(edf_path: Path, report: dict, out_path: Path, batch_windows: int = 512):
    raw, bad_chs = preprocess_raw(edf_path, report)
    data = raw.get_data()
    ch_names = raw.ch_names
    sfreq = float(raw.info["sfreq"])

    segments, marks = parse_segments_and_marks(raw)
    if not segments:
        print("  (no PD_START/PD_STOP or NON_PD_START/NON_PD_STOP segments — skipped)")
        return

    all_windows, all_onsets, all_types, all_segids = [], [], [], []
    for seg_id, (seg_start, seg_stop, seg_type) in enumerate(segments):
        w, o, t, sid = generate_windows_for_segment(data, sfreq, seg_start, seg_stop, seg_type, seg_id)
        all_windows.extend(w)
        all_onsets.extend(o)
        all_types.extend(t)
        all_segids.extend(sid)

    if not all_windows:
        print("  (segments found but no full-length windows fit inside them — skipped)")
        return

    windows = np.stack(all_windows)
    onsets = np.asarray(all_onsets, dtype=np.float64)

    order = np.argsort(onsets)
    windows = windows[order]
    onsets = onsets[order]
    seg_types_arr = np.asarray(all_types, dtype=object)[order]
    seg_ids_arr = np.asarray(all_segids, dtype=np.int64)[order]

    labels = label_windows(onsets, marks, WINDOW_SEC)

    n_windows = windows.shape[0]
    fnames = feature_names(ch_names)
    X = np.empty((n_windows, len(fnames)), dtype=np.float32)

    for start in range(0, n_windows, batch_windows):
        end = min(start + batch_windows, n_windows)
        for i in range(start, end):
            X[i] = extract_features_for_one_window(windows[i], sfreq)
        print(f"  features [{end}/{n_windows}]")

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
        ch_names=np.array(ch_names, dtype=object),
        feature_names=np.array(fnames, dtype=object),
        window_onsets_sec=onsets,
        segment_type=seg_types_arr,
        segment_id=seg_ids_arr,
        pd_marks_sec=marks,
        sfreq=sfreq,
        window_sec=WINDOW_SEC,
        stride_sec=STRIDE_SEC,
        bad_channels=np.array(bad_chs, dtype=object),
        source_edf=edf_path.name,
    )
    print(f"  Saved: {out_path} | X={X.shape} | POSITIVE={n_pos} NEGATIVE={n_neg} | "
          f"marks={len(marks)} | PD_segs={n_pd_seg} NON_PD_segs={n_nonpd_seg}")


# =================================================
# Main
# =================================================

def main():
    report = parse_channel_detection_report(REPORT_PATH)

    edf_files = sorted(DATA_DIR.glob("*.edf"))
    if not edf_files:
        raise RuntimeError(f"No *.edf files in {DATA_DIR}")

    print(f"Found {len(edf_files)} EDF files")
    print(f"Window: {WINDOW_SEC}s, stride: {STRIDE_SEC}s (overlap={OVERLAP*100:.0f}%)")
    print("Windows are kept only if fully inside a PD or NON_PD segment.\n")

    for edf_path in edf_files:
        print(f"[{edf_path.stem}]")
        out_path = OUT_DIR / f"{edf_path.stem}_features.npz"
        try:
            process_file(edf_path, report, out_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()