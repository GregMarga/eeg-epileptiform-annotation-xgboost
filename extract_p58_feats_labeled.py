from pathlib import Path
import argparse
import numpy as np
import mne
from scipy.stats import skew, kurtosis
from scipy.signal import welch


# =================================================
# Paths
# =================================================
DATA_DIR    = Path("../../../data/evaluation_recordings")
WINDOWS_DIR = DATA_DIR / "labram_sliding_windows_1s_75overlap"   # *_windows.npz live here
OUT_DIR     = DATA_DIR / "features_1s_labeled"                   # outputs go here

# =================================================
# Labeling parameters (seconds) -- same rule as the embedding labeling script
# =================================================
M_SEC = 0.10   # edge margin: a '*' must sit at least m inside the window to be POSITIVE
G_SEC = 0.30   # guard band: a no-mark window is NEGATIVE only if nearest mark is >= g away

POSITIVE, NEGATIVE, IGNORE = 1, 0, -1

NOTE_PREFIX = "Note : "
PD_MARK = "*"
PD, NON_PD = "PD", "NON_PD"


# =================================================
# Feature primitives (1D) - time domain
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


# =================================================
# Feature primitives (1D) - frequency domain (NO HF)
# =================================================

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
        "delta": (1.0, 3.0),
        "theta": (4.0, 8.0),
        "alpha": (9.0, 13.0),
        "beta":  (14.0, 20.0),
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


# =================================================
# Per-window feature extraction
# =================================================

def extract_features_for_one_window(window_2d, fs):
    """window_2d: (n_channels, n_samples) -> (n_channels * 16,) feature vector."""
    n_ch = window_2d.shape[0]
    feats = []
    for ch in range(n_ch):
        x = window_2d[ch].astype(float)
        x = x - np.mean(x)  # center

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
# Labeling from EDF '*' marks (same rule as the embedding labeler)
# =================================================

def strip_prefix(desc: str) -> str:
    # Exports can have trailing whitespace, e.g. 'Note : * ' -> strip it.
    return desc.removeprefix(NOTE_PREFIX).strip()


def read_marks_from_edf(edf_path: Path) -> np.ndarray:
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = [strip_prefix(d) for d in raw.annotations.description]
    marks = np.array([t for t, d in zip(onsets, descs) if d == PD_MARK], dtype=float)
    return np.sort(marks)


def label_windows(onsets, seg_types, marks, window_sec, m, g):
    onsets = np.asarray(onsets, dtype=float)
    seg_types = np.asarray([str(s).upper() for s in seg_types])
    marks = np.asarray(marks, dtype=float)

    labels = np.empty(len(onsets), dtype=np.int8)
    for i, s in enumerate(onsets):
        e = s + window_sec
        if seg_types[i] != PD:
            labels[i] = NEGATIVE
            continue
        if marks.size == 0:
            labels[i] = NEGATIVE
            continue
        in_core = (marks >= s + m) & (marks <= e - m)
        if in_core.any():
            labels[i] = POSITIVE
            continue
        gap = np.maximum.reduce([s - marks, marks - e, np.zeros_like(marks)])
        labels[i] = IGNORE if gap.min() < g else NEGATIVE
    return labels


# =================================================
# Process one window file
# =================================================

def process_file(win_path: Path, edf_path: Path, out_path: Path, m: float, g: float,
                 batch_windows: int = 512):
    z = np.load(win_path, allow_pickle=True)

    windows = z["windows"]                 # (N, n_channels, win_len)
    ch_names = z["ch_names"]
    sfreq = float(z["sfreq"])
    onsets = z["window_onsets_sec"]
    seg_types = z["segment_type"]
    window_sec = float(z["window_sec"]) if "window_sec" in z.files else 1.0

    if windows.ndim != 3 or windows.shape[0] == 0:
        print("  (no windows — skipped)")
        return

    n_windows = windows.shape[0]
    fnames = feature_names(ch_names)
    X = np.empty((n_windows, len(fnames)), dtype=np.float32)

    for start in range(0, n_windows, batch_windows):
        end = min(start + batch_windows, n_windows)
        for i in range(start, end):
            X[i] = extract_features_for_one_window(windows[i], sfreq)
        print(f"  features [{end}/{n_windows}]")

    # Labels from EDF marks
    marks = read_marks_from_edf(edf_path)
    if "pd_marks_sec" in z.files and len(np.asarray(z["pd_marks_sec"])) != len(marks):
        print(f"  ! mark count differs: EDF={len(marks)} vs NPZ={len(z['pd_marks_sec'])} "
              f"(using EDF marks)")

    labels = label_windows(onsets, seg_types, marks, window_sec, m, g)
    valid_mask = labels != IGNORE

    n_pos = int((labels == POSITIVE).sum())
    n_neg = int((labels == NEGATIVE).sum())
    n_ign = int((labels == IGNORE).sum())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y=labels,                      # 1 / 0 / -1  (filter with y != -1 before training)
        labels=labels,                 # alias, same values
        valid_mask=valid_mask,
        ch_names=np.array(ch_names, dtype=object),
        feature_names=np.array(fnames, dtype=object),
        window_onsets_sec=np.asarray(onsets, dtype=np.float64),
        segment_type=np.asarray(seg_types, dtype=object),
        segment_id=z["segment_id"] if "segment_id" in z.files else np.array([]),
        pd_marks_sec=marks,
        sfreq=sfreq,
        window_sec=window_sec,
        stride_sec=float(z["stride_sec"]) if "stride_sec" in z.files else float("nan"),
        label_m_sec=np.float32(m),
        label_g_sec=np.float32(g),
        source_npz=win_path.name,
        source_edf=edf_path.name,
    )
    print(f"  Saved: {out_path} | X={X.shape} | "
          f"POSITIVE={n_pos} NEGATIVE={n_neg} IGNORE={n_ign} | marks={len(marks)}")


# =================================================
# Main
# =================================================

def main():
    ap = argparse.ArgumentParser(description="Extract handcrafted features + labels for P58 sliding windows.")
    ap.add_argument("--m", type=float, default=M_SEC, help="Edge margin in seconds")
    ap.add_argument("--g", type=float, default=G_SEC, help="Guard band in seconds")
    args = ap.parse_args()

    win_files = sorted(WINDOWS_DIR.glob("*_windows.npz"))
    if not win_files:
        raise RuntimeError(f"No *_windows.npz files in {WINDOWS_DIR}")

    print(f"Found {len(win_files)} window files | m={args.m}s g={args.g}s")
    print("Labels: 1=positive, 0=negative, -1=ignore. Filter with `y != -1`.\n")

    for win_path in win_files:
        stem = win_path.name.replace("_windows.npz", "")
        edf_path = DATA_DIR / f"{stem}.edf"
        out_path = OUT_DIR / f"{stem}_features.npz"

        print(f"[{stem}]")
        if not edf_path.exists():
            print(f"  ! EDF not found ({edf_path}) — skipping")
            continue

        try:
            process_file(win_path, edf_path, out_path, args.m, args.g)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()