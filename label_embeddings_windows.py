import argparse
from pathlib import Path

import numpy as np
import mne


# --------------------------------------------------
# Labeling parameters (seconds)
# --------------------------------------------------

# m: edge margin. A '*' mark must sit at least m inside the window for the
#    window to count as positive, so the whole discharge waveform fits in it.
#    Tie this to ~half the typical discharge duration (~0.2-0.3 s -> m ~ 0.1-0.15).
M_SEC = 0.10

# g: guard band. A window with no qualifying mark is only labeled NEGATIVE if the
#    nearest mark is at least g away from the window interval; otherwise the window
#    sits in the transition zone and is EXCLUDED. g is larger than m on purpose:
#    it buffers against a neighboring discharge "bleeding" into the window.
G_SEC = 0.30

# Label values
POSITIVE = 1
NEGATIVE = 0
IGNORE   = -1

# Annotation parsing (matches the windowing script)
NOTE_PREFIX = "Note : "
PD_MARK = "*"


def strip_prefix(desc: str) -> str:
    # NOTE: exports can have trailing whitespace, e.g. 'Note : * ' -> strip it.
    return desc.removeprefix(NOTE_PREFIX).strip()


def read_marks_from_edf(edf_path: Path) -> np.ndarray:
    """Return absolute onset times (seconds) of all '*' discharge marks."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose="ERROR")
    onsets = np.asarray(raw.annotations.onset, dtype=float)
    descs = [strip_prefix(d) for d in raw.annotations.description]
    marks = np.array([t for t, d in zip(onsets, descs) if d == PD_MARK], dtype=float)
    return np.sort(marks)


# --------------------------------------------------
# Core labeling rule
# --------------------------------------------------

def label_windows(onsets, seg_types, marks, window_sec, m, g):
    """Assign a label to each window.

    Per window [s, e=s+window_sec]:
      - NON_PD segment                       -> NEGATIVE
      - PD segment, a mark inside [s+m, e-m]  -> POSITIVE
      - PD segment, no such mark but nearest
        mark closer than g to the interval    -> IGNORE (transition zone)
      - PD segment, otherwise                 -> NEGATIVE (clean inter-discharge)
    """
    onsets = np.asarray(onsets, dtype=float)
    seg_types = np.asarray([str(s).upper() for s in seg_types])
    marks = np.asarray(marks, dtype=float)

    labels = np.empty(len(onsets), dtype=np.int8)

    for i, s in enumerate(onsets):
        e = s + window_sec

        if seg_types[i] != "PD":
            labels[i] = NEGATIVE
            continue

        if marks.size == 0:
            labels[i] = NEGATIVE
            continue

        # Positive if a mark sits well inside the window (core region).
        in_core = (marks >= s + m) & (marks <= e - m)
        if in_core.any():
            labels[i] = POSITIVE
            continue

        # Distance from each mark to the window interval (0 if inside [s, e]).
        gap = np.maximum.reduce([s - marks, marks - e, np.zeros_like(marks)])
        if gap.min() < g:
            labels[i] = IGNORE      # transition zone: mark near an edge / just outside
        else:
            labels[i] = NEGATIVE    # clean inter-discharge background

    return labels


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("../../../data/evaluation_recordings")
# Folder with the *_embeddings.npz produced by the embedding-extraction script
EMB_DIR  = DATA_DIR / "labram_sliding_windows_1s_75overlap_embeddings"
# All labeled outputs go here
OUT_DIR  = DATA_DIR / "labeled"


def label_one(edf_path: Path, npz_path: Path, out_path: Path, m: float, g: float):
    d = np.load(npz_path, allow_pickle=True)

    onsets = d["window_onsets_sec"]
    seg_types = d["segment_type"]
    window_sec = float(d["window_sec"]) if "window_sec" in d.files else 1.0

    marks = read_marks_from_edf(edf_path)

    # Sanity check against the marks carried in the NPZ, if present.
    if "pd_marks_sec" in d.files:
        npz_marks = np.asarray(d["pd_marks_sec"], dtype=float)
        if len(npz_marks) != len(marks):
            print(f"  ! mark count differs: EDF={len(marks)} vs NPZ={len(npz_marks)} "
                  f"(using EDF marks)")

    print(f"  windows={len(onsets)} | window_sec={window_sec} | marks={len(marks)} "
          f"| m={m}s g={g}s")

    labels = label_windows(onsets, seg_types, marks, window_sec, m, g)

    n_pos = int((labels == POSITIVE).sum())
    n_neg = int((labels == NEGATIVE).sum())
    n_ign = int((labels == IGNORE).sum())
    print(f"  POSITIVE={n_pos} | NEGATIVE={n_neg} | IGNORE={n_ign} (transition zone)")

    payload = {k: d[k] for k in d.files}
    payload["labels"] = labels
    payload["valid_mask"] = labels != IGNORE
    payload["label_m_sec"] = np.float32(m)
    payload["label_g_sec"] = np.float32(g)

    np.savez_compressed(out_path, **payload)
    print(f"  saved: {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Label sliding-window embeddings from EDF '*' marks.")
    ap.add_argument("--m", type=float, default=M_SEC, help="Edge margin in seconds")
    ap.add_argument("--g", type=float, default=G_SEC, help="Guard band in seconds")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_files = sorted(EMB_DIR.glob("*_embeddings.npz"))
    if not emb_files:
        raise RuntimeError(f"No *_embeddings.npz files in {EMB_DIR}")

    print(f"Found {len(emb_files)} embedding files")
    print(f"Labels: 1=positive, 0=negative, -1=ignore. Filter with `labels != -1`.\n")

    for npz_path in emb_files:
        stem = npz_path.name.replace("_embeddings.npz", "")
        edf_path = DATA_DIR / f"{stem}.edf"
        out_path = OUT_DIR / f"{stem}_embeddings_labeled.npz"

        print(f"[{stem}]")
        if not edf_path.exists():
            print(f"  ! EDF not found ({edf_path}) — skipping")
            continue

        try:
            label_one(edf_path, npz_path, out_path, args.m, args.g)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()