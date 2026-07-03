from pathlib import Path

import numpy as np
import mne


# --------------------------------------------------
# Labeling rule
# --------------------------------------------------
# A window is POSITIVE if at least one '*' discharge mark falls inside it,
# and NEGATIVE otherwise. No edge margin, no guard band, no IGNORE class,
# no dependence on segment type ('*' marks only occur inside PD segments,
# so NON_PD windows have no marks and become NEGATIVE automatically).

POSITIVE = 1
NEGATIVE = 0

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

def label_windows(onsets, marks, window_sec):
    """Assign a label to each window.

    Per window [s, e = s + window_sec]:
      - POSITIVE if at least one '*' mark falls inside [s, e]
      - NEGATIVE otherwise
    """
    onsets = np.asarray(onsets, dtype=float)
    marks = np.sort(np.asarray(marks, dtype=float))

    labels = np.full(len(onsets), NEGATIVE, dtype=np.int8)
    if marks.size == 0:
        return labels

    starts = onsets
    ends = onsets + window_sec
    # A mark lies in [s, e] iff there is a mark index in [lo, hi).
    lo = np.searchsorted(marks, starts, side="left")
    hi = np.searchsorted(marks, ends, side="right")
    labels[hi > lo] = POSITIVE
    return labels


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_DIR = Path("../../../data/evaluation_recordings")
# Folder with the *_embeddings.npz produced by the embedding-extraction script
EMB_DIR  = DATA_DIR / "labram_sliding_windows_1s_75overlap_embeddings"
# All labeled outputs go here
OUT_DIR  = DATA_DIR / "labeled"


def label_one(edf_path: Path, npz_path: Path, out_path: Path):
    d = np.load(npz_path, allow_pickle=True)

    onsets = d["window_onsets_sec"]
    window_sec = float(d["window_sec"]) if "window_sec" in d.files else 1.0

    marks = read_marks_from_edf(edf_path)

    # Sanity check against the marks carried in the NPZ, if present.
    if "pd_marks_sec" in d.files:
        npz_marks = np.asarray(d["pd_marks_sec"], dtype=float)
        if len(npz_marks) != len(marks):
            print(f"  ! mark count differs: EDF={len(marks)} vs NPZ={len(npz_marks)} "
                  f"(using EDF marks)")

    print(f"  windows={len(onsets)} | window_sec={window_sec} | marks={len(marks)}")

    labels = label_windows(onsets, marks, window_sec)

    n_pos = int((labels == POSITIVE).sum())
    n_neg = int((labels == NEGATIVE).sum())
    print(f"  POSITIVE={n_pos} | NEGATIVE={n_neg}")

    payload = {k: d[k] for k in d.files}
    payload["labels"] = labels
    payload["valid_mask"] = np.ones(len(labels), dtype=bool)  # no IGNORE class now

    np.savez_compressed(out_path, **payload)
    print(f"  saved: {out_path}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_files = sorted(EMB_DIR.glob("*_embeddings.npz"))
    if not emb_files:
        raise RuntimeError(f"No *_embeddings.npz files in {EMB_DIR}")

    print(f"Found {len(emb_files)} embedding files")
    print("Labels: 1=positive, 0=negative. A window is positive iff a '*' mark is inside it.\n")

    for npz_path in emb_files:
        stem = npz_path.name.replace("_embeddings.npz", "")
        edf_path = DATA_DIR / f"{stem}.edf"
        out_path = OUT_DIR / f"{stem}_embeddings_labeled.npz"

        print(f"[{stem}]")
        if not edf_path.exists():
            print(f"  ! EDF not found ({edf_path}) — skipping")
            continue

        try:
            label_one(edf_path, npz_path, out_path)
        except Exception as e:
            print(f"  ERROR: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()