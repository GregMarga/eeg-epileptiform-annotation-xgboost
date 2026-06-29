from pathlib import Path
import numpy as np
import mne

DATA_DIR = Path("../../../data/evaluation_recordings")
LABELED  = DATA_DIR / "labeled" / "P58_GHB_M1681_0000031_embeddings_labeled.npz"
EDF      = DATA_DIR / "P58_GHB_M1681_0000031.edf"

NOTE_PREFIX = "Note : "

# -------------------------------------------------
# 1) What is inside the labeled NPZ?
# -------------------------------------------------
z = np.load(LABELED, allow_pickle=True)
print("Keys:", list(z.files))

labels = z["labels"].ravel()
vals, counts = np.unique(labels, return_counts=True)
print("Label distribution:", dict(zip(vals.tolist(), counts.tolist())))

seg = np.asarray([str(s).upper() for s in z["segment_type"]])
svals, scounts = np.unique(seg, return_counts=True)
print("segment_type distribution:", dict(zip(svals.tolist(), scounts.tolist())))

onsets = np.asarray(z["window_onsets_sec"], dtype=float)
print(f"window_onsets_sec: n={len(onsets)} range=[{onsets.min():.1f}, {onsets.max():.1f}]")

npz_marks = np.asarray(z["pd_marks_sec"], dtype=float) if "pd_marks_sec" in z.files else np.array([])
print(f"pd_marks_sec in NPZ: n={len(npz_marks)}"
      + (f" range=[{npz_marks.min():.1f}, {npz_marks.max():.1f}]" if len(npz_marks) else ""))

# -------------------------------------------------
# 2) What '*' marks does the EDF actually contain?
# -------------------------------------------------
raw = mne.io.read_raw_edf(str(EDF), preload=False, verbose="ERROR")
descs_raw = list(raw.annotations.description)
print(f"\nTotal EDF annotations: {len(descs_raw)}")

# Show the unique descriptions (repr exposes hidden spaces / glyphs)
uniq = sorted(set(descs_raw))
print("Unique annotation descriptions (repr):")
for u in uniq:
    print("   ", repr(u))

# Try the exact same parsing the labeling script uses
stripped = [d.removeprefix(NOTE_PREFIX) for d in descs_raw]
marks_exact = [t for t, d in zip(raw.annotations.onset, stripped) if d == "*"]
print(f"\nMarks matched by exact '*' rule: {len(marks_exact)}")

# A looser check: anything containing a star
marks_loose = [t for t, d in zip(raw.annotations.onset, descs_raw) if "*" in d]
print(f"Annotations containing '*' anywhere: {len(marks_loose)}")

# -------------------------------------------------
# 3) If marks exist, do they land inside PD-segment windows?
# -------------------------------------------------
if len(npz_marks):
    pd_mask = seg == "PD"
    if pd_mask.any():
        pd_on = onsets[pd_mask]
        lo, hi = pd_on.min(), pd_on.max() + 1.0
        inside = np.sum((npz_marks >= lo) & (npz_marks <= hi))
        print(f"\nPD windows time span: [{lo:.1f}, {hi:.1f}]")
        print(f"NPZ marks falling within that span: {inside}/{len(npz_marks)}")
    else:
        print("\nNo PD-segment windows at all -> positives impossible.")