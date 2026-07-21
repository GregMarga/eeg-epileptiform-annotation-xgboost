"""
generate_clinical_review_edfs.py
================================
Produces clinician-review EDF files for BOTH representations.

For every recording:
  1. Build the annotation-centered POOL and the sliding TEST set (identical
     protocol to active_learning_both_representations.py).
  2. Run ACTIVE acquisition (uncertainty + density, w=0.5) up to a FIXED
     budget of 20 labels (n_shot = 10: 10 positive-equivalent, 10 negative).
  3. Predict on the full sliding TEST set at DECISION_THRESHOLD.
  4. Union the overlapping positive windows into intervals (no gap bridging,
     no minimum duration, no removal of isolated positives).
  5. Re-export the original EDF with ALL original annotations preserved
     verbatim, plus the model intervals as `MODEL_A_PD` / `MODEL_B_PD`.
  6. Write a per-model CSV with the raw probabilities (NOT shown in the EDF).

Blinding:  MODEL_A = handcrafted, MODEL_B = LaBraM.
The mapping is written to `MODEL_MAPPING.txt` for your records only — do not
send that file to the clinician.

Requires: mne, edfio  (pip install edfio)
"""

import csv
import shutil
from pathlib import Path

import numpy as np
import mne
from sklearn.preprocessing import StandardScaler


# =================================================================
# Config
# =================================================================

RANDOM_SEED = 42

BUDGET = 20                  # total labels; n_shot = 10 per class
STRATEGY = "active"
ACTIVE_DENSITY_WEIGHT = 0.5
DECISION_THRESHOLD = 0.4     # same operating point as the reported results

IGNORE_LABEL = -1
PD, NON_PD = "PD", "NON_PD"
POOL_NEG_NONPD_ONLY = True

# Merge two consecutive positive windows only if their intervals overlap or
# touch. Set to 0.0 for strict overlap-only. This is NOT gap bridging.
MERGE_TOLERANCE_SEC = 0.0

# How model intervals are written into the EDF:
#   "start_stop"  -> two point annotations, MODEL_X_PD_START / MODEL_X_PD_STOP,
#                    mirroring the clinician's own PD_START / PD_STOP marks.
#                    The START also carries the interval duration, so viewers
#                    that shade durations still show the region.
#   "duration"    -> a single annotation per interval carrying the duration
#                    (the previous behaviour).
ANNOTATION_STYLE = "start_stop"

# EDF header physical_min / physical_max are 8-character fields, so any channel
# whose amplitude exceeds ~1e8 uV cannot be written. Such channels are junk
# (DC / trigger / mis-scaled) rather than EEG, so they are dropped and reported.
EDF_PHYS_LIMIT_UV = 99_999_999.0
DROP_UNEXPORTABLE_CHANNELS = True

# Channel selection for the exported EDF. read_raw_edf types nearly everything
# as 'eeg', so selection has to be by NAME, not by channel type.
#
# EXPORT_CHANNELS: explicit keep-list (case-insensitive). None -> keep all
#   channels except those matching DROP_CHANNEL_PATTERNS.
# DROP_CHANNEL_PATTERNS: substrings marking non-physiological channels. The
#   'Unspec *' channels are recorder telemetry (CPU / memory / network) and
#   carry values far outside the EDF physical range.
EXPORT_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz",
    "C3", "C4", "Cz", "T3", "T4", "T5", "T6",
    "P3", "P4", "Pz", "O1", "O2", "ECG",
]
DROP_CHANNEL_PATTERNS = ("unspec",)

# How many requested channels may legitimately be absent before the export is
# treated as a label-matching failure rather than a missing electrode.
MAX_MISSING_CHANNELS = 2


def _norm_ch(name: str) -> str:
    """Normalize an EDF channel label for matching.

    Recorders prefix labels ('EEG Fp1', 'ECG EKG') and add reference suffixes
    ('Fp1-Ref'), so matching has to be done on a canonical form.
    """
    s = str(name).strip().lower()
    for prefix in ("eeg ", "ecg ", "eog ", "emg ", "poly "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    for suffix in ("-ref", "-le", "-avg", "ref"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    return "".join(c for c in s if c.isalnum())


_UV_TYPES = ("eeg", "eog", "ecg", "emg", "seeg", "ecog", "bio", "dbs")

# Original EDFs live here; review EDFs are written to the same folder.
DATA_DIR = Path("../../../data/evaluation_recordings")
OUTPUT_DIR = DATA_DIR
CSV_DIR = Path("clinical_review_csv")

# ---- handcrafted paths (pool + test share a folder, differ by suffix) -------
HANDCRAFTED_FEATURES_DIR = DATA_DIR / "features_1s_labeled_persegment"
HANDCRAFTED_POOL_SUFFIX = "_annot_centered.npz"
HANDCRAFTED_TEST_SUFFIX = "_features.npz"

# ---- LaBraM paths (pool and test in different folders) ---------------------
LABRAM_POOL_DIR = DATA_DIR / "labram_annot_centered_windows_1s_embeddings"
LABRAM_TEST_DIR = DATA_DIR / "labeled"
LABRAM_POOL_SUFFIX = "_embeddings.npz"
LABRAM_TEST_SUFFIX = "_embeddings_labeled.npz"


# =================================================================
# Loaders — now also return onsets / segment ids / window length
# =================================================================

def _clean(z, X, labels, seg):
    """Drop IGNORE windows and return the keep mask so metadata stays aligned."""
    keep = labels != IGNORE_LABEL
    return X[keep], labels[keep].astype(np.uint8), seg[keep], keep


def _meta(z, keep):
    """Extract per-window metadata, filtered by the same keep mask."""
    onsets = (np.asarray(z["window_onsets_sec"], dtype=np.float64)[keep]
              if "window_onsets_sec" in z.files else None)
    seg_id = (np.asarray(z["segment_id"])[keep]
              if "segment_id" in z.files and len(np.asarray(z["segment_id"])) == len(keep)
              else None)
    window_sec = float(z["window_sec"]) if "window_sec" in z.files else 1.0
    return onsets, seg_id, window_sec


def load_handcrafted_recording(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    X = X.reshape(X.shape[0], n_channels, -1).mean(axis=1)     # (N, 16)
    labels = z["y"].astype(np.int64).ravel()
    seg = np.asarray([str(s).upper() for s in z["segment_type"]])
    X, y, seg, keep = _clean(z, X, labels, seg)
    onsets, seg_id, window_sec = _meta(z, keep)
    return X, y, seg, onsets, seg_id, window_sec


def load_labram_recording(path: Path):
    z = np.load(path, allow_pickle=True)
    X = z["embeddings"].astype(np.float32)
    labels = z["labels"].astype(np.int64).ravel()
    seg = np.asarray([str(s).upper() for s in z["segment_type"]])
    X, y, seg, keep = _clean(z, X, labels, seg)
    onsets, seg_id, window_sec = _meta(z, keep)
    return X, y, seg, onsets, seg_id, window_sec


# =================================================================
# Prototypical classifier (identical to the experiment)
# =================================================================

def prototypical_predict(X_support, y_support, X_query, standardize,
                         distance="euclidean"):
    if standardize:
        scaler = StandardScaler()
        X_support = scaler.fit_transform(X_support)
        X_query = scaler.transform(X_query)

    proto_pos = X_support[y_support == 1].mean(axis=0)
    proto_neg = X_support[y_support == 0].mean(axis=0)

    if distance == "euclidean":
        d_pos = np.linalg.norm(X_query - proto_pos, axis=1)
        d_neg = np.linalg.norm(X_query - proto_neg, axis=1)
    elif distance == "cosine":
        def cosine_dist(X, proto):
            num = X @ proto
            den = np.linalg.norm(X, axis=1) * np.linalg.norm(proto) + 1e-8
            return 1 - num / den
        d_pos = cosine_dist(X_query, proto_pos)
        d_neg = cosine_dist(X_query, proto_neg)
    else:
        raise ValueError(f"Unknown distance: {distance}")

    logit_pos, logit_neg = -d_pos, -d_neg
    m = np.maximum(logit_pos, logit_neg)
    exp_pos, exp_neg = np.exp(logit_pos - m), np.exp(logit_neg - m)
    return exp_pos / (exp_pos + exp_neg)


# =================================================================
# Density / acquisition
# =================================================================

def density_scores(X_pool):
    n = len(X_pool)
    if n <= 1:
        return np.ones(n)
    sq = np.einsum("ij,ij->i", X_pool, X_pool)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X_pool @ X_pool.T)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2)
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    return sim.sum(axis=1) / (n - 1)


def _minmax(v):
    vmin, vmax = v.min(), v.max()
    if vmax > vmin:
        return (v - vmin) / (vmax - vmin)
    return np.ones_like(v)


def make_pool(y_pool, seg_pool, rng):
    pos = np.where(y_pool == 1)[0].copy()
    neg = np.where(y_pool == 0)[0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)

    if POOL_NEG_NONPD_ONLY:
        neg = np.array([int(i) for i in neg if seg_pool[i] == NON_PD], dtype=int)

    if len(pos) < 1 or len(neg) < 1:
        return None

    seed_idx = np.array([int(pos[0]), int(neg[0])])
    pool_idx = np.concatenate([pos, neg])
    return pool_idx, seed_idx


def acquire_to_budget(X_pool, y_pool, pool_idx, seed_idx, standardize, budget):
    """Run ACTIVE acquisition until `budget` labels are held. Deterministic."""
    labeled = [int(i) for i in seed_idx]
    pool = [int(i) for i in pool_idx if int(i) not in set(labeled)]

    # Density is a property of the initial pool -> computed once.
    p0 = np.array(pool)
    if standardize:
        Xp = StandardScaler().fit_transform(X_pool[p0]).astype(np.float64)
    else:
        Xp = X_pool[p0].astype(np.float64)
    dens_lookup = np.zeros(len(X_pool), dtype=np.float64)
    dens_lookup[p0] = density_scores(Xp)

    while len(labeled) < budget and pool:
        proba = prototypical_predict(
            X_pool[labeled], y_pool[np.array(labeled)], X_pool[pool], standardize
        )
        uncertainty = 1.0 - 2.0 * np.abs(proba - 0.5)
        dens = dens_lookup[np.array(pool)]
        w = ACTIVE_DENSITY_WEIGHT
        score = (1.0 - w) * _minmax(uncertainty) + w * _minmax(dens)
        pick = int(pool[int(np.argmax(score))])
        labeled.append(pick)
        pool.remove(pick)

    return labeled


# =================================================================
# Window intervals -> merged events
# =================================================================

def merge_positive_windows(onsets, window_sec, positive_mask, segment_id=None,
                           tol=MERGE_TOLERANCE_SEC):
    """Union of overlapping positive-window intervals.

    Merges [s_i, s_i + window_sec] with the next interval only when they
    already overlap or touch (next_start <= current_end + tol). No gap
    bridging, no minimum duration, isolated positives are kept as-is.
    Windows from different segments are never merged.
    """
    idx = np.where(positive_mask)[0]
    if len(idx) == 0:
        return []

    order = np.argsort(onsets[idx], kind="stable")
    idx = idx[order]

    events = []
    cur_start = onsets[idx[0]]
    cur_end = cur_start + window_sec
    cur_seg = segment_id[idx[0]] if segment_id is not None else None

    for i in idx[1:]:
        s = onsets[i]
        e = s + window_sec
        seg = segment_id[i] if segment_id is not None else None
        same_segment = (segment_id is None) or (seg == cur_seg)

        if same_segment and s <= cur_end + tol:
            cur_end = max(cur_end, e)
        else:
            events.append((cur_start, cur_end))
            cur_start, cur_end, cur_seg = s, e, seg

    events.append((cur_start, cur_end))
    return events


# =================================================================
# EDF writing
# =================================================================

def find_unexportable_channels(raw):
    """Channels whose amplitude cannot fit the 8-character EDF header fields."""
    types = raw.get_channel_types()
    bad = []
    for i, ch in enumerate(raw.ch_names):
        x = raw.get_data(picks=[i])[0]
        scale = 1e6 if types[i] in _UV_TYPES else 1.0
        peak = float(np.max(np.abs(x))) * scale
        if not np.isfinite(peak) or peak > EDF_PHYS_LIMIT_UV:
            bad.append((ch, types[i], peak))
    return bad


def write_review_edf(edf_path: Path, events, label: str, out_path: Path):
    """Re-export the original EDF with original annotations + model events.

    NOTE: MNE re-writes the signal on export (16-bit re-quantization). The
    traces remain visually equivalent but are not bit-identical to the source.
    """
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")

    if EXPORT_CHANNELS is not None:
        wanted = {_norm_ch(c) for c in EXPORT_CHANNELS}
        keep = [c for c in raw.ch_names if _norm_ch(c) in wanted]
        missing = wanted - {_norm_ch(c) for c in keep}
        if missing:
            print(f"    ! requested channels not present: {sorted(missing)}")
        # A partial match almost always means a label-format mismatch rather
        # than genuinely absent electrodes -> refuse to write a crippled file.
        if len(missing) > MAX_MISSING_CHANNELS:
            raise ValueError(
                f"{edf_path.name}: only {len(keep)}/{len(wanted)} requested "
                f"channels matched. Available labels: {raw.ch_names}"
            )
    else:
        keep = [c for c in raw.ch_names
                if not any(p in c.lower() for p in DROP_CHANNEL_PATTERNS)]

    dropped = [c for c in raw.ch_names if c not in keep]
    if dropped:
        print(f"    dropped channels: {dropped}")
    print(f"    kept {len(keep)}: {keep}")
    raw.pick(keep)

    bad = find_unexportable_channels(raw)
    if bad:
        for ch, t, peak in bad:
            print(f"    ! unexportable channel {ch} ({t}): peak={peak:.3e} uV")
        if DROP_UNEXPORTABLE_CHANNELS:
            raw.drop_channels([ch for ch, _, _ in bad])
            print(f"    dropped {len(bad)} unexportable channel(s)")
        else:
            raise ValueError(
                f"{edf_path.name}: channels exceed the EDF physical range "
                f"({[c for c, _, _ in bad]}). Set DROP_UNEXPORTABLE_CHANNELS=True "
                f"or fix the source file."
            )

    if len(raw.ch_names) == 0:
        raise ValueError(f"{edf_path.name}: no exportable channels left")

    orig = raw.annotations
    onsets = list(orig.onset)
    durations = list(orig.duration)
    descriptions = list(orig.description)

    for start, end in events:
        if ANNOTATION_STYLE == "start_stop":
            # START keeps the duration so duration-aware viewers still shade
            # the region; STOP is an explicit point marker at the end.
            onsets.append(float(start))
            durations.append(float(end - start))
            descriptions.append(f"{label}_START")

            onsets.append(float(end))
            durations.append(0.0)
            descriptions.append(f"{label}_STOP")
        elif ANNOTATION_STYLE == "duration":
            onsets.append(float(start))
            durations.append(float(end - start))
            descriptions.append(label)
        else:
            raise ValueError(f"Unknown ANNOTATION_STYLE: {ANNOTATION_STYLE}")

    combined = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=descriptions,
        orig_time=orig.orig_time,
    )
    raw.set_annotations(combined)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mne.export.export_raw(str(out_path), raw, fmt="edf",
                          overwrite=True, verbose="ERROR")
    return len(orig), len(events)


# =================================================================
# Recording assembly
# =================================================================

def build_recordings(loader, pool_dir, pool_suffix, test_dir, test_suffix):
    pool_files = sorted(Path(pool_dir).glob(f"*{pool_suffix}"))
    if not pool_files:
        raise RuntimeError(f"No *{pool_suffix} files in {pool_dir}")

    recordings = {}
    for pool_path in pool_files:
        stem = pool_path.name.replace(pool_suffix, "")
        test_path = Path(test_dir) / f"{stem}{test_suffix}"
        if not test_path.exists():
            print(f"  [SKIP] no sliding test file for {stem}")
            continue

        X_ann, y_ann, seg_ann, _, _, _ = loader(pool_path)
        pos_mask = y_ann == 1

        X_sl, y_sl, seg_sl, onsets_sl, segid_sl, window_sec = loader(test_path)
        neg_mask = y_sl == 0

        if onsets_sl is None:
            print(f"  [SKIP] {stem}: no window_onsets_sec in the test file")
            continue

        X_pool = np.concatenate([X_ann[pos_mask], X_sl[neg_mask]], axis=0)
        y_pool = np.concatenate([
            np.ones(int(pos_mask.sum()), dtype=np.uint8),
            np.zeros(int(neg_mask.sum()), dtype=np.uint8),
        ])
        seg_pool = np.concatenate([seg_ann[pos_mask], seg_sl[neg_mask]])

        # Pool index -> test index, for the negative block only.
        n_pos = int(pos_mask.sum())
        neg_test_idx = np.where(neg_mask)[0]

        recordings[stem] = dict(
            X_pool=X_pool, y_pool=y_pool, seg_pool=seg_pool,
            X_test=X_sl, y_test=y_sl, seg_test=seg_sl,
            onsets_test=onsets_sl, segid_test=segid_sl,
            window_sec=window_sec, n_pos_pool=n_pos, neg_test_idx=neg_test_idx,
        )
        print(f"  {stem}: pool n={len(y_pool)} (pos={n_pos}) | "
              f"test n={len(y_sl)} (pos={int(y_sl.sum())})")
    return recordings


REPRESENTATIONS = {
    "MODEL_A": dict(
        real_name="handcrafted features",
        loader=load_handcrafted_recording,
        pool_dir=HANDCRAFTED_FEATURES_DIR, pool_suffix=HANDCRAFTED_POOL_SUFFIX,
        test_dir=HANDCRAFTED_FEATURES_DIR, test_suffix=HANDCRAFTED_TEST_SUFFIX,
        standardize=True,
    ),
    "MODEL_B": dict(
        real_name="LaBraM embeddings",
        loader=load_labram_recording,
        pool_dir=LABRAM_POOL_DIR, pool_suffix=LABRAM_POOL_SUFFIX,
        test_dir=LABRAM_TEST_DIR, test_suffix=LABRAM_TEST_SUFFIX,
        standardize=False,
    ),
}

CSV_FIELDS = [
    "edf_name", "model", "model_real_name", "segment_id", "segment_type",
    "window_onset_sec", "window_end_sec", "reference_label",
    "probability", "binary_prediction", "was_labeled",
]


# =================================================================
# Main
# =================================================================

def run_one(model_tag, cfg):
    print(f"\n{'#' * 64}\n#  {model_tag}  ({cfg['real_name']})\n{'#' * 64}")

    recordings = build_recordings(
        cfg["loader"], cfg["pool_dir"], cfg["pool_suffix"],
        cfg["test_dir"], cfg["test_suffix"],
    )
    if not recordings:
        raise RuntimeError(f"[{model_tag}] no usable recordings")

    csv_rows = []
    for stem, rec in recordings.items():
        edf_path = DATA_DIR / f"{stem}.edf"
        if not edf_path.exists():
            print(f"  [SKIP] original EDF not found: {edf_path}")
            continue

        rng = np.random.default_rng(RANDOM_SEED)
        built = make_pool(rec["y_pool"], rec["seg_pool"], rng)
        if built is None:
            print(f"  [SKIP] {stem}: pool needs >=1 positive and >=1 NON_PD negative")
            continue
        pool_idx, seed_idx = built

        labeled = acquire_to_budget(
            rec["X_pool"], rec["y_pool"], pool_idx, seed_idx,
            cfg["standardize"], BUDGET,
        )
        n_pos_lab = int(rec["y_pool"][np.array(labeled)].sum())
        print(f"  {stem}: labeled={len(labeled)} "
              f"(pos={n_pos_lab}, neg={len(labeled) - n_pos_lab})")

        proba = prototypical_predict(
            rec["X_pool"][labeled], rec["y_pool"][np.array(labeled)],
            rec["X_test"], cfg["standardize"],
        )
        pred = proba >= DECISION_THRESHOLD

        # Which TEST windows were consumed as negative support.
        was_labeled = np.zeros(len(rec["y_test"]), dtype=bool)
        for pi in labeled:
            if pi >= rec["n_pos_pool"]:
                was_labeled[rec["neg_test_idx"][pi - rec["n_pos_pool"]]] = True

        events = merge_positive_windows(
            rec["onsets_test"], rec["window_sec"], pred, rec["segid_test"],
        )

        out_path = OUTPUT_DIR / f"{stem}_{model_tag}_review.edf"
        n_orig, n_ev = write_review_edf(
            edf_path, events, f"{model_tag}_PD", out_path,
        )
        print(f"    -> {out_path.name} | original annots={n_orig} | "
              f"model intervals={n_ev} (from {int(pred.sum())} positive windows)")

        segid = rec["segid_test"]
        for i in range(len(rec["y_test"])):
            csv_rows.append({
                "edf_name": out_path.name,
                "model": model_tag,
                "model_real_name": cfg["real_name"],
                "segment_id": segid[i] if segid is not None else "",
                "segment_type": rec["seg_test"][i],
                "window_onset_sec": float(rec["onsets_test"][i]),
                "window_end_sec": float(rec["onsets_test"][i] + rec["window_sec"]),
                "reference_label": int(rec["y_test"][i]),
                "probability": float(proba[i]),
                "binary_prediction": int(pred[i]),
                "was_labeled": int(was_labeled[i]),
            })

    CSV_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = CSV_DIR / f"predictions_{model_tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n  Probabilities saved to: {csv_path}")


def main():
    try:
        import edfio  # noqa: F401
    except ImportError:
        raise SystemExit("edfio is required for EDF export:  pip install edfio")

    print(f"Budget={BUDGET} (n_shot=10) | strategy={STRATEGY} | "
          f"threshold={DECISION_THRESHOLD} | seed={RANDOM_SEED}")

    for model_tag, cfg in REPRESENTATIONS.items():
        run_one(model_tag, cfg)

    mapping = OUTPUT_DIR / "MODEL_MAPPING.txt"
    with open(mapping, "w") as f:
        f.write("INTERNAL — do not send to the clinician\n")
        for tag, cfg in REPRESENTATIONS.items():
            f.write(f"{tag} = {cfg['real_name']}\n")
        f.write(f"\nbudget={BUDGET} (n_shot=10), strategy={STRATEGY}, "
                f"threshold={DECISION_THRESHOLD}, seed={RANDOM_SEED}\n")
    print(f"\nMapping written to: {mapping}  (keep this out of the clinician's folder)")
    print("Done.")


if __name__ == "__main__":
    main()