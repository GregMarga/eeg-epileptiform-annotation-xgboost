import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

# -------------------------------------------------
# Paths
# -------------------------------------------------

EMBEDDINGS_DIR = Path("../../../data/labram_classification_1s")
FEATURES_DIR   = Path("../../../data/80hz_freq_time_features_cache_basic")

# -------------------------------------------------
# Flags (exactly one True)
# -------------------------------------------------

USE_LABRAM_ONLY      = False
USE_HANDCRAFTED_ONLY = True
USE_COMBINED         = False

# Threshold and distance
THRESHOLD = 0.4
DISTANCE  = "euclidean"  # "euclidean" or "cosine"

# Επιλεγμένο handcrafted feature index (0-based) — το 7ο κατά σειρά
HC_FEATURE_IDX = 6

# Πόσες handcrafted features κρατάμε στο τελικό X (για το slicing του scaler στο COMBINED)
N_HC = 1

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_indices(directory: Path, suffix: str) -> tuple[dict[str, list[Path]], dict[str, Path]]:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"No *{suffix} files found in {directory}")

    patient_index: dict[str, list[Path]] = {}
    path_map: dict[str, Path] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"  Skipping (no patient id): {f.name}")
            continue
        patient_index.setdefault(pid, []).append(f)
        base = f.name.replace(suffix, "")
        path_map[base] = f
    return patient_index, path_map


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples  = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # (N, 16)

    # ✂ κρατάμε μόνο το feature με index HC_FEATURE_IDX → shape (N, 1)
    X = X[:, HC_FEATURE_IDX:HC_FEATURE_IDX + 1]

    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_recording(base: str, emb_paths: dict[str, Path], feat_paths: dict[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    if USE_HANDCRAFTED_ONLY:
        if base not in feat_paths:
            raise ValueError(f"No handcrafted features for {base}")
        return load_features(feat_paths[base])

    if USE_LABRAM_ONLY:
        if base not in emb_paths:
            raise ValueError(f"No embeddings for {base}")
        return load_embeddings(emb_paths[base])

    # USE_COMBINED
    if base not in emb_paths or base not in feat_paths:
        raise ValueError(f"Missing data for {base}")
    X_feat, y_feat = load_features(feat_paths[base])
    X_emb,  y_emb  = load_embeddings(emb_paths[base])
    if len(y_feat) != len(y_emb):
        raise ValueError(f"Window mismatch for {base}")
    if not np.array_equal(y_feat, y_emb):
        raise ValueError(f"Label mismatch for {base}")
    return np.hstack([X_feat, X_emb]), y_emb


def load_patient(
    patient_bases: list[str],
    emb_paths: dict[str, Path],
    feat_paths: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for base in patient_bases:
        X, y = load_recording(base, emb_paths, feat_paths)
        xs.append(X)
        ys.append(y)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def get_patient_bases(
    patient: str,
    emb_patient_index: dict[str, list[Path]],
    feat_patient_index: dict[str, list[Path]],
) -> list[str]:
    emb_bases  = {p.name.replace("_embeddings_labeled.npz", "") for p in emb_patient_index.get(patient, [])}
    feat_bases = {p.name.replace("_features.npz", "")          for p in feat_patient_index.get(patient, [])}

    if USE_LABRAM_ONLY:
        return sorted(emb_bases)
    if USE_HANDCRAFTED_ONLY:
        return sorted(feat_bases)
    return sorted(emb_bases & feat_bases)


def fit_scaler(X_train: np.ndarray) -> StandardScaler | None:
    if USE_HANDCRAFTED_ONLY:
        return StandardScaler().fit(X_train)
    if USE_COMBINED:
        return StandardScaler().fit(X_train[:, :N_HC])
    return None


def apply_scaler(X: np.ndarray, scaler: StandardScaler | None) -> np.ndarray:
    if scaler is None:
        return X
    if USE_HANDCRAFTED_ONLY:
        return scaler.transform(X)
    X = X.copy()
    X[:, :N_HC] = scaler.transform(X[:, :N_HC])
    return X


def compute_global_prototypes(X_train: np.ndarray, y_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if (y_train == 1).sum() == 0:
        raise ValueError("No positive samples in training set")
    if (y_train == 0).sum() == 0:
        raise ValueError("No negative samples in training set")
    proto_pos = X_train[y_train == 1].mean(axis=0)
    proto_neg = X_train[y_train == 0].mean(axis=0)
    return proto_pos, proto_neg


def prototypical_predict_proba(
    X_query: np.ndarray,
    proto_pos: np.ndarray,
    proto_neg: np.ndarray,
    distance: str = "euclidean",
) -> np.ndarray:
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

    logit_pos = -d_pos
    logit_neg = -d_neg
    m = np.maximum(logit_pos, logit_neg)
    exp_pos = np.exp(logit_pos - m)
    exp_neg = np.exp(logit_neg - m)
    return exp_pos / (exp_pos + exp_neg)


def safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def safe_pr_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def evaluate_fold(y_true, y_proba, thr=0.5) -> dict[str, float]:
    y_pred = (y_proba >= thr).astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc":       float(accuracy_score(y_true, y_pred)),
        "bacc":      float(balanced_accuracy_score(y_true, y_pred)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   safe_roc_auc(y_true, y_proba),
        "pr_auc":    safe_pr_auc(y_true, y_proba),
        "tp": float(tp), "fp": float(fp),
        "tn": float(tn), "fn": float(fn),
    }


def summarize_metric(values):
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


# -------------------------------------------------
# LOPO with prototypical classifier
# -------------------------------------------------

def main():
    assert sum([USE_LABRAM_ONLY, USE_HANDCRAFTED_ONLY, USE_COMBINED]) == 1, \
        "Exactly one of the mode flags must be True"

    mode = ("LaBraM only" if USE_LABRAM_ONLY
            else f"Handcrafted only (feature idx={HC_FEATURE_IDX})" if USE_HANDCRAFTED_ONLY
            else f"Combined (handcrafted feature idx={HC_FEATURE_IDX} + LaBraM)")
    print(f"Mode: {mode}")
    print(f"Distance: {DISTANCE}, threshold: {THRESHOLD}\n")

    emb_patient_index,  emb_paths  = build_indices(EMBEDDINGS_DIR, "_embeddings_labeled.npz")
    feat_patient_index, feat_paths = build_indices(FEATURES_DIR,   "_features.npz")

    if USE_LABRAM_ONLY:
        patients = sorted(emb_patient_index.keys())
    elif USE_HANDCRAFTED_ONLY:
        patients = sorted(feat_patient_index.keys())
    else:
        patients = sorted(set(emb_patient_index) & set(feat_patient_index))

    print(f"Patients: {patients} (n={len(patients)})")

    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    fold_metrics: dict[str, dict[str, float]] = {}
    collected = {k: [] for k in keys}
    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}

    for test_pid in patients:
        train_pids = [p for p in patients if p != test_pid]

        # ------------------- TRAIN -------------------
        x_train_list, y_train_list = [], []
        for pid in train_pids:
            bases = get_patient_bases(pid, emb_patient_index, feat_patient_index)
            if not bases:
                continue
            x, y = load_patient(bases, emb_paths, feat_paths)
            x_train_list.append(x)
            y_train_list.append(y)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        scaler = fit_scaler(x_train)
        x_train_s = apply_scaler(x_train, scaler)

        proto_pos, proto_neg = compute_global_prototypes(x_train_s, y_train)

        # ------------------- TEST --------------------
        bases_test = get_patient_bases(test_pid, emb_patient_index, feat_patient_index)
        x_test, y_test = load_patient(bases_test, emb_paths, feat_paths)
        x_test_s = apply_scaler(x_test, scaler)

        y_proba = prototypical_predict_proba(x_test_s, proto_pos, proto_neg, distance=DISTANCE)

        m = evaluate_fold(y_test, y_proba, thr=THRESHOLD)
        fold_metrics[test_pid] = m

        for k in keys:
            collected[k].append(m[k])
        for c in ["tp", "fp", "tn", "fn"]:
            agg[c] += m[c]

        print(
            f"[{test_pid}] n={len(y_test)} pos={int(y_test.sum())} "
            f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
            f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}"
        )

    print("\n=== LOPO summary (mean ± std across patients) ===")
    for k in keys:
        mean, std = summarize_metric(collected[k])
        print(f"{k:>10}: {mean:.4f} ± {std:.4f}")

    print(f"\n=== Aggregate confusion (sum across folds, at thr={THRESHOLD}) ===")
    print(f"TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

    worst = sorted(fold_metrics.items(), key=lambda kv: kv[1]["f1"])[:5]
    print("\nWorst by F1:")
    for pid, m in worst:
        print(f"  {pid}: f1={m['f1']:.3f} (prec={m['precision']:.3f}, rec={m['recall']:.3f})")

    # ------------------- PLOT -------------------------
    patient_pattern = {
        "P20": "LPD", "P28": "LPD", "P36": "LPD",
        "P48": "GPD", "P49": "GPD", "P54": "GPD",
        "P55": "LRDA", "P58": "LPD", "P70": "GPD", "P73": "LPD",
    }
    pattern_colors = {"LPD": "tab:blue", "GPD": "tab:orange", "LRDA": "tab:green"}

    items = sorted(fold_metrics.items(), key=lambda kv: kv[1]["acc"])
    pids  = [pid for pid, _ in items]
    accs  = [m["acc"] for _, m in items]
    colors = [pattern_colors.get(patient_pattern.get(pid, "UNK"), "gray") for pid in pids]

    plt.figure(figsize=(10, 4))
    plt.bar(pids, accs, color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Patient (sorted)")
    plt.title(f"LOPO Prototypical Classification — {mode}")
    plt.grid(axis="y", alpha=0.3)
    plt.legend(handles=[
        Patch(facecolor="tab:blue",   label="LPD"),
        Patch(facecolor="tab:orange", label="GPD"),
        Patch(facecolor="tab:green",  label="LRDA"),
    ], title="Pattern")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()