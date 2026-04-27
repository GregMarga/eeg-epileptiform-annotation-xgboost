import re
import csv
from pathlib import Path
import numpy as np
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

EMBEDDINGS_DIR = Path("../../../data/labram_classification_1s")
FEATURES_DIR   = Path("../../../data/80hz_freq_time_features_cache_basic")

N_SHOT_RANGE = list(range(1, 21))
RANDOM_SEED  = 42

USE_LABRAM_ONLY      = False
USE_HANDCRAFTED_ONLY = True
USE_COMBINED         = False

# Επιλεγμένο handcrafted feature index (0-based)
HC_FEATURE_IDX = 6
N_HC = 1   # κρατάμε ΕΝΑ feature → χρειάζεται για το slicing στο COMBINED

CSV_OUTPUT = Path("results_handcrafted_f7.csv")

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def build_index(directory: Path, suffix: str) -> dict[str, Path]:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"No *{suffix} files found in {directory}")
    return {f.name.replace(suffix, ""): f for f in files}

def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    return z["X"].astype(np.float32), z["y"].astype(np.uint8).ravel()

def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples  = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # (N, 16)

    # ✂ κρατάμε μόνο το feature με index HC_FEATURE_IDX
    X = X[:, HC_FEATURE_IDX:HC_FEATURE_IDX + 1]   # shape (N, 1)

    y = z["y"].astype(np.uint8).ravel()
    return X, y

def load_recording(base, emb_index, feat_index):
    if USE_HANDCRAFTED_ONLY:
        if base not in feat_index:
            raise ValueError(f"No handcrafted features for {base}")
        return load_features(feat_index[base])
    if USE_COMBINED:
        X_emb, y_emb = load_embeddings(emb_index[base])
        if base not in feat_index:
            raise ValueError(f"No handcrafted features for {base}")
        X_feat, y_feat = load_features(feat_index[base])
        if len(y_feat) != len(y_emb):
            raise ValueError(f"Window mismatch for {base}")
        if not np.array_equal(y_feat, y_emb):
            raise ValueError(f"Label mismatch for {base}")
        return np.hstack([X_feat, X_emb]), y_emb
    return load_embeddings(emb_index[base])

def prototypical_predict(X_support, y_support, X_query, distance="euclidean"):
    if USE_HANDCRAFTED_ONLY:
        scaler = StandardScaler()
        X_support = scaler.fit_transform(X_support)
        X_query   = scaler.transform(X_query)
    elif USE_COMBINED:
        scaler = StandardScaler()
        X_support = X_support.copy()
        X_query   = X_query.copy()
        X_support[:, :N_HC] = scaler.fit_transform(X_support[:, :N_HC])
        X_query[:, :N_HC]   = scaler.transform(X_query[:, :N_HC])

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

    logit_pos = -d_pos
    logit_neg = -d_neg
    exp_pos = np.exp(logit_pos - np.maximum(logit_pos, logit_neg))
    exp_neg = np.exp(logit_neg - np.maximum(logit_pos, logit_neg))
    return exp_pos / (exp_pos + exp_neg)

def sample_support(X, y, n_shot, rng):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < n_shot:
        raise ValueError(f"Not enough positives: {len(pos_idx)} < {n_shot}")
    if len(neg_idx) < n_shot:
        raise ValueError(f"Not enough negatives: {len(neg_idx)} < {n_shot}")
    chosen_pos = rng.choice(pos_idx, size=n_shot, replace=False)
    chosen_neg = rng.choice(neg_idx, size=n_shot, replace=False)
    support_idx = np.concatenate([chosen_pos, chosen_neg])
    query_idx = np.setdiff1d(np.arange(len(y)), support_idx)
    return X[support_idx], y[support_idx], X[query_idx], y[query_idx]

def safe_roc_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))

def safe_pr_auc(y_true, y_proba):
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))

def evaluate(y_true, y_proba, thr=0.4) -> dict[str, float]:
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

def summarize_metric(values) -> tuple[float, float]:
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))

def run_for_n_shot(n_shot, recordings, emb_index, feat_index) -> dict[str, float]:
    rng = np.random.default_rng(RANDOM_SEED)
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}
    skipped = 0

    for base in recordings:
        try:
            X, y = load_recording(base, emb_index, feat_index)
            X_sup, y_sup, X_qry, y_qry = sample_support(X, y, n_shot, rng)
            if len(np.unique(y_qry)) < 2:
                skipped += 1
                if n_shot == 1:
                    print(f"  [SKIP] {base}: single class in query (pos={y_qry.sum()}, neg={(y_qry == 0).sum()})")
                continue
            y_proba = prototypical_predict(X_sup, y_sup, X_qry, distance="euclidean")
            m = evaluate(y_qry, y_proba)
            for k in keys:
                collected[k].append(m[k])
        except ValueError as e:
            skipped += 1
            if n_shot == 1:
                print(f"  [SKIP] {base}: {e}")

    means = {k: float(np.nanmean(collected[k])) if collected[k] else float("nan") for k in keys}
    stds  = {f"{k}_std": float(np.nanstd(collected[k])) if collected[k] else float("nan") for k in keys}
    return {"n_shot": n_shot, "n_recordings": len(recordings) - skipped, **means, **stds}

def main():
    assert sum([USE_LABRAM_ONLY, USE_HANDCRAFTED_ONLY, USE_COMBINED]) == 1
    mode = ("LaBraM only" if USE_LABRAM_ONLY
            else f"Handcrafted only (feature idx={HC_FEATURE_IDX})" if USE_HANDCRAFTED_ONLY
            else f"Combined (handcrafted feature idx={HC_FEATURE_IDX} + LaBraM)")
    print(f"Mode: {mode}")

    emb_index  = build_index(EMBEDDINGS_DIR, "_embeddings_labeled.npz")
    feat_index = build_index(FEATURES_DIR, "_features.npz")

    recordings = sorted(emb_index.keys())
    print(f"Recordings: {len(recordings)}")
    print(f"Sweeping N_SHOT: {N_SHOT_RANGE}\n")

    all_rows = []
    for n_shot in N_SHOT_RANGE:
        row = run_for_n_shot(n_shot, recordings, emb_index, feat_index)
        all_rows.append(row)
        print(
            f"n_shot={n_shot:2d} | n_rec={row['n_recordings']:3d} | "
            f"acc={row['acc']:.4f} ± {row['acc_std']:.4f} | "
            f"bacc={row['bacc']:.4f} | f1={row['f1']:.4f} | "
            f"roc_auc={row['roc_auc']:.4f}"
        )

    fieldnames = list(all_rows[0].keys())
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nResults saved to: {CSV_OUTPUT}")

if __name__ == "__main__":
    main()