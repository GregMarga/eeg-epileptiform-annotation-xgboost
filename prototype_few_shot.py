import re
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

# -------------------------------------------------
# Paths
# -------------------------------------------------

EMBEDDINGS_DIR = Path("../data/labram_classification")
FEATURES_DIR   = Path("../data/80hz_freq_time_features_cache_basic")

N_SHOT      = 12       # positives AND negatives per support set
RANDOM_SEED = 42
USE_COMBINED = False   # True = handcrafted + LaBraM, False = LaBraM only
USE_HANDCRAFTED_ONLY = True
# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_index(directory: Path, suffix: str) -> dict[str, Path]:
    """Returns {base_name: path} per recording file."""
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
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_recording(base, emb_index, feat_index):
    if USE_HANDCRAFTED_ONLY and base in feat_index:
        return load_features(feat_index[base])

    X_emb, y_emb = load_embeddings(emb_index[base])
    if USE_COMBINED and base in feat_index:
        X_feat, y_feat = load_features(feat_index[base])
        if len(y_feat) != len(y_emb):
            raise ValueError(f"Window mismatch for {base}")
        if not np.array_equal(y_feat, y_emb):
            raise ValueError(f"Label mismatch for {base}")
        X = np.hstack([X_feat, X_emb])
        return X, y_emb

    return X_emb, y_emb


def sample_support(
    X: np.ndarray,
    y: np.ndarray,
    n_shot: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Randomly sample n_shot positives + n_shot negatives.
    Returns (X_support, y_support, X_query, y_query).
    """
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


def prototypical_predict(
    X_support: np.ndarray,
    y_support: np.ndarray,
    X_query: np.ndarray,
    distance: str = "euclidean",
) -> np.ndarray:
    """
    Compute class prototypes and return probability of class 1 for each query.
    Uses softmax over negative distances.
    """
    if USE_COMBINED or USE_HANDCRAFTED_ONLY:
        scaler = StandardScaler()
        X_support = X_support.copy()
        X_query = X_query.copy()
        X_support[:, :16] = scaler.fit_transform(X_support[:, :16])
        X_query[:, :16]   = scaler.transform(X_query[:, :16])

    proto_pos = X_support[y_support == 1].mean(axis=0)  # (D,)
    proto_neg = X_support[y_support == 0].mean(axis=0)  # (D,)

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

    # softmax over negative distances → probability of class 1
    logit_pos = -d_pos
    logit_neg = -d_neg
    exp_pos = np.exp(logit_pos - np.maximum(logit_pos, logit_neg))
    exp_neg = np.exp(logit_neg - np.maximum(logit_pos, logit_neg))
    proba_pos = exp_pos / (exp_pos + exp_neg)

    return proba_pos


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


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    emb_index  = build_index(EMBEDDINGS_DIR, "_embeddings_labeled.npz")
    feat_index = build_index(FEATURES_DIR, "_features.npz") if (USE_COMBINED or USE_HANDCRAFTED_ONLY) else {}

    recordings = sorted(emb_index.keys())
    print(f"Recordings: {len(recordings)}")
    print(f"Mode: {'Handcrafted + LaBraM' if USE_COMBINED else 'Handcrafted only' if USE_HANDCRAFTED_ONLY else 'LaBraM only'}")
    print(f"Support set: {N_SHOT} pos + {N_SHOT} neg per recording\n")

    rng = np.random.default_rng(RANDOM_SEED)

    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}
    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    fold_metrics = {}

    for base in recordings:
        pid = patient_from_filename(base + "_")

        try:
            X, y = load_recording(base, emb_index, feat_index)

            X_sup, y_sup, X_qry, y_qry = sample_support(X, y, N_SHOT, rng)

            y_proba = prototypical_predict(X_sup, y_sup, X_qry, distance="euclidean")

            if len(np.unique(y_qry)) < 2:
                print(f"[SKIP] {base} — only one class in query set")
                continue

            m = evaluate(y_qry, y_proba)
            fold_metrics[base] = m

            for k in keys:
                collected[k].append(m[k])
            for c in ["tp", "fp", "tn", "fn"]:
                agg[c] += m[c]

            print(
                f"[{pid}] {base} | "
                f"query={len(y_qry)} pos={int(y_qry.sum())} | "
                f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
                f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
                f"roc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}"
            )

        except ValueError as e:
            print(f"[SKIP] {base}: {e}")

    print("\n=== Summary (mean ± std across recordings) ===")
    for k in keys:
        mean, std = summarize_metric(collected[k])
        print(f"{k:>10}: {mean:.4f} ± {std:.4f}")

    print("\n=== Aggregate confusion (at thr=0.4) ===")
    print(f"TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

    worst = sorted(fold_metrics.items(), key=lambda kv: kv[1]["f1"])[:5]
    print("\nWorst by F1:")
    for name, m in worst:
        print(f"  {name}: f1={m['f1']:.3f} (prec={m['precision']:.3f}, rec={m['recall']:.3f})")


if __name__ == "__main__":
    main()