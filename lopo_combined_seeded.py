import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from xgboost import XGBClassifier
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

FEATURES_DIR   = Path("../data/80hz_freq_time_features_cache_basic")
EMBEDDINGS_DIR = Path("../data/labram_classification_1s")

SEED_WEIGHT = 15.0
RANDOM_SEED = 42

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_index(directory: Path, suffix: str) -> dict[str, list[Path]]:
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"No *{suffix} files found in {directory}")

    index: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"  Skipping (no patient id): {f.name}")
            continue
        index.setdefault(pid, []).append(f)
    return index


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_combined(feat_paths: list[Path], emb_paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []

    feat_map = {p.name.replace("_features.npz", ""): p for p in feat_paths}
    emb_map  = {p.name.replace("_embeddings_labeled.npz", ""): p for p in emb_paths}

    common  = sorted(set(feat_map) & set(emb_map))
    missing = set(feat_map) ^ set(emb_map)
    if missing:
        print(f"  WARNING: no match for: {missing}")

    for base in common:
        X_feat, y_feat = load_features(feat_map[base])
        X_emb,  y_emb  = load_embeddings(emb_map[base])

        if len(y_feat) != len(y_emb):
            raise ValueError(
                f"Window count mismatch for {base}: "
                f"features={len(y_feat)}, embeddings={len(y_emb)}"
            )
        if not np.array_equal(y_feat, y_emb):
            raise ValueError(f"Label mismatch for {base}!")

        xs.append(np.hstack([X_feat, X_emb]))
        ys.append(y_feat)

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_xgb(x_train: np.ndarray, y_train: np.ndarray,
              sample_weight: np.ndarray | None = None) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


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
# LOPO CV with seed samples from test patient
# -------------------------------------------------

def main():
    feat_index = build_index(FEATURES_DIR, "_features.npz")
    emb_index  = build_index(EMBEDDINGS_DIR, "_embeddings_labeled.npz")

    patients = sorted(set(feat_index) & set(emb_index))
    print(f"Patients: {patients} (n={len(patients)})")
    print(f"Seed: 5 pos + 5 neg from test patient (weight={SEED_WEIGHT})\n")

    rng = np.random.default_rng(RANDOM_SEED)
    thr = 0.5

    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    fold_metrics: dict[str, dict[str, float]] = {}
    collected = {k: [] for k in keys}
    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}

    for test_pid in patients:
        train_pids = [p for p in patients if p != test_pid]

        # --- training data από τους άλλους ασθενείς ---
        x_train_list, y_train_list = [], []
        for pid in train_pids:
            x, y = load_combined(feat_index[pid], emb_index[pid])
            x_train_list.append(x)
            y_train_list.append(y)

        n_train_other = sum(len(y) for y in y_train_list)

        # --- test patient: όλα τα δεδομένα ---
        x_test_full, y_test_full = load_combined(feat_index[test_pid], emb_index[test_pid])

        # --- seed: 5 τυχαία pos + 5 τυχαία neg από τον test patient ---
        pos_idx = np.where(y_test_full == 1)[0]
        neg_idx = np.where(y_test_full == 0)[0]

        pos_seed = rng.choice(pos_idx, size=min(5, len(pos_idx)), replace=False)
        neg_seed = rng.choice(neg_idx, size=min(5, len(neg_idx)), replace=False)
        seed_idx = np.concatenate([pos_seed, neg_seed])

        # --- test set: όλα εκτός από τα seed ---
        test_mask = np.ones(len(y_test_full), dtype=bool)
        test_mask[seed_idx] = False
        x_test = x_test_full[test_mask]
        y_test = y_test_full[test_mask]

        # --- προσθήκη seed στο training ---
        x_train_list.append(x_test_full[seed_idx])
        y_train_list.append(y_test_full[seed_idx])

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # --- sample weights: 1.0 για τους άλλους, SEED_WEIGHT για τα seed ---
        sample_weight = np.ones(len(y_train), dtype=np.float32)
        sample_weight[n_train_other:] = SEED_WEIGHT

        # --- train & evaluate ---
        model = train_xgb(x_train, y_train, sample_weight=sample_weight)

        score = model.get_booster().get_score(importance_type="gain")
        hc_used     = sum(1 for k in score if int(k[1:]) < 16)
        labram_used = sum(1 for k in score if int(k[1:]) >= 16)
        print(f"  HC features used: {hc_used}/16 | LaBraM features used: {labram_used}/200")

        y_proba = model.predict_proba(x_test)[:, 1]
        m = evaluate_fold(y_test, y_proba, thr=thr)
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

    print("\n=== Aggregate confusion (sum across folds, at thr=0.5) ===")
    print(f"TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

    worst = sorted(fold_metrics.items(), key=lambda kv: kv[1]["f1"])[:5]
    print("\nWorst by F1:")
    for pid, m in worst:
        print(f"  {pid}: f1={m['f1']:.3f} (prec={m['precision']:.3f}, rec={m['recall']:.3f})")

    # --- Plot ---
    patient_pattern = {
        "P20": "LPD", "P28": "LPD", "P36": "LPD",
        "P48": "GPD", "P49": "GPD", "P54": "GPD",
        "P55": "LRDA", "P58": "LPD", "P70": "GPD", "P73": "LPD",
    }
    pattern_colors = {"LPD": "tab:blue", "GPD": "tab:orange", "LRDA": "tab:green"}

    items  = sorted(fold_metrics.items(), key=lambda kv: kv[1]["acc"])
    pids   = [pid for pid, _ in items]
    accs   = [m["acc"] for _, m in items]
    colors = [pattern_colors.get(patient_pattern.get(pid, "UNK"), "gray") for pid in pids]

    plt.figure(figsize=(10, 4))
    plt.bar(pids, accs, color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Patient (sorted)")
    plt.title("LOPO Accuracy per Patient — Handcrafted + LaBraM + Seed (colored by PD pattern)")
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