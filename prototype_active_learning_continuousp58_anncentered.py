import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
# POOL positives : annotation-centered '*' embeddings (positives only)
#     "{stem}_embeddings.npz" in the annot-centered embeddings folder.
# TEST + POOL negatives : labeled sliding embeddings
#     "{stem}_embeddings_labeled.npz" in the "labeled" folder.
#
# POOL is built from TWO sources:
#   positives -> annotation-centered '*' embeddings (clean, mark at center)
#   negatives -> NON_PD sliding embeddings (nothing to center on background)
# TEST = ALL sliding embeddings (realistic continuous scan).
#
# Note: pool negatives are a subset of the sliding TEST embeddings, so the few
# labeled negatives (<= budget/2) overlap the test set -> negligible leakage.
POOL_DIR = Path("../../../data/evaluation_recordings/labram_annot_centered_windows_1s_embeddings")
TEST_DIR = Path("../../../data/evaluation_recordings/labeled")
POOL_SUFFIX = "_embeddings.npz"
TEST_SUFFIX = "_embeddings_labeled.npz"

CSV_OUTPUT = Path("results_active_learning_labram_annotpool_slidingtest.csv")

RANDOM_SEED = 42

# -------------------------------------------------
# Active learning config
# -------------------------------------------------

BUDGET_RANGE = list(range(2, 41, 2))     # total labels; n_shot = budget // 2
STRATEGIES = ["active", "random"]
ACTIVE_DENSITY_WEIGHT = 0.5

N_REPEATS          = 5                    # varies seed + acquisition path (test is fixed)
DECISION_THRESHOLD = 0.4

IGNORE_LABEL = -1
PD, NON_PD = "PD", "NON_PD"

# If True, the active-learning POOL keeps negatives from NON_PD segments only.
# The TEST set (all sliding embeddings) keeps ALL negatives.
POOL_NEG_NONPD_ONLY = True


# -------------------------------------------------
# Loading: LaBraM embeddings, drop ignored windows, keep segment_type.
# Works for both the annotation-centered pool file and the sliding test file.
# -------------------------------------------------

def load_labeled_recording(path: Path):
    """Return (X, y, seg) with ignored windows removed.

    X   : (N, D) LaBraM embeddings
    y   : (N,)   binary labels (1 = discharge, 0 = background)
    seg : (N,)   segment type per window ("PD" / "NON_PD")
    """
    z = np.load(path, allow_pickle=True)
    X = z["embeddings"].astype(np.float32)
    labels = z["labels"].astype(np.int64).ravel()
    seg = np.asarray([str(s).upper() for s in z["segment_type"]])

    keep = labels != IGNORE_LABEL
    return X[keep], labels[keep].astype(np.uint8), seg[keep]


# -------------------------------------------------
# Prototypical classifier (LaBraM embeddings, raw / no scaling).
# NOTE: prototypes are built from the POOL (annotation-centered) and applied to
# the TEST (sliding). Pool and test come from different windowings, so a mild
# covariate shift is expected by design.
# -------------------------------------------------

def prototypical_predict(X_support, y_support, X_query, distance="euclidean"):
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


# -------------------------------------------------
# Density helper (raw embedding space, matching the classifier)
# -------------------------------------------------

def density_scores(X_pool):
    """Mean inverse-distance density using ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b."""
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


# -------------------------------------------------
# Pool construction (indices refer to the POOL arrays). No test split: the whole
# sliding set is the test. Negatives restricted to NON_PD; seed = 1 pos + 1 neg.
# -------------------------------------------------

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


# -------------------------------------------------
# Acquisition (indices refer to the POOL arrays)
# -------------------------------------------------

def select_next(strategy, X_pool, y_pool, labeled, pool, rng, dens_lookup=None):
    if strategy == "random":
        return int(rng.choice(pool))

    if strategy == "active":
        proba = prototypical_predict(X_pool[labeled], y_pool[np.array(labeled)], X_pool[pool])
        uncertainty = 1.0 - 2.0 * np.abs(proba - 0.5)
        dens = dens_lookup[np.array(pool)]
        w = ACTIVE_DENSITY_WEIGHT
        score = (1.0 - w) * _minmax(uncertainty) + w * _minmax(dens)
        return int(pool[int(np.argmax(score))])

    raise ValueError(f"Unknown strategy: {strategy}")


# -------------------------------------------------
# One curve: label from the POOL, evaluate on the (fixed) sliding TEST
# -------------------------------------------------

def active_learning_curve(X_pool, y_pool, X_test, y_test, rng, strategy,
                          budgets, pool_idx, seed_idx):
    labeled = [int(i) for i in seed_idx]
    pool = [int(i) for i in pool_idx if int(i) not in set(labeled)]

    dens_lookup = None
    if strategy == "active":
        p0 = np.array(pool)
        dens_lookup = np.zeros(len(X_pool), dtype=np.float64)
        dens_lookup[p0] = density_scores(X_pool[p0].astype(np.float64))

    out = {}
    for budget in sorted(budgets):
        while len(labeled) < budget and pool:
            pick = select_next(strategy, X_pool, y_pool, labeled, pool, rng, dens_lookup)
            labeled.append(pick)
            pool.remove(pick)

        proba_test = prototypical_predict(X_pool[labeled], y_pool[np.array(labeled)], X_test)
        out[budget] = evaluate(y_test, proba_test, thr=DECISION_THRESHOLD)
        out[budget]["n_labeled"] = len(labeled)
    return out


# -------------------------------------------------
# Metrics
# -------------------------------------------------

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


# -------------------------------------------------
# Sweep across recordings x repeats, aggregate per budget
# -------------------------------------------------

def run_active_learning(recordings):
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {s: {b: {k: [] for k in keys} for b in BUDGET_RANGE} for s in STRATEGIES}
    n_used = {s: {b: 0 for b in BUDGET_RANGE} for s in STRATEGIES}

    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(RANDOM_SEED + repeat)

        for stem, (X_pool, y_pool, seg_pool, X_test, y_test) in recordings.items():
            built = make_pool(y_pool, seg_pool, rng)
            if built is None:
                if repeat == 0:
                    print(f"  [SKIP] {stem}: pool needs >= 1 positive and "
                          f">= 1 NON_PD negative")
                continue
            pool_idx, seed_idx = built

            for strategy in STRATEGIES:
                strat_rng = np.random.default_rng(
                    RANDOM_SEED + 1000 * repeat + hash(strategy) % 997
                )
                curve = active_learning_curve(
                    X_pool, y_pool, X_test, y_test, strat_rng, strategy,
                    BUDGET_RANGE, pool_idx, seed_idx,
                )
                for b, m in curve.items():
                    for k in keys:
                        collected[strategy][b][k].append(m[k])
                    n_used[strategy][b] += 1

    rows = []
    for strategy in STRATEGIES:
        for b in BUDGET_RANGE:
            row = {"strategy": strategy, "budget": b, "n_shot": b // 2,
                   "n_folds": n_used[strategy][b]}
            for k in keys:
                vals = collected[strategy][b][k]
                row[k] = float(np.nanmean(vals)) if vals else float("nan")
                row[f"{k}_std"] = float(np.nanstd(vals)) if vals else float("nan")
            rows.append(row)
    return rows


# -------------------------------------------------
# Plotting: 1 figure, 2 subplots (bacc / roc_auc vs n_shot)
# -------------------------------------------------

STRATEGY_STYLE = {
    "active": dict(color="tab:blue", label="Active (uncertainty + density)"),
    "random": dict(color="tab:gray", label="Random baseline"),
}


def plot_curves(rows):
    metrics = [("bacc", "Balanced accuracy"), ("roc_auc", "ROC AUC")]
    by_strategy = {s: [x for x in rows if x["strategy"] == s] for s in STRATEGIES}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    for ax, (metric, ylabel) in zip(axes, metrics):
        for strategy in STRATEGIES:
            srows = sorted(by_strategy[strategy], key=lambda x: x["n_shot"])
            xs = np.array([x["n_shot"] for x in srows])
            means = np.array([x[metric] for x in srows], dtype=float)
            stds = np.array([x[f"{metric}_std"] for x in srows], dtype=float)

            style = STRATEGY_STYLE[strategy]
            ax.plot(xs, means, marker="o", color=style["color"], label=style["label"])
            ax.fill_between(xs, means - stds, means + stds, alpha=0.15, color=style["color"])

        ax.set_title(ylabel)
        ax.set_xlabel("n_shot (labels per class equivalent)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(alpha=0.3)
        ax.legend()

    fig.suptitle("Active learning vs random — pool: annotation-centered, "
                 "test: sliding (LaBraM embeddings)")
    fig.tight_layout()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    pool_files = sorted(POOL_DIR.glob(f"*{POOL_SUFFIX}"))
    if not pool_files:
        raise RuntimeError(f"No *{POOL_SUFFIX} files in {POOL_DIR}")

    recordings = {}
    for pool_path in pool_files:
        stem = pool_path.name.replace(POOL_SUFFIX, "")
        test_path = TEST_DIR / f"{stem}{TEST_SUFFIX}"
        if not test_path.exists():
            print(f"  [SKIP] no sliding test file for {stem} ({test_path.name})")
            continue

        # Positives: annotation-centered '*' embeddings.
        X_ann, y_ann, seg_ann = load_labeled_recording(pool_path)
        pos_mask = y_ann == 1

        # Sliding set: used for the TEST (all windows) and as the source of the
        # POOL negatives (segment filtering to NON_PD happens in make_pool).
        X_sl, y_sl, seg_sl = load_labeled_recording(test_path)
        neg_mask = y_sl == 0

        # Build the pool: annot positives + sliding negatives.
        X_pool = np.concatenate([X_ann[pos_mask], X_sl[neg_mask]], axis=0)
        y_pool = np.concatenate([
            np.ones(int(pos_mask.sum()), dtype=np.uint8),
            np.zeros(int(neg_mask.sum()), dtype=np.uint8),
        ])
        seg_pool = np.concatenate([seg_ann[pos_mask], seg_sl[neg_mask]])

        # Test = all sliding embeddings.
        X_test, y_test = X_sl, y_sl
        recordings[stem] = (X_pool, y_pool, seg_pool, X_test, y_test)

        pool_nonpd_neg = int(np.sum((y_pool == 0) & (seg_pool == NON_PD)))
        print(f"{stem}:")
        print(f"  POOL: n={len(y_pool)} pos={int(y_pool.sum())} (annot-centered) "
              f"neg={int((y_pool == 0).sum())} sliding (NON_PD negs={pool_nonpd_neg})")
        print(f"  TEST (sliding): n={len(y_test)} pos={int(y_test.sum())} "
              f"neg={int((y_test == 0).sum())}")

    if not recordings:
        raise RuntimeError("No recordings with both pool and test files.")

    print(f"\nStrategies: {STRATEGIES} | Repeats: {N_REPEATS} | Budgets: {BUDGET_RANGE}")
    print(f"Pool negatives NON_PD-only: {POOL_NEG_NONPD_ONLY} | "
          f"Test = all sliding windows\n")

    rows = run_active_learning(recordings)

    for strategy in STRATEGIES:
        print(f"\n===== {strategy} =====")
        for r in sorted([x for x in rows if x["strategy"] == strategy], key=lambda r: r["n_shot"]):
            print(
                f"n_shot={r['n_shot']:2d} | folds={r['n_folds']:3d} | "
                f"bacc={r['bacc']:.4f} ± {r['bacc_std']:.4f} | "
                f"roc_auc={r['roc_auc']:.4f} ± {r['roc_auc_std']:.4f} | "
                f"f1={r['f1']:.4f}"
            )

    fieldnames = list(rows[0].keys())
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults saved to: {CSV_OUTPUT}")

    plot_curves(rows)
    plt.show()


if __name__ == "__main__":
    main()