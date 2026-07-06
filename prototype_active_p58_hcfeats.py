import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
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
# Labeled handcrafted features produced by the feature-extraction script
# (keys: X, y [1/0/-1], segment_type, ch_names, ...).
FEATURES_DIR = Path("../../../data/evaluation_recordings/features_1s_labeled_persegment")

CSV_OUTPUT = Path("results_active_learning_eval_handcrafted.csv")

RANDOM_SEED = 42

# -------------------------------------------------
# Active learning config
# -------------------------------------------------

BUDGET_RANGE = list(range(2, 41, 2))     # total labels; n_shot = budget // 2
STRATEGIES = ["active", "random"]
ACTIVE_DENSITY_WEIGHT = 0.5              # weight of density vs uncertainty in [0, 1]

N_REPEATS          = 5
TEST_FRAC          = 0.3
MIN_TEST_PER_CLASS = 1
DECISION_THRESHOLD = 0.4

IGNORE_LABEL = -1
PD, NON_PD = "PD", "NON_PD"

# If True, the active-learning POOL (the windows eligible to be picked for
# labeling) contains negatives from NON_PD segments only. The TEST set is
# unaffected and keeps ALL negatives (PD-segment + NON_PD) for a realistic
# evaluation distribution. This restriction lives in make_splits and applies
# equally to the active and random strategies (both draw from the same pool).
POOL_NEG_NONPD_ONLY = True


# -------------------------------------------------
# Loading: handcrafted features, averaged across channels, ignore dropped
# -------------------------------------------------

def load_labeled_recording(path: Path):
    """Return (X, y, seg) with ignored windows removed.

    X   : (N, 16) handcrafted features averaged across channels
    y   : (N,)    binary labels (1 = discharge, 0 = background)
    seg : (N,)    segment type per window ("PD" / "NON_PD")

    All windows are kept here (including PD-segment negatives); the NON_PD
    pool restriction is applied later, in make_splits.
    """
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # (N, 16)

    labels = z["y"].astype(np.int64).ravel()
    seg = np.asarray([str(s).upper() for s in z["segment_type"]])

    keep = labels != IGNORE_LABEL
    return X[keep], labels[keep].astype(np.uint8), seg[keep]


# -------------------------------------------------
# Prototypical classifier (handcrafted features -> standardized)
# -------------------------------------------------

def prototypical_predict(X_support, y_support, X_query, distance="euclidean"):
    # Standardize: handcrafted features live on very different scales.
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

    logit_pos = -d_pos
    logit_neg = -d_neg
    exp_pos = np.exp(logit_pos - np.maximum(logit_pos, logit_neg))
    exp_neg = np.exp(logit_neg - np.maximum(logit_pos, logit_neg))
    return exp_pos / (exp_pos + exp_neg)


# -------------------------------------------------
# Density helper (efficient; standardized space)
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
# Splits: fixed test + pool + seed; first NEGATIVE seed from a NON_PD segment
# -------------------------------------------------

def make_splits(y, seg, rng, test_frac, min_test_per_class):
    pos = np.where(y == 1)[0].copy()
    neg = np.where(y == 0)[0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)

    if len(pos) < min_test_per_class + 2 or len(neg) < min_test_per_class + 2:
        return None

    n_test_pos = max(min_test_per_class, int(round(test_frac * len(pos))))
    n_test_neg = max(min_test_per_class, int(round(test_frac * len(neg))))
    n_test_pos = min(n_test_pos, len(pos) - 2)
    n_test_neg = min(n_test_neg, len(neg) - 2)

    # TEST set uses ALL negatives (PD-segment + NON_PD) -> realistic distribution.
    test_idx = np.concatenate([pos[:n_test_pos], neg[:n_test_neg]])
    pool_pos = pos[n_test_pos:]
    leftover_neg = neg[n_test_neg:]

    # POOL negatives: only NON_PD-segment ones are eligible to be labeled.
    # Leftover PD-segment negatives are neither in test nor selectable -> dropped
    # from the pool. Applies equally to active and random (shared pool).
    if POOL_NEG_NONPD_ONLY:
        pool_neg = np.array([int(i) for i in leftover_neg if seg[i] == NON_PD], dtype=int)
    else:
        pool_neg = leftover_neg

    # Need at least one selectable positive and one NON_PD negative for the seeds.
    if len(pool_pos) < 1 or len(pool_neg) < 1:
        return None

    pos_seed = int(pool_pos[0])
    neg_seed = int(pool_neg[0])   # guaranteed NON_PD when POOL_NEG_NONPD_ONLY

    seed_idx = np.array([pos_seed, neg_seed])
    pool_idx = np.concatenate([pool_pos, pool_neg])
    return test_idx, pool_idx, seed_idx


# -------------------------------------------------
# Acquisition
# -------------------------------------------------

def select_next(strategy, X, y, labeled, pool, rng, dens_lookup=None):
    if strategy == "random":
        return int(rng.choice(pool))

    if strategy == "active":
        X_pool = X[pool]
        proba = prototypical_predict(X[labeled], y[np.array(labeled)], X_pool)
        uncertainty = 1.0 - 2.0 * np.abs(proba - 0.5)
        dens = dens_lookup[np.array(pool)]
        w = ACTIVE_DENSITY_WEIGHT
        score = (1.0 - w) * _minmax(uncertainty) + w * _minmax(dens)
        return int(pool[int(np.argmax(score))])

    raise ValueError(f"Unknown strategy: {strategy}")


# -------------------------------------------------
# One curve for a single recording / repeat / strategy
# -------------------------------------------------

def active_learning_curve(X, y, rng, strategy, budgets, test_idx, pool_idx, seed_idx):
    X_test, y_test = X[test_idx], y[test_idx]

    labeled = [int(i) for i in seed_idx]
    pool = [int(i) for i in pool_idx if int(i) not in set(labeled)]

    # Precompute density ONCE on the initial pool (standardized), not per step.
    dens_lookup = None
    if strategy == "active":
        p0 = np.array(pool)
        Xp = StandardScaler().fit_transform(X[p0]).astype(np.float64)
        dens_lookup = np.zeros(len(X), dtype=np.float64)
        dens_lookup[p0] = density_scores(Xp)

    out = {}
    for budget in sorted(budgets):
        while len(labeled) < budget and pool:
            pick = select_next(strategy, X, y, labeled, pool, rng, dens_lookup)
            labeled.append(pick)
            pool.remove(pick)

        proba_test = prototypical_predict(X[labeled], y[np.array(labeled)], X_test)
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

        for stem, (X, y, seg) in recordings.items():
            split = make_splits(y, seg, rng, TEST_FRAC, MIN_TEST_PER_CLASS)
            if split is None:
                if repeat == 0:
                    n_pos = int((y == 1).sum())
                    n_neg = int((y == 0).sum())
                    print(f"  [SKIP small] {stem}: pos={n_pos}, neg={n_neg} "
                          f"(need >= {MIN_TEST_PER_CLASS + 2} per class)")
                continue
            test_idx, pool_idx, seed_idx = split

            for strategy in STRATEGIES:
                strat_rng = np.random.default_rng(
                    RANDOM_SEED + 1000 * repeat + hash(strategy) % 997
                )
                curve = active_learning_curve(
                    X, y, strat_rng, strategy,
                    BUDGET_RANGE, test_idx, pool_idx, seed_idx,
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
    "active": dict(color="tab:orange", label="Active (uncertainty + density)"),
    "random": dict(color="tab:gray",   label="Random baseline"),
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

    fig.suptitle("Active learning vs random — evaluation recording (handcrafted features)")
    fig.tight_layout()


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    feature_files = sorted(FEATURES_DIR.glob("*_features.npz"))
    if not feature_files:
        raise RuntimeError(f"No *_features.npz files in {FEATURES_DIR}")

    recordings = {}
    for path in feature_files:
        stem = path.name.replace("_features.npz", "")
        X, y, seg = load_labeled_recording(path)
        recordings[stem] = (X, y, seg)
        n_nonpd_neg = int(np.sum((y == 0) & (seg == NON_PD)))
        print(f"{stem}: windows={len(y)} | pos={int(y.sum())} | neg={int((y == 0).sum())} "
              f"(NON_PD negs={n_nonpd_neg})")

    print(f"\nStrategies: {STRATEGIES} | Repeats: {N_REPEATS} | Budgets: {BUDGET_RANGE}")
    print(f"Pool negatives NON_PD-only (test keeps all negatives): {POOL_NEG_NONPD_ONLY}\n")

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