import re
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

# NOTE: embeddings are written by the extraction script with keys
# "embeddings" / "labels" and a "*_embeddings.npz" suffix.
EMBEDDINGS_DIR = Path("../../../data/labram_embeddings_1s_new")
FEATURES_DIR   = Path("../../../data/80hz_freq_time_features_cache_basic")

RANDOM_SEED  = 42

USE_LABRAM_ONLY      = False
USE_HANDCRAFTED_ONLY = True
USE_COMBINED         = False

# Number of handcrafted features (used to scale the handcrafted block in COMBINED).
N_HC = 16

# Feature modes to sweep in a single run (each becomes one row of subplots).
FEATURE_MODES = ["handcrafted", "labram"]
FEATURE_MODE_TITLE = {"handcrafted": "Handcrafted", "labram": "LaBraM"}


def set_feature_mode(mode: str):
    """Toggle the global feature-mode flags read by load_recording / prototypical_predict."""
    global USE_LABRAM_ONLY, USE_HANDCRAFTED_ONLY, USE_COMBINED
    USE_LABRAM_ONLY      = (mode == "labram")
    USE_HANDCRAFTED_ONLY = (mode == "handcrafted")
    USE_COMBINED         = (mode == "combined")

# -------------------------------------------------
# Active learning config
# -------------------------------------------------

# Total label budget (TOTAL labels, not per class).
# budget=2 is just the seed (1 pos + 1 neg).
BUDGET_RANGE = list(range(2, 41, 2))

# Strategies to compare:
#   "active" = active learning; weighted average of uncertainty + density scores
#   "random" = baseline
STRATEGIES = ["active", "random"]

# Weight of the density term in the combined active score, in [0, 1].
#   w=0   -> pure uncertainty sampling
#   w=1   -> pure density (representativeness) sampling
#   w=0.5 -> equal blend
ACTIVE_DENSITY_WEIGHT = 0.5

N_REPEATS          = 5      # repeats with different seeds -> averaged
TEST_FRAC          = 0.4    # fraction of each class held out as a fixed test set
MIN_TEST_PER_CLASS = 3      # minimum test samples per class
DECISION_THRESHOLD = 0.4    # same threshold as evaluate()

CSV_OUTPUT = Path("results_active_learning.csv")
FIG_OUTPUT = Path("fig8_active_vs_random_paired.pdf")


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
    X = z["embeddings"].astype(np.float32)
    y = z["labels"].astype(np.uint8).ravel()
    return X, y


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples  = X.shape[0]
    # Average across channels -> (N, 16); keep all handcrafted features.
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)
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
        # Standardize only the handcrafted block; LaBraM dims are left as-is.
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


# -------------------------------------------------
# Density helpers (for the "density" acquisition strategy)
# -------------------------------------------------

def scaled_pool_representation(X_pool):
    """Return the pool features in the same scaling regime the prototypes use.

    Density must be measured in the same space as the distance-to-prototype
    classification, so we mirror the per-mode scaling of prototypical_predict.
    """
    if USE_HANDCRAFTED_ONLY:
        return StandardScaler().fit_transform(X_pool).astype(np.float64)
    if USE_COMBINED:
        Xp = X_pool.astype(np.float64).copy()
        Xp[:, :N_HC] = StandardScaler().fit_transform(Xp[:, :N_HC])
        return Xp
    return X_pool.astype(np.float64)  # LaBraM: raw, matching the classifier


def density_scores(X_pool_scaled):
    """Mean inverse-distance density of each pool point w.r.t. the other pool points.

    Higher value => the point sits in a denser region (many nearby instances).
    Uses a 1 / (1 + d) similarity so no kernel bandwidth needs tuning.
    """
    n = len(X_pool_scaled)
    if n <= 1:
        return np.ones(n)
    diff = X_pool_scaled[:, None, :] - X_pool_scaled[None, :, :]
    D = np.linalg.norm(diff, axis=2)          # (n, n) pairwise euclidean
    sim = 1.0 / (1.0 + D)
    np.fill_diagonal(sim, 0.0)
    return sim.sum(axis=1) / (n - 1)


# -------------------------------------------------
# Splits: fixed held-out test  +  unlabeled pool  +  seed (1 pos / 1 neg)
# -------------------------------------------------

def make_splits(y, rng, test_frac, min_test_per_class):
    """Return (test_idx, pool_idx, seed_idx) or None if the recording is too small.

    The test set is fixed and never touched by the active learner.
    The seed is one positive + one negative drawn from the pool so that
    both prototypes are defined from the very first step.
    """
    pos = np.where(y == 1)[0].copy()
    neg = np.where(y == 0)[0].copy()
    rng.shuffle(pos)
    rng.shuffle(neg)

    # Need: test (>=min) + seed (1) + at least 1 extra pool candidate per class.
    if len(pos) < min_test_per_class + 2 or len(neg) < min_test_per_class + 2:
        return None

    n_test_pos = max(min_test_per_class, int(round(test_frac * len(pos))))
    n_test_neg = max(min_test_per_class, int(round(test_frac * len(neg))))

    # Leave at least seed(1) + 1 candidate per class in the pool.
    n_test_pos = min(n_test_pos, len(pos) - 2)
    n_test_neg = min(n_test_neg, len(neg) - 2)

    test_idx = np.concatenate([pos[:n_test_pos], neg[:n_test_neg]])
    pool_pos = pos[n_test_pos:]
    pool_neg = neg[n_test_neg:]

    seed_idx = np.array([pool_pos[0], pool_neg[0]])
    pool_idx = np.concatenate([pool_pos, pool_neg])

    return test_idx, pool_idx, seed_idx


# -------------------------------------------------
# Acquisition: pick the next index to label from the pool
# -------------------------------------------------

def _minmax(v):
    """Min-max normalize a 1D array to [0, 1]; all-equal -> ones."""
    vmin, vmax = v.min(), v.max()
    if vmax > vmin:
        return (v - vmin) / (vmax - vmin)
    return np.ones_like(v)


def select_next(strategy, X, y, labeled, pool, rng):
    """Return the pool index to label next, according to the strategy."""
    if strategy == "random":
        return int(rng.choice(pool))

    if strategy == "active":
        X_pool = X[pool]
        proba = prototypical_predict(X[labeled], y[np.array(labeled)], X_pool)
        # Informativeness: 1.0 at the boundary (proba=0.5), 0.0 when fully confident.
        uncertainty = 1.0 - 2.0 * np.abs(proba - 0.5)
        # Representativeness: density of each point w.r.t. the rest of the pool.
        dens = density_scores(scaled_pool_representation(X_pool))
        # Weighted average of the two (both normalized to [0, 1] so the weight is meaningful).
        w = ACTIVE_DENSITY_WEIGHT
        score = (1.0 - w) * _minmax(uncertainty) + w * _minmax(dens)
        return int(pool[int(np.argmax(score))])

    raise ValueError(f"Unknown strategy: {strategy}")


# -------------------------------------------------
# One active-learning curve for a single recording / repeat / strategy
# -------------------------------------------------

def active_learning_curve(X, y, rng, strategy, budgets, test_idx, pool_idx, seed_idx):
    """Incrementally label samples up to each budget and evaluate on the fixed test set.

    Returns {budget: metrics_dict}. Budgets are processed in increasing order so the
    labeled set is grown once and snapshotted at each target budget.
    """
    X_test, y_test = X[test_idx], y[test_idx]

    labeled = [int(i) for i in seed_idx]
    pool = [int(i) for i in pool_idx if int(i) not in set(labeled)]

    out = {}
    for budget in sorted(budgets):
        # Acquire until we reach this budget (or the pool is exhausted).
        while len(labeled) < budget and pool:
            pick = select_next(strategy, X, y, labeled, pool, rng)
            labeled.append(pick)
            pool.remove(pick)

        # Evaluate current labeled set on the fixed test set.
        proba_test = prototypical_predict(X[labeled], y[np.array(labeled)], X_test)
        out[budget] = evaluate(y_test, proba_test, thr=DECISION_THRESHOLD)
        out[budget]["n_labeled"] = len(labeled)

    return out


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
# Sweep: for each strategy, run all recordings x repeats and aggregate per budget
# -------------------------------------------------

def run_active_learning(recordings, emb_index, feat_index):
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]

    # collected[strategy][budget][metric] = list of values across (recording, repeat)
    collected = {
        s: {b: {k: [] for k in keys} for b in BUDGET_RANGE}
        for s in STRATEGIES
    }
    n_used = {s: {b: 0 for b in BUDGET_RANGE} for s in STRATEGIES}

    # Pre-load all recordings once.
    data = {}
    for base in recordings:
        try:
            data[base] = load_recording(base, emb_index, feat_index)
        except ValueError as e:
            print(f"  [SKIP load] {base}: {e}")

    for repeat in range(N_REPEATS):
        rng = np.random.default_rng(RANDOM_SEED + repeat)

        for base, (X, y) in data.items():
            split = make_splits(y, rng, TEST_FRAC, MIN_TEST_PER_CLASS)
            if split is None:
                if repeat == 0:
                    print(f"  [SKIP small] {base}: not enough samples per class")
                continue
            test_idx, pool_idx, seed_idx = split

            # IMPORTANT: the same split is reused across strategies within this repeat
            # so the comparison is controlled (only the acquisition strategy differs).
            for strategy in STRATEGIES:
                # Each strategy gets its own rng stream derived from the repeat seed,
                # so the random-baseline draws don't depend on the other strategies' calls.
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

    # Aggregate -> rows.
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
# Plotting: 1 figure, 2 subplots (bacc / roc_auc), BOTH representations overlaid
# -------------------------------------------------
# Supervisor's note: put LaBraM and handcrafted on the SAME plot; splitting by
# metric (bacc vs roc_auc) is fine. So each subplot now shows 4 curves with a
# double encoding (same idea as Figure 6):
#     colour     -> representation (handcrafted vs LaBraM)
#     line/marker -> strategy       (active = solid/o, random = dashed/x)

REPR_COLOR = {
    "handcrafted": "tab:blue",
    "labram":      "tab:red",
}
STRATEGY_STYLE = {
    "active": dict(linestyle="-",  marker="o", label="Active (unc.+dens.)"),
    "random": dict(linestyle="--", marker="x", label="Random"),
}


def plot_curves(results, out_path: Path = FIG_OUTPUT):
    metrics = [("bacc", "Balanced accuracy"), ("roc_auc", "ROC AUC")]
    feature_modes = [m for m in FEATURE_MODES if m in results]

    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 5.2), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    for c, (metric, ylabel) in enumerate(metrics):
        ax = axes[c]
        for fmode in feature_modes:
            rows = results[fmode]
            by_strategy = {s: [x for x in rows if x["strategy"] == s] for s in STRATEGIES}
            for strategy in STRATEGIES:
                srows = sorted(by_strategy[strategy], key=lambda x: x["n_shot"])
                xs = np.array([x["n_shot"] for x in srows])
                means = np.array([x[metric] for x in srows], dtype=float)
                stds = np.array([x[f"{metric}_std"] for x in srows], dtype=float)

                col = REPR_COLOR[fmode]
                st = STRATEGY_STYLE[strategy]
                label = f"{FEATURE_MODE_TITLE[fmode]} — {st['label']}"

                ax.plot(
                    xs, means, color=col,
                    linestyle=st["linestyle"], marker=st["marker"], markersize=5,
                    linewidth=1.8, label=label,
                )
                ax.fill_between(xs, means - stds, means + stds, alpha=0.12, color=col)

        ax.set_title(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(alpha=0.3)
        ax.set_xlabel("n_shot (labels per class equivalent)")
        if c == 0:
            ax.set_ylabel("Score")

    # One shared legend below both subplots (4 entries, grouped 2 per row).
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2,
        frameon=True, bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.5, handletextpad=0.6,
    )

    fig.suptitle("Active learning vs random sampling (prototypical few-shot): Handcrafted vs LaBraM")
    # Reserve room for the bottom legend + suptitle (tight_layout ignores fig.legend).
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.22, wspace=0.10)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"\nSaved figure to {out_path} and {out_path.with_suffix('.png')}")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    emb_index  = build_index(EMBEDDINGS_DIR, "_embeddings.npz")
    feat_index = build_index(FEATURES_DIR, "_features.npz")

    # Same recordings for both feature modes -> fair apples-to-apples comparison.
    recordings = sorted(set(emb_index.keys()) & set(feat_index.keys()))
    print(f"Recordings (in both features & embeddings): {len(recordings)}")
    print(f"Feature modes: {FEATURE_MODES}")
    print(f"Strategies: {STRATEGIES}")
    print(f"Budgets: {BUDGET_RANGE}")
    print(f"Repeats: {N_REPEATS}")

    results = {}
    all_rows = []
    for fmode in FEATURE_MODES:
        set_feature_mode(fmode)
        print(f"\n########## Feature mode: {FEATURE_MODE_TITLE[fmode]} ##########")

        rows = run_active_learning(recordings, emb_index, feat_index)
        results[fmode] = rows

        # Print a detailed table per strategy.
        for strategy in STRATEGIES:
            print(f"\n===== {fmode} / {strategy} =====")
            for r in sorted([x for x in rows if x["strategy"] == strategy], key=lambda r: r["budget"]):
                print(
                    f"budget={r['budget']:2d} | folds={r['n_folds']:3d} | "
                    f"acc={r['acc']:.4f} ± {r['acc_std']:.4f} | "
                    f"bacc={r['bacc']:.4f} ± {r['bacc_std']:.4f} | "
                    f"f1={r['f1']:.4f} | "
                    f"roc_auc={r['roc_auc']:.4f} ± {r['roc_auc_std']:.4f}"
                )

        for r in rows:
            all_rows.append({"feature_mode": fmode, **r})

    fieldnames = list(all_rows[0].keys())
    with open(CSV_OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults saved to: {CSV_OUTPUT}")

    plot_curves(results)
    plt.show()


if __name__ == "__main__":
    main()