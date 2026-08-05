import re
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
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

FEATURES_DIR   = Path("../../../data/80hz_freq_time_features_cache_basic")
EMBEDDINGS_DIR = Path("../../../data/labram_embeddings_1s_new")


# -------------------------------------------------
# Config
# -------------------------------------------------

# Both modes are run in a single execution and compared side by side.
MODES = ["handcrafted", "labram"]

N_SHOT_RANGE = list(range(1, 21))   # 1..20 inclusive
THRESHOLD = 0.5


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def build_path_map(directory: Path, suffix: str) -> dict[str, Path]:
    """Map basename → path, e.g. 'P20_GHB_00015_0000348' -> Path(...)."""
    files = sorted(directory.glob(f"*{suffix}"))
    if not files:
        raise RuntimeError(f"No *{suffix} files found in {directory}")
    return {f.name.replace(suffix, ""): f for f in files}


def load_features(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    n_channels = len(z["ch_names"])
    n_samples = X.shape[0]
    X = X.reshape(n_samples, n_channels, -1).mean(axis=1)  # avg across channels → (N, 16)
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path, allow_pickle=True)
    X = z["embeddings"].astype(np.float32)
    y = z["labels"].astype(np.uint8).ravel()
    return X, y


def load_recording(
    base: str,
    mode: str,
    feat_map: dict[str, Path],
    emb_map: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray]:
    """Load one recording for the requested mode."""
    if mode == "handcrafted":
        if base not in feat_map:
            raise ValueError(f"No features for {base}")
        return load_features(feat_map[base])

    if mode == "labram":
        if base not in emb_map:
            raise ValueError(f"No embeddings for {base}")
        return load_embeddings(emb_map[base])

    raise ValueError(f"Unknown mode: {mode}")


# -------------------------------------------------
# Few-shot split: first n_pos + first n_neg as train, rest as test
# -------------------------------------------------

def split_few_shot(X: np.ndarray, y: np.ndarray, n_pos: int, n_neg: int):
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]

    if len(pos_idx) < n_pos or len(neg_idx) < n_neg:
        return None

    train_idx = np.concatenate([pos_idx[:n_pos], neg_idx[:n_neg]])
    train_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask

    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


# -------------------------------------------------
# Model
# -------------------------------------------------

def train_xgb(x_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=30,
        learning_rate=0.05,
        max_depth=1,
        subsample=1,
        colsample_bytree=1,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        # min_child_weight=0 removes the low-sample artefact. The default (1),
        # combined with the logistic-loss Hessian (~0.25 per window at the
        # initial prediction), requires the summed Hessian on each side of a
        # split to be >= 1 (~4 windows per child, ~8 in total), which forbids
        # ANY split below ~8 training windows and collapses the 1-3 shot points
        # to a constant 0.50. Setting it to 0 lets a split form even with very
        # few windows, so those points now reflect the model's actual (weak)
        # low-sample behavior instead of the artefact.
        min_child_weight=0,
    )
    model.fit(x_train, y_train)
    return model


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
# Sweep one N_SHOT value across all files (for one mode)
# -------------------------------------------------

def run_for_n_shot(
    n_shot: int,
    mode: str,
    recordings: list[str],
    feat_map: dict[str, Path],
    emb_map: dict[str, Path],
    threshold: float,
) -> dict[str, float]:
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}
    n_files_used = 0
    n_files_skipped = 0
    agg_tp = agg_fp = agg_tn = agg_fn = 0

    for base in recordings:
        try:
            X, y = load_recording(base, mode, feat_map, emb_map)
        except ValueError:
            n_files_skipped += 1
            continue

        split = split_few_shot(X, y, n_pos=n_shot, n_neg=n_shot)
        if split is None:
            n_files_skipped += 1
            continue

        X_train, y_train, X_test, y_test = split
        if len(y_test) == 0 or len(np.unique(y_test)) < 2:
            n_files_skipped += 1
            continue

        model = train_xgb(X_train, y_train)
        y_proba = model.predict_proba(X_test)[:, 1]

        m = evaluate_fold(y_test, y_proba, thr=threshold)
        for k in keys:
            collected[k].append(m[k])

        agg_tp += m["tp"]; agg_fp += m["fp"]
        agg_tn += m["tn"]; agg_fn += m["fn"]
        n_files_used += 1

    row = {
        "n_shot": n_shot,
        "n_files_used": n_files_used,
        "n_files_skipped": n_files_skipped,
        "tp_sum": int(agg_tp),
        "fp_sum": int(agg_fp),
        "tn_sum": int(agg_tn),
        "fn_sum": int(agg_fn),
    }
    for k in keys:
        mean, std = summarize_metric(collected[k])
        row[k] = mean
        row[f"{k}_std"] = std
    return row


def run_sweep_for_mode(
    mode: str,
    recordings: list[str],
    feat_map: dict[str, Path],
    emb_map: dict[str, Path],
    threshold: float,
) -> list[dict[str, float]]:
    print(f"\n========== Mode: {mode} ==========")
    rows = []
    for n_shot in N_SHOT_RANGE:
        row = run_for_n_shot(n_shot, mode, recordings, feat_map, emb_map, threshold)
        rows.append(row)
        print(
            f"n_shot={n_shot:2d} | files_used={row['n_files_used']:3d} "
            f"(skipped={row['n_files_skipped']:3d}) | "
            f"acc={row['acc']:.4f} ± {row['acc_std']:.4f} | "
            f"bacc={row['bacc']:.4f} ± {row['bacc_std']:.4f} | "
            f"f1={row['f1']:.4f} | "
            f"roc_auc={row['roc_auc']:.4f} ± {row['roc_auc_std']:.4f}"
        )
    return rows


def save_rows(rows: list[dict[str, float]], csv_output: Path):
    fieldnames = list(rows[0].keys())
    with csv_output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {csv_output.resolve()}")


# -------------------------------------------------
# Plotting: 1 figure, 2 subplots (cols = metric); BOTH modes overlaid per axis
# -------------------------------------------------

MODE_TITLES = {"handcrafted": "Handcrafted", "labram": "LaBraM"}
MODE_COLORS = {"handcrafted": "tab:orange", "labram": "tab:blue"}

# (metric key, axis label) for each plotted column
PLOT_METRICS = [
    ("bacc",    "Balanced accuracy"),
    ("roc_auc", "ROC AUC"),
]

FIG_OUTPUT = Path("fig7_xgb_fewshot_paired.pdf")


def plot_all(results: dict[str, list[dict]], out_path: Path = FIG_OUTPUT):
    """Single figure, 1x2: columns = metric (balanced accuracy, ROC AUC).
    Handcrafted and LaBraM are drawn on the SAME axes (colour = representation)."""
    fig, axes = plt.subplots(
        1, len(PLOT_METRICS),
        figsize=(12, 5.2),
        sharex=True, sharey=True,
    )
    axes = np.atleast_1d(axes)

    xticks = None
    for c, (metric, ylabel) in enumerate(PLOT_METRICS):
        ax = axes[c]
        for mode in MODES:
            rows = results[mode]
            xs = np.array([row["n_shot"] for row in rows])
            means = np.array([row[metric] for row in rows], dtype=float)
            stds = np.array([row[f"{metric}_std"] for row in rows], dtype=float)

            color = MODE_COLORS[mode]
            ax.plot(xs, means, marker="o", markersize=4, linewidth=1.8,
                    color=color, label=MODE_TITLES[mode])
            ax.fill_between(xs, means - stds, means + stds, alpha=0.18, color=color)
            xticks = xs

        ax.set_title(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.set_xlabel("n_shot (labels per class)")
        if xticks is not None:
            ax.set_xticks(xticks[::2])
        if c == 0:
            ax.set_ylabel("Score")

    # One shared legend below both subplots.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2,
        frameon=True, bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.8, handletextpad=0.6,
    )

    fig.suptitle("Few-shot XGBoost baseline: Handcrafted vs LaBraM")
    # Reserve room for the bottom legend (tight_layout ignores fig.legend).
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.20, wspace=0.10)

    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"\nFigure saved to: {out_path} and {out_path.with_suffix('.png')}")


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    print(f"Threshold: {THRESHOLD}")

    feat_map = build_path_map(FEATURES_DIR,   "_features.npz")
    emb_map  = build_path_map(EMBEDDINGS_DIR, "_embeddings.npz")

    # Same recordings for both modes -> fair apples-to-apples comparison
    recordings = sorted(set(feat_map.keys()) & set(emb_map.keys()))
    print(f"Recordings (in both features & embeddings): {len(recordings)}")
    print(f"Sweeping N_SHOT: {N_SHOT_RANGE}")

    results: dict[str, list[dict]] = {}
    for mode in MODES:
        rows = run_sweep_for_mode(mode, recordings, feat_map, emb_map, THRESHOLD)
        results[mode] = rows
        save_rows(rows, Path(f"results_xgb_fewshot_{mode}.csv"))

    plot_all(results)
    plt.show()


if __name__ == "__main__":
    main()