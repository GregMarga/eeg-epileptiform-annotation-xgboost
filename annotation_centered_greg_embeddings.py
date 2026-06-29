import re
from pathlib import Path
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
# Config
# -------------------------------------------------

# Άλλαξε το path αν χρειάζεται. Σταθερό σημείο για να το πειράζεις εύκολα.
IN_DIR = Path("../../../data/labram_embeddings_2s_new")

# Glob pattern για τα npz σου. Αν τα ονόματα έχουν συγκεκριμένο suffix
# (π.χ. "*_embeddings.npz") βάλ' το εδώ· αλλιώς "*.npz" τα πιάνει όλα.
FILE_GLOB = "*.npz"

N_FEATURES = 200       # LaBraM-base embedding dim
THRESHOLD = 0.5

# Πιθανά ονόματα keys μέσα στα npz (ανεκτικός loader)
FEATURE_KEYS = ("X", "embeddings", "features", "emb")
LABEL_KEYS = ("y", "labels", "label")
WINDOW_KEYS = ("windows",)
SOURCE_KEYS = ("source_edf", "edf_name", "edf")


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    """
    Extract patient id from filenames like:
    P20_GHB_00015_0000348_embeddings.npz -> 'P20'
    """
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def _first_present(candidates: tuple[str, ...], available: set[str]) -> str | None:
    for k in candidates:
        if k in available:
            return k
    return None


def load_features(npz_path: Path):
    """
    Robust loader. Required: a features array (2D) and a labels array.
    Optional: windows, source_edf (loaded if present, otherwise None).
    """
    z = np.load(npz_path, allow_pickle=True)
    keys = set(z.files)

    xk = _first_present(FEATURE_KEYS, keys)
    yk = _first_present(LABEL_KEYS, keys)

    if xk is None or yk is None:
        raise KeyError(
            f"Could not find feature/label keys in {npz_path.name}. "
            f"Looked for features in {FEATURE_KEYS} and labels in {LABEL_KEYS}. "
            f"Found keys: {sorted(keys)}"
        )

    X = np.asarray(z[xk]).astype(np.float32)
    y = np.asarray(z[yk]).astype(np.uint8).ravel()

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape} in {npz_path.name}")

    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch in {npz_path.name}: len(X)={len(X)} vs len(y)={len(y)}"
        )

    wk = _first_present(WINDOW_KEYS, keys)
    sk = _first_present(SOURCE_KEYS, keys)
    windows = z[wk] if wk is not None else None
    source_edf = z[sk] if sk is not None else None

    if windows is not None and len(windows) != len(y):
        raise ValueError(
            f"Window mismatch in {npz_path.name}: "
            f"len(windows)={len(windows)} vs len(y)={len(y)}"
        )

    return X, y, windows, source_edf


def build_patient_index(in_dir: Path) -> dict[str, list[Path]]:
    files = sorted(in_dir.glob(FILE_GLOB))
    if not files:
        raise RuntimeError(f"No {FILE_GLOB} files found in {in_dir}")

    patient_files: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        patient_files.setdefault(pid, []).append(f)

    if not patient_files:
        raise RuntimeError("No patient files parsed. Check filename pattern.")

    return patient_files


def concat_patient_files(file_list: list[Path]):
    xs, ys = [], []
    for f in file_list:
        x, y, _, _ = load_features(f)
        xs.append(x)
        ys.append(y)

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
    )


def train_xgb(x_train: np.ndarray, y_train: np.ndarray):
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
    model.fit(x_train, y_train)
    score = model.get_booster().get_score(importance_type="gain")

    return model, score


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_proba))


def safe_pr_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_proba))


def evaluate_fold(y_true: np.ndarray, y_proba: np.ndarray, thr: float = 0.5) -> dict[str, float]:
    y_pred = (y_proba >= thr).astype(np.uint8)

    acc = float(accuracy_score(y_true, y_pred))
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    ra = safe_roc_auc(y_true, y_proba)
    pa = safe_pr_auc(y_true, y_proba)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "acc": acc,
        "bacc": bacc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "roc_auc": ra,
        "pr_auc": pa,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def summarize_metric(values: list[float]) -> tuple[float, float]:
    a = np.array(values, dtype=float)
    return float(np.nanmean(a)), float(np.nanstd(a))


# -------------------------------------------------
# LOPO CV
# -------------------------------------------------

def main():
    patient_files = build_patient_index(IN_DIR)
    patients = sorted(patient_files.keys())

    print(f"Patients: {patients} (n={len(patients)})")

    thr = THRESHOLD

    fold_metrics: dict[str, dict[str, float]] = {}
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}

    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    all_importances = []

    for test_pid in patients:
        train_pids = [p for p in patients if p != test_pid]

        x_train_list, y_train_list = [], []
        for pid in train_pids:
            x, y = concat_patient_files(patient_files[pid])
            x_train_list.append(x)
            y_train_list.append(y)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        x_test, y_test = concat_patient_files(patient_files[test_pid])

        model, score = train_xgb(x_train, y_train)

        imp_vec = np.zeros(N_FEATURES)
        for k, v in score.items():
            idx = int(k[1:])  # "f6" -> 6
            if idx < N_FEATURES:
                imp_vec[idx] = v
        all_importances.append(imp_vec)

        y_proba = model.predict_proba(x_test)[:, 1]

        m = evaluate_fold(y_test, y_proba, thr=thr)
        fold_metrics[test_pid] = m

        for k in keys:
            collected[k].append(m[k])
        for c in ["tp", "fp", "tn", "fn"]:
            agg[c] += m[c]

        print(
            f"[{test_pid}] "
            f"n={len(y_test)} pos={int(y_test.sum())} "
            f"acc={m['acc']:.3f} bacc={m['bacc']:.3f} f1={m['f1']:.3f} "
            f"prec={m['precision']:.3f} rec={m['recall']:.3f} "
            f"roc_auc={m['roc_auc']:.3f} pr_auc={m['pr_auc']:.3f}"
        )

    print("\n=== LOPO summary (mean +/- std across patients) ===")
    for k in keys:
        mean, std = summarize_metric(collected[k])
        print(f"{k:>10}: {mean:.4f} +/- {std:.4f}")

    print("\n=== Aggregate confusion (sum across folds, at thr=0.5) ===")
    print(f"TP={int(agg['tp'])}  FP={int(agg['fp'])}  TN={int(agg['tn'])}  FN={int(agg['fn'])}")

    worst = sorted(fold_metrics.items(), key=lambda kv: kv[1]["f1"])[:5]
    print("\nWorst by F1:")
    for pid, m in worst:
        print(f"  {pid}: f1={m['f1']:.3f} (prec={m['precision']:.3f}, rec={m['recall']:.3f})")

    all_importances = np.array(all_importances)  # (n_folds, n_features)
    mean_imp = all_importances.mean(axis=0)
    std_imp = all_importances.std(axis=0)
    idx = np.argsort(mean_imp)[::-1]

    # -------------------------------------------------
    # Per-patient accuracy plot, colored by PD pattern
    # -------------------------------------------------
    patient_pattern = {
        "P20": "LPD",
        "P28": "LPD",
        "P36": "LPD",
        "P48": "GPD",
        "P49": "GPD",
        "P54": "GPD",
        "P55": "LRDA",
        "P58": "LPD",
        "P70": "GPD",
        "P73": "LPD",
    }
    pattern_colors = {
        "LPD": "tab:blue",
        "GPD": "tab:orange",
        "LRDA": "tab:green",
    }

    items = sorted(fold_metrics.items(), key=lambda kv: kv[1]["acc"])
    pids = [pid for pid, _ in items]
    accs = [m["acc"] for _, m in items]

    colors = [
        pattern_colors.get(patient_pattern.get(pid, "UNK"), "gray")
        for pid in pids
    ]

    plt.figure(figsize=(10, 4))
    plt.bar(pids, accs, color=colors)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Patient (sorted)")
    plt.title("LOPO Accuracy per Patient (colored by PD pattern)")
    plt.grid(axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_elems = [
        Patch(facecolor="tab:blue", label="LPD"),
        Patch(facecolor="tab:orange", label="GPD"),
        Patch(facecolor="tab:green", label="LRDA"),
    ]
    plt.legend(handles=legend_elems, title="Pattern")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()