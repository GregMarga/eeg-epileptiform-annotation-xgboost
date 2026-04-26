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
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    """
    Extract patient id from filenames like:
    P20_GHB_00015_0000348_embeddings_labeled.npz -> 'P20'
    """
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_features(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)

    if not isinstance(z, np.lib.npyio.NpzFile):
        raise TypeError(f"Expected .npz file, got {type(z)} for {npz_path}")

    required = {"X", "y", "windows", "source_edf"}
    keys = set(z.files)
    if not required.issubset(keys):
        raise KeyError(
            f"Missing required keys in {npz_path}. "
            f"Required={sorted(required)}, found={sorted(keys)}"
        )

    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.uint8).ravel()
    windows = z["windows"]
    source_edf = z["source_edf"]

    if X.ndim != 2:
        raise ValueError(f"Expected X to be 2D, got shape {X.shape} in {npz_path}")

    if len(X) != len(y):
        raise ValueError(
            f"Row mismatch in {npz_path}: len(X)={len(X)} vs len(y)={len(y)}"
        )

    if len(windows) != len(y):
        raise ValueError(
            f"Window mismatch in {npz_path}: len(windows)={len(windows)} vs len(y)={len(y)}"
        )

    return X, y, windows, source_edf


def build_patient_index(in_dir: Path) -> dict[str, list[Path]]:
    files = sorted(in_dir.glob("*_embeddings_labeled.npz"))
    if not files:
        raise RuntimeError(f"No *_embeddings_labeled.npz files found in {in_dir}")

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


def concat_patient_files(
        file_list: list[Path]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws, edfs = [], [], [], []

    for f in file_list:
        x, y, windows, source_edf = load_features(f)
        xs.append(x)
        ys.append(y)
        ws.append(windows)
        edfs.append(source_edf)

    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.concatenate(ws, axis=0),
        np.concatenate(edfs, axis=0),
    )


def train_xgb(x_train: np.ndarray, y_train: np.ndarray) -> tuple[XGBClassifier, dict[str, float | list[float]]]:
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

def plot_probability_histograms(
    y_train: np.ndarray,
    y_proba_train: np.ndarray,
    y_test: np.ndarray,
    y_proba_test: np.ndarray,
    test_pid: str,
):
    plt.figure(figsize=(10, 4))
    plt.hist(y_proba_train[y_train == 0], bins=30, alpha=0.5, density=True, label="Train y=0")
    plt.hist(y_proba_train[y_train == 1], bins=30, alpha=0.5, density=True, label="Train y=1")
    plt.xlabel("Predicted probability for class 1")
    plt.ylabel("Density")
    plt.title(f"Train probability distribution - {test_pid}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.hist(y_proba_test[y_test == 0], bins=30, alpha=0.5, density=True, label="Test y=0")
    plt.hist(y_proba_test[y_test == 1], bins=30, alpha=0.5, density=True, label="Test y=1")
    plt.xlabel("Predicted probability for class 1")
    plt.ylabel("Density")
    plt.title(f"Test probability distribution - {test_pid}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


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


def format_window(window) -> str:
    """
    Robust string formatting for a window entry.
    Works whether each window is scalar, vector, list, or ndarray.
    """
    w = np.asarray(window)

    if w.ndim == 0:
        return str(w.item())

    flat = w.ravel()
    return "[" + ", ".join(str(v) for v in flat) + "]"


# -------------------------------------------------
# LOPO CV
# -------------------------------------------------

def main():
    in_dir = Path("../../../Data/labram_classification_1s")
    patient_files = build_patient_index(in_dir)
    patients = sorted(patient_files.keys())

    print(f"Patients: {patients} (n={len(patients)})")

    thr = 0.5

    fold_metrics: dict[str, dict[str, float]] = {}
    keys = ["acc", "bacc", "f1", "precision", "recall", "roc_auc", "pr_auc"]
    collected = {k: [] for k in keys}

    agg = {"tp": 0.0, "fp": 0.0, "tn": 0.0, "fn": 0.0}
    n_features = 200
    all_importances = []
    for test_pid in patients:
        train_pids = [p for p in patients if p != test_pid]

        x_train_list, y_train_list = [], []
        for pid in train_pids:
            x, y, _, _ = concat_patient_files(patient_files[pid])
            x_train_list.append(x)
            y_train_list.append(y)

        x_train = np.concatenate(x_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        x_test, y_test, windows_test, source_edf_test = concat_patient_files(
            patient_files[test_pid]
        )

        model, score = train_xgb(x_train, y_train)

        train_proba=model.predict_proba(x_train)[:,1]
        test_proba=model.predict_proba(x_test)[:,1]

        # plot_probability_histograms(
        #     y_train, train_proba,
        #     y_test, test_proba,
        #     test_pid
        # )

        imp_vec = np.zeros(n_features)

        for k, v in score.items():
            idx = int(k[1:])  # "f6" → 6
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

    all_importances = np.array(all_importances)  # (n_folds, n_features)

    mean_imp = all_importances.mean(axis=0)
    std_imp = all_importances.std(axis=0)

    idx = np.argsort(mean_imp)[::-1]

    # for i in idx:
    #     print(f"f{i:02d}  mean={mean_imp[i]:.6f}  std={std_imp[i]:.6f}")
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
