import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from xgboost import XGBClassifier


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def patient_from_filename(name: str) -> str | None:
    """
    Extract patient id from filenames like:
    P20_GHB_00015_000348_features.npz -> 'P20'
    """
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_features(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    X = z["X"].astype(np.float32)
    y = z["y"].astype(np.uint8).ravel()
    return X, y


def plot_proba_histograms(y_proba: np.ndarray, y_true: np.ndarray, title: str, bins: int = 50, threshold: float | None = None):
    """
    Overlaid histograms of predicted probabilities split by true label.
    """
    y_true = y_true.astype(int).ravel()

    p0 = y_proba[y_true == 0]
    p1 = y_proba[y_true == 1]

    # two colors by default matplotlib cycle; no manual color setting needed
    plt.hist(p0, bins=bins, alpha=0.6, label=f"y=0 (n={len(p0)})")
    plt.hist(p1, bins=bins, alpha=0.6, label=f"y=1 (n={len(p1)})")

    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"thr={threshold:.2f}")

    plt.xlim(0, 1)
    plt.xlabel("Predicted probability P(y=1)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.legend()

def find_low_false_neg(y_proba: np.ndarray, y_true: np.ndarray):
    y_true = y_true.astype(int).ravel()
    pos_idx=np.where(y_true==1)[0]
    p1 = y_proba[pos_idx]
    lowest_idx=np.argsort(p1)
    return  pos_idx[lowest_idx[:10]], y_proba[pos_idx[lowest_idx[:10]]]

def sec_to_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def locate_test_sample(test_sources, global_idx: int):
    offset = 0
    for feat_path, n in test_sources:
        if global_idx < offset + n:
            return feat_path, global_idx - offset
        offset += n
    raise IndexError("global_idx out of range")

def describe_proba(y_proba: np.ndarray, y_true: np.ndarray, name: str):
    """
    Quick numeric sanity checks for skew/collapse by class.
    """
    y_true = y_true.astype(int).ravel()
    p0 = y_proba[y_true == 0]
    p1 = y_proba[y_true == 1]

    def stats(a: np.ndarray):
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size),
            "min": float(np.min(a)),
            "p10": float(np.quantile(a, 0.10)),
            "p50": float(np.median(a)),
            "p90": float(np.quantile(a, 0.90)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
        }

    s0 = stats(p0)
    s1 = stats(p1)
    print(f"\n[{name}] proba stats by true class")
    print("  y=0:", s0)
    print("  y=1:", s1)


# -------------------------------------------------
# Main: train on everyone except P20, plot proba histograms
# -------------------------------------------------

def main():
    in_dir = Path("../data/features_cache_basic")
    files = sorted(in_dir.glob("*_features.npz"))
    if not files:
        raise RuntimeError(f"No *_features.npz files found in {in_dir}")

    # group files by patient (Pxx)
    patient_files: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid is None:
            print(f"Skipping (no patient id): {f.name}")
            continue
        patient_files.setdefault(pid, []).append(f)

    patients = sorted(patient_files.keys())
    print(f"Found {len(files)} files, {len(patients)} patients:")
    print(patients)

    test_pid = "P20"
    if test_pid not in patients:
        raise RuntimeError(f"{test_pid} not found in dataset patients: {patients}")

    # -------------------------------------------------
    # Build train/test splits:
    # - Train: all patients except P20 (annotated/balanced windows)
    # - Test:  P20 annotated/balanced windows (NOT full EEG sliding windows)
    # -------------------------------------------------
    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []
    test_sources = []

    for pid, flist in patient_files.items():
        for f in flist:
            x, y = load_features(f)
            if pid == test_pid:
                x_test_list.append(x)
                y_test_list.append(y)
                test_sources.append((f, len(y)))  # <-- ΚΡΙΣΙΜΟ
            else:
                x_train_list.append(x)
                y_train_list.append(y)

    if not x_train_list or not x_test_list:
        raise RuntimeError("Empty train or test split. Check your files/patient grouping.")

    x_train = np.concatenate(x_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    x_test = np.concatenate(x_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    print(
        f"\nSplit summary:\n"
        f"  Train (all except {test_pid}): n={len(y_train)} pos={int(y_train.sum())} neg={int((y_train==0).sum())}\n"
        f"  Test  ({test_pid}): n={len(y_test)} pos={int(y_test.sum())} neg={int((y_test==0).sum())}"
    )

    # handle class imbalance (in case it's not perfectly balanced)
    pos = int(y_train.sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(x_train, y_train)

    # probabilities for (a) train and (b) test (P20)
    y_proba_train = model.predict_proba(x_train)[:, 1]
    y_proba_test = model.predict_proba(x_test)[:, 1]

    # quick numeric sanity checks
    describe_proba(y_proba_train, y_train, name="TRAIN")
    describe_proba(y_proba_test, y_test, name=f"TEST ({test_pid})")

    # -------------------------------------------------
    # Plots: separate histograms by true label
    # -------------------------------------------------
    threshold_for_line = 0.5  # optional reference line; set to None to disable

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plot_proba_histograms(
        y_proba_train,
        y_train,
        title=f"Train (all except {test_pid}): probas by true label",
        bins=50,
        threshold=threshold_for_line,
    )

    plt.subplot(1, 2, 2)
    plot_proba_histograms(
        y_proba_test,
        y_test,
        title=f"Test ({test_pid}): probas by true label",
        bins=50,
        threshold=threshold_for_line,
    )
    idx, probs = find_low_false_neg(y_proba_test, y_test)

    print("\nLowest-probability FALSE NEGATIVES (P20):")
    for gi, p in zip(idx, probs):
        feat_path, local_idx = locate_test_sample(test_sources, int(gi))

        z = np.load(feat_path, allow_pickle=True)

        desc = str(z["kept_desc"][local_idx])
        onset = float(z["kept_onset_sec"][local_idx])
        ann_i = int(z["kept_ann_idx"][local_idx])

        print(
            f"global={gi:4d}  local={local_idx:3d}  "
            f"proba={p:.6f}  "
            f"desc={desc}  "
            f"time={sec_to_hms(onset)}  "
            f"(onset_sec={onset:.3f})  "
            f"ann_idx_in_raw={ann_i}  "
            f"file={feat_path.name}"
        )

    plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    main()
