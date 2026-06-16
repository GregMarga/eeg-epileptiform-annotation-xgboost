import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def patient_from_filename(name: str) -> str | None:
    m = re.match(r"^(P\d+)_", name, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None


def load_features(npz_path: Path):
    z = np.load(npz_path, allow_pickle=True)
    required = {"X", "y", "windows", "source_edf"}
    if not required.issubset(set(z.files)):
        raise KeyError(f"Missing keys in {npz_path}. Found={sorted(z.files)}")
    return z["X"].astype(np.float32), z["y"].astype(np.uint8).ravel()


def build_patient_index(in_dir: Path) -> dict[str, list[Path]]:
    files = sorted(in_dir.glob("*_embeddings_labeled.npz"))
    if not files:
        raise RuntimeError(f"No *_embeddings_labeled.npz files found in {in_dir}")
    patient_files: dict[str, list[Path]] = {}
    for f in files:
        pid = patient_from_filename(f.name)
        if pid:
            patient_files.setdefault(pid, []).append(f)
    return patient_files


def main():
    in_dir = Path("../../../Data/labram_classification_1s")
    patient_files = build_patient_index(in_dir)
    patients = sorted(patient_files.keys())
    print(f"Patients: {patients} (n={len(patients)})")

    # --- Load all data ---
    all_X, all_y = [], []
    for pid in patients:
        for f in patient_files[pid]:
            x, y = load_features(f)
            all_X.append(x)
            all_y.append(y)

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)
    n_samples, n_features = X_all.shape
    print(f"\nTotal samples: {n_samples}, Features: {n_features}")
    print(f"Positive rate: {y_all.mean():.3f}")

    # --- Standardize + Full PCA ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100

    # --- Print thresholds ---
    thresholds = [50, 70, 80, 90, 95, 99]
    print("\n=== Components needed to reach variance threshold ===")
    for thr in thresholds:
        n_needed = int(np.searchsorted(cumvar, thr)) + 1
        print(f"  {thr}% variance → {n_needed} components")

    component_counts = [5, 10, 15, 20, 25, 30, 35, 50, 75, 100]
    component_counts = [c for c in component_counts if c <= min(n_samples, n_features)]
    print("\n=== Variance captured per n_components ===")
    for n in component_counts:
        print(f"  n_components={n:4d} → {cumvar[n - 1]:.2f}% variance explained")

    # -------------------------------------------------
    # Plots
    # -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Plot 1: Scree plot ---
    ax = axes[0]
    n_show = min(100, len(pca_full.explained_variance_ratio_))
    ax.bar(
        range(1, n_show + 1),
        pca_full.explained_variance_ratio_[:n_show] * 100,
        color="steelblue", alpha=0.8
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained (%)")
    ax.set_title("Scree Plot (top 100 components)")
    ax.grid(axis="y", alpha=0.3)

    # --- Plot 2: Cumulative variance curve ---
    ax = axes[1]
    ax.plot(range(1, len(cumvar) + 1), cumvar, color="steelblue", linewidth=2)

    colors_thr = ["gray", "orange", "gold", "green", "red", "purple"]
    for thr, col in zip(thresholds, colors_thr):
        n_needed = int(np.searchsorted(cumvar, thr)) + 1
        ax.axhline(thr, color=col, linestyle="--", alpha=0.6, linewidth=1)
        ax.axvline(n_needed, color=col, linestyle="--", alpha=0.6, linewidth=1)
        ax.scatter([n_needed], [thr], color=col, zorder=5, s=50,
                   label=f"{thr}% → {n_needed} PCs")

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("Cumulative Variance vs. n_components")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, min(200, len(cumvar)))
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("pca_variance_analysis.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: pca_variance_analysis.png")


if __name__ == "__main__":
    main()