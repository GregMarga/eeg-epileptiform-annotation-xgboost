import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# -------------------------------------------------
# Paths
# -------------------------------------------------

FEATURES_DIR   = Path("../../../data/80hz_freq_time_features_cache_basic")
EMBEDDINGS_DIR = Path("../../../data/labram_classification_1s")

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
        if pid:
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


def load_all(index: dict[str, list[Path]], loader_fn) -> tuple[np.ndarray, np.ndarray]:
    all_X, all_y = [], []
    for pid in sorted(index):
        for f in index[pid]:
            x, y = loader_fn(f)
            all_X.append(x)
            all_y.append(y)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  → {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def subsample(X: np.ndarray, y: np.ndarray, max_samples: int = 5000, random_state: int = 42):
    if len(X) <= max_samples:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=max_samples, replace=False)
    return X[idx], y[idx]


def run_tsne(X: np.ndarray, n_pca: int = 50, random_state: int = 42) -> np.ndarray:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA μόνο αν έχουμε περισσότερα features από n_pca
    if X_scaled.shape[1] > n_pca:
        n_pca = min(n_pca, X_scaled.shape[0] - 1)
        pca = PCA(n_components=n_pca, random_state=random_state)
        X_scaled = pca.fit_transform(X_scaled)
        print(f"  PCA {n_pca} components → {pca.explained_variance_ratio_.sum()*100:.1f}% variance")
    else:
        print(f"  Skipping PCA ({X_scaled.shape[1]} features, no reduction needed)")

    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1,
    )
    return tsne.fit_transform(X_scaled)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    max_samples  = 5000
    random_state = 42

    # --- Load ---
    print("Loading handcrafted features...")
    feat_index = build_index(FEATURES_DIR, "_features.npz")
    X_hc, y_hc = load_all(feat_index, load_features)

    print("Loading LaBraM embeddings...")
    emb_index = build_index(EMBEDDINGS_DIR, "_embeddings_labeled.npz")
    X_lb, y_lb = load_all(emb_index, load_embeddings)

    # --- Subsample ---
    print(f"\nSubsampling to max {max_samples} samples...")
    X_hc, y_hc = subsample(X_hc, y_hc, max_samples, random_state)
    X_lb, y_lb = subsample(X_lb, y_lb, max_samples, random_state)
    print(f"  Handcrafted : {X_hc.shape[0]} samples")
    print(f"  LaBraM      : {X_lb.shape[0]} samples")

    # --- t-SNE ---
    print("\nRunning t-SNE on handcrafted features...")
    Z_hc = run_tsne(X_hc, n_pca=50, random_state=random_state)

    print("Running t-SNE on LaBraM embeddings...")
    Z_lb = run_tsne(X_lb, n_pca=50, random_state=random_state)

    # --- Plot ---
    colors = {0: "#4C72B0", 1: "#DD8452"}
    labels = {0: "NON-PD", 1: "PD"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, Z, y, title in [
        (axes[0], Z_hc, y_hc, "(a) Handcrafted Features"),
        (axes[1], Z_lb, y_lb, "(b) LaBraM Embeddings"),
    ]:
        for cls in [0, 1]:
            mask = y == cls
            ax.scatter(
                Z[mask, 0], Z[mask, 1],
                c=colors[cls],
                label=labels[cls],
                alpha=0.35,
                s=8,
                linewidths=0,
            )
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.legend(markerscale=3, fontsize=10)
        ax.grid(alpha=0.2)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle("Figure 5.X: t-SNE Visualization of Feature Spaces", fontsize=14)
    plt.tight_layout()
    plt.savefig("tsne_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nSaved: tsne_comparison.png")


if __name__ == "__main__":
    main()