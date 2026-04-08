from pathlib import Path
import numpy as np


# -------------------------------------------------
# Paths
# -------------------------------------------------

EMB_DIR = Path("../data/embeddings_8sLabeledWindows-noReference/embeddings")
LABEL_DIR = Path("../data/embeddings_8sLabeledWindows-noReference/labels")
OUT_DIR = Path("../data/labram_classification")


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def embedding_base_name(path: Path) -> str:
    """
    Example:
    P20_GHB_00015_000348_embeddings.npy
    ->
    P20_GHB_00015_000348
    """
    name = path.name
    suffix = "_embeddings.npy"
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected embedding filename: {name}")
    return name[:-len(suffix)]

def load_label_file_npy(path: Path):
    """Load labels from a 1D .npy file (binary 0/1 per window)."""
    y = np.load(path, allow_pickle=True).astype(np.uint8).ravel()
    n = len(y)
    # Δημιουργούμε placeholder windows (index-based) αφού δεν υπάρχουν
    windows = np.arange(n).reshape(-1, 1)
    edf_name = path.stem.replace("_embeddings", "")
    ch_names = None
    return y, windows, edf_name, ch_names, ["labels"]

def candidate_label_paths(base: str) -> list[Path]:
    return [
        LABEL_DIR / f"{base}_features.npz",
        LABEL_DIR / f"{base}_windows.npz",
        LABEL_DIR / f"{base}_labeled_windows.npz",
        LABEL_DIR / f"{base}.npz",
        LABEL_DIR / f"{base}_embeddings.npy",  # ← νέο
    ]


def find_label_file(base: str) -> Path:
    candidates = candidate_label_paths(base)
    existing = [p for p in candidates if p.exists()]

    if not existing:
        raise FileNotFoundError(
            f"No matching label file found for base '{base}'. Tried:\n" +
            "\n".join(str(p) for p in candidates)
        )

    if len(existing) > 1:
        raise RuntimeError(
            f"Multiple matching label files found for base '{base}':\n" +
            "\n".join(str(p) for p in existing)
        )

    return existing[0]


def load_embedding_file(path: Path) -> np.ndarray:
    X = np.load(path, allow_pickle=True)

    if not isinstance(X, np.ndarray):
        raise TypeError(f"Embedding file is not an ndarray: {path}")

    X = X.astype(np.float32)

    if X.ndim != 2:
        raise ValueError(f"Expected 2D embeddings array, got shape {X.shape} in {path}")

    return X


def load_label_file(path: Path):
    if path.suffix == ".npy":
        return load_label_file_npy(path)
    z = np.load(path, allow_pickle=True)

    if not isinstance(z, np.lib.npyio.NpzFile):
        raise TypeError(f"Expected .npz label file, got {type(z)} for {path}")

    keys = set(z.files)

    required = {"labels", "windows", "edf_name"}
    if not required.issubset(keys):
        raise KeyError(f"Missing required keys in {path}. Found keys: {sorted(keys)}")

    y = z["labels"].astype(np.uint8).ravel()
    windows = z["windows"]
    edf_name = str(z["edf_name"])
    ch_names = z["ch_names"] if "ch_names" in z else None

    return y, windows, edf_name, ch_names, sorted(keys)


# -------------------------------------------------
# Main merge
# -------------------------------------------------

def main():
    if not EMB_DIR.exists():
        raise FileNotFoundError(f"Embedding directory not found: {EMB_DIR}")

    if not LABEL_DIR.exists():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    emb_files = sorted(EMB_DIR.glob("*_embeddings.npy"))
    if not emb_files:
        raise RuntimeError(f"No *_embeddings.npy files found in {EMB_DIR}")

    print(f"Found {len(emb_files)} embedding files\n")

    matched = 0
    failed = 0

    for emb_path in emb_files:
        base = embedding_base_name(emb_path)

        try:
            label_path = find_label_file(base)

            X = load_embedding_file(emb_path)
            y, windows, edf_name, ch_names, label_keys = load_label_file(label_path)

            n_emb = X.shape[0]
            n_y = len(y)
            n_win = len(windows)

            print(f"[CHECK] {base}")
            print(f"  embeddings: {emb_path.name} -> shape={X.shape}")
            print(f"  labels:     {label_path.name} -> labels.shape={y.shape}")
            print(f"  windows.shape={windows.shape}")
            print(f"  edf_name={edf_name}")
            print(f"  label keys: {label_keys}")

            if n_emb != n_y:
                raise ValueError(
                    f"Annotation count mismatch for {base}: embeddings={n_emb}, labels={n_y}"
                )

            if n_emb != n_win:
                raise ValueError(
                    f"Window count mismatch for {base}: embeddings={n_emb}, windows={n_win}"
                )

            out_path = OUT_DIR / f"{base}_embeddings_labeled.npz"

            save_kwargs = {
                "X": X,
                "y": y,
                "windows": windows,
                "source_edf": np.array([edf_name] * n_emb, dtype=object),
                "embedding_file": str(emb_path),
                "label_file": str(label_path),
            }

            if ch_names is not None:
                save_kwargs["ch_names"] = ch_names

            np.savez_compressed(out_path, **save_kwargs)

            print(f"  OK -> saved: {out_path.name}\n")
            matched += 1

        except Exception as e:
            failed += 1
            print(f"[FAILED] {base}")
            print(f"  {type(e).__name__}: {e}\n")

    print("=== SUMMARY ===")
    print(f"Matched and saved: {matched}")
    print(f"Failed:           {failed}")
    print(f"Output dir:       {OUT_DIR}")


if __name__ == "__main__":
    main()