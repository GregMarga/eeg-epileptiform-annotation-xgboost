import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Config
# -------------------------------------------------

CSV_FILES = {
    "LaBraM (Foundation Model)":      Path("results_labram.csv"),
    "Handcrafted Features":            Path("results_handcrafted.csv"),
    "Power Feature Only":              Path("results_power_feature.csv"),
    "Combined (Handcrafted + LaBraM)": Path("results_combined.csv"),
}

METRIC  = "acc"      # change to "bacc", "f1", "roc_auc" etc.

COLORS  = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0"]
MARKERS = ["o", "s", "^", "D"]

# -------------------------------------------------
# Load
# -------------------------------------------------

def load_csv(path: Path) -> tuple[list[int], list[float]]:
    n_shots, means = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            n_shots.append(int(row["n_shot"]))
            means.append(float(row[METRIC]))
    return n_shots, means

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.style.use("default")
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

for (label, path), color, marker in zip(CSV_FILES.items(), COLORS, MARKERS):
    if not path.exists():
        print(f"[WARN] {path} not found, skipping.")
        continue
    n_shots, means = load_csv(path)
    ax.plot(n_shots, means, marker=marker, color=color, label=label,
            linewidth=2, markersize=6)

ax.set_xlabel("Number of Support Samples (N-shot per class)", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Few-Shot Learning Curve — Prototypical Network", fontsize=13)
ax.set_xticks(n_shots)
ax.set_xlim(0.5, max(n_shots) + 0.5)
ax.set_ylim(0.5, 1.0)
ax.legend(fontsize=10, loc="lower right")
ax.grid(True, alpha=0.3, linestyle="--")

plt.tight_layout()
plt.savefig("few_shot_curves.png", dpi=150, facecolor="white")
print("Saved: few_shot_curves.png")
plt.show()