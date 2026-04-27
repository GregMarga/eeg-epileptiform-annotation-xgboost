import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# Config
# -------------------------------------------------

CSV_FILES = {
    "LaBraM (Foundation Model)":       Path("results_labram.csv"),
    "Handcrafted Features":            Path("results_handcrafted.csv"),
    "Power Feature Only":              Path("results_handcrafted_f7.csv"),
    "Combined (Handcrafted + LaBraM)": Path("results_combined.csv"),
}

# LOPO baseline accuracies (από τα δικά σου LOPO runs)
LOPO_BASELINES = {
    "LaBraM (Foundation Model)":       0.8255,
    "Handcrafted Features":            0.8554,
    "Power Feature Only":              0.8776,
    "Combined (Handcrafted + LaBraM)": 0.8688,
}

# Χρώματα και markers ανά curve — match στο reference plot
STYLE = {
    "LaBraM (Foundation Model)":       {"color": "tab:blue",   "marker": "o"},
    "Handcrafted Features":            {"color": "tab:green",  "marker": "s"},
    "Power Feature Only":              {"color": "tab:orange", "marker": "^"},
    "Combined (Handcrafted + LaBraM)": {"color": "tab:purple", "marker": "D"},
}

METRIC = "acc"   # στήλη για y-axis

# -------------------------------------------------
# Plot
# -------------------------------------------------

def main():
    plt.figure(figsize=(14, 7))

    # Few-shot curves
    for label, path in CSV_FILES.items():
        if not path.exists():
            print(f"  [WARN] Missing {path}, skipping")
            continue
        df = pd.read_csv(path)
        st = STYLE[label]
        plt.plot(
            df["n_shot"], df[METRIC],
            color=st["color"], marker=st["marker"],
            linewidth=1.8, markersize=7,
            label=label,
        )

    # LOPO baselines (οριζόντιες γραμμές)
    for label, baseline in LOPO_BASELINES.items():
        st = STYLE[label]
        plt.axhline(
            y=baseline,
            color=st["color"], linestyle="--", linewidth=1.3,
            alpha=0.8,
            label=f"{label.split(' (')[0]} LOPO ({baseline*100:.2f}%)",
        )

    plt.xlabel("Number of Support Samples (N-shot per class)")
    plt.ylabel("Accuracy")
    plt.title("Few-Shot Learning Curve — Prototypical Network")
    plt.ylim(0.5, 1.0)
    plt.xticks(range(1, 21))
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("few_shot_learning_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()