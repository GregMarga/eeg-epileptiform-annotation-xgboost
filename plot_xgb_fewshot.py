import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------
# Config
# -------------------------------------------------

CSV_FILES = {
    "LaBraM (Foundation Model)":       Path("results_xgb_fewshot_labram.csv"),
    "Handcrafted Features":            Path("results_xgb_fewshot_perfile.csv"),
    "Combined (Handcrafted + LaBraM)": Path("results_xgb_fewshot_combined.csv"),
}

# LOPO baseline accuracies (same values as in the reference plot)
LOPO_BASELINES = {
    "LaBraM (Foundation Model)":       0.8824,
    "Handcrafted Features":            0.8808,
    "Combined (Handcrafted + LaBraM)": 0.9021,
}

# Colors and markers per curve — matched to the reference plot
STYLE = {
    "LaBraM (Foundation Model)":       {"color": "tab:blue",   "marker": "o"},
    "Handcrafted Features":            {"color": "tab:green",  "marker": "s"},
    "Combined (Handcrafted + LaBraM)": {"color": "tab:purple", "marker": "D"},
}

METRIC = "acc"   # column for y-axis

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

    # LOPO baselines (horizontal dashed lines)
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
    plt.title("Few-Shot Learning Curve — XGBoost (per-file)")
    plt.ylim(0.5, 1.0)
    plt.xticks(range(1, 21))
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig("few_shot_xgb_learning_curve.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()