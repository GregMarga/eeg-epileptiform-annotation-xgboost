from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# paths (adjust if needed)
f1_path = Path("../data/lopo_f1_curve.npz")
recall_path = Path("../data/p20_recall_curve.npz")

# load curves
f1z = np.load(f1_path)
rz = np.load(recall_path)

thresholds_f1 = f1z["thresholds"]
mean_f1 = f1z["mean_f1"]

thresholds_rec = rz["thresholds"]
recall = rz["recall"]

# --- sanity check: thresholds must match ---
if not np.allclose(thresholds_f1, thresholds_rec):
    raise RuntimeError("Threshold grids do NOT match between F1 and Recall!")

# plot
plt.figure()
plt.plot(mean_f1, recall, marker="o", markersize=3)
plt.xlabel("Mean F1 (LOPO)")
plt.ylabel("Hit ratio / Recall (P20)")
plt.title("F1 vs Event Recall")
plt.grid(True)
plt.tight_layout()
plt.show()