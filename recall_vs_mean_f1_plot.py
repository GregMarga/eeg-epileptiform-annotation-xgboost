from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# paths
f1_path = Path("../data/p20_f1_curve.npz")
hit_path = Path("../data/p20_hit_ratio_curve.npz")

# load curves
fz = np.load(f1_path)
hz = np.load(hit_path)

thresholds_f1 = fz["thresholds"]
p20_f1 = fz["f1"]

thresholds_hit = hz["thresholds"]
hit_ratio = hz["hit_ratio"]

# sanity check
if not np.allclose(thresholds_f1, thresholds_hit):
    raise RuntimeError("Threshold grids do NOT match!")

# plot
plt.figure()

# color-coded scatter by threshold
sc = plt.scatter(p20_f1, hit_ratio, c=thresholds_f1, cmap="viridis", s=30)
plt.colorbar(sc, label="Threshold")

# connect points lightly
plt.plot(p20_f1, hit_ratio, alpha=0.3)

# mark specific thresholds
# mark_thresholds = [0.2, 0.4, 0.6, 0.8]

# for t in mark_thresholds:
#     i = np.argmin(np.abs(thresholds_f1 - t))
#     plt.scatter(p20_f1[i], hit_ratio[i], s=80)
#     plt.text(
#         p20_f1[i] + 0.01,
#         hit_ratio[i] + 0.01,
#         f"{t:.2f}",
#         fontsize=9,
#     )

plt.xlabel("P20 F1 on annotated windows")
plt.ylabel("Hit ratio on full P20 EEG")
plt.title("P20: Annotated Performance vs Full-EEG Hit Ratio")

plt.grid(True)
plt.tight_layout()
plt.show()
