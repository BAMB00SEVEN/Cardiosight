import numpy as np
import matplotlib.pyplot as plt

np.random.seed(7)

classes = ["Normal", "PVC", "APB", "AFib", "Other"]
n = len(classes)

# Illustrative confusion matrix - diagonally dominant, i.e. a "good" classifier sample
base = np.array([
    [935,  8,  6,  3,  8],
    [ 12, 88,  4,  2,  3],
    [  9,  5, 76,  3,  4],
    [  4,  2,  3, 91,  2],
    [ 10,  4,  5,  3, 70],
])

fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
im = ax.imshow(base, cmap="Greens")

ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels(classes)
ax.set_yticklabels(classes)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
ax.set_title("Sample Confusion Matrix — ECG Beat Classification\n(Random Forest on extracted features)", fontsize=12, fontweight="bold")

for i in range(n):
    for j in range(n):
        val = base[i, j]
        color = "white" if val > base.max()/2 else "black"
        ax.text(j, i, str(val), ha="center", va="center", color=color, fontsize=10)

plt.colorbar(im, fraction=0.046, pad=0.04, label="Number of beats")
fig.text(0.5, -0.02, "Illustrative sample — replace with your own model's confusion matrix after training/testing",
          ha="center", fontsize=8.5, style="italic", color="#555555")
plt.tight_layout()
plt.savefig("/home/claude/CardioSight/assets/confusion_matrix_sample.png", bbox_inches="tight", facecolor="white")
print("saved confusion_matrix_sample.png")
