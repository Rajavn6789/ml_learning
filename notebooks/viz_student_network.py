# Draw the StudentGradeModel architecture (3 -> 16 -> 16 -> 4) as neurons +
# fully-connected edges, and save it as a PNG into notebooks/images.
#
# Edge color/width is driven by the *actual trained weights* (blue = positive,
# red = negative), so the picture reflects what the network learned. This mirrors
# the style of ak_micrograd_network.py at the repo root.

import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # headless backend, just write a file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.manual_seed(0)
HERE = os.path.dirname(os.path.abspath(__file__))

# -----------------------
# Rebuild + train the model so edges show real learned weights
# -----------------------
df = pd.read_csv(os.path.join(HERE, "datasets", "student_grade_training.csv"))
X = torch.tensor(df[["subject1", "subject2", "subject3"]].values, dtype=torch.float32) / 100.0
y = torch.tensor(df["grade"].values, dtype=torch.long)

model = nn.Sequential(
    nn.Linear(3, 16), nn.ReLU(),
    nn.Linear(16, 16), nn.ReLU(),
    nn.Linear(16, 4),
)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
for _ in range(3000):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
print(f"trained, final loss {loss.item():.4f}")

linears = [m for m in model if isinstance(m, nn.Linear)]
weights = [lin.weight.detach() for lin in linears]  # (16,3), (16,16), (4,16)

# -----------------------
# Layout: one column of neurons per layer
# -----------------------
layer_sizes = [3, 16, 16, 4]
layer_names = ["input\n(3)", "hidden 1\nReLU (16)", "hidden 2\nReLU (16)", "output\n(4)"]
layer_colors = ["#cfe8ff", "#d6f5d6", "#d6f5d6", "#ffe0b3"]
node_labels = [
    ["s1", "s2", "s3"],
    [str(i) for i in range(1, 17)],
    [str(i) for i in range(1, 17)],
    ["Fail", "C", "B", "A"],
]

x_gap = 4.0
y_gap = 1.15
positions = []
for li, n in enumerate(layer_sizes):
    x = li * x_gap
    ys = [(i - (n - 1) / 2) * y_gap for i in range(n)]
    positions.append([(x, yy) for yy in ys])

fig, ax = plt.subplots(figsize=(13, 11))
ax.axis("off")
ax.set_title("StudentGradeModel — MLP architecture  3 -> 16 -> 16 -> 4\n"
             "(edge color = trained weight: blue +, red -)",
             fontsize=14, fontweight="bold")

# -----------------------
# Edges: W[j, i] = weight from src neuron i -> dst neuron j
# -----------------------
max_w = max(w.abs().max().item() for w in weights)
for li, W in enumerate(weights):
    src, dst = positions[li], positions[li + 1]
    for j, (x2, y2) in enumerate(dst):
        for i, (x1, y1) in enumerate(src):
            w = W[j, i].item()
            color = "#1f6fd6" if w >= 0 else "#d62728"
            lw = 0.2 + 2.0 * abs(w) / max_w
            alpha = 0.12 + 0.5 * abs(w) / max_w
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha, zorder=1)

# -----------------------
# Neurons on top of edges
# -----------------------
for li, layer in enumerate(positions):
    big = layer_sizes[li] <= 4
    for ni, (x, yv) in enumerate(layer):
        ax.scatter(x, yv, s=900 if big else 620, color=layer_colors[li],
                   edgecolors="black", linewidths=1.4, zorder=2)
        if node_labels[li][ni]:
            ax.text(x, yv, node_labels[li][ni], ha="center", va="center",
                    fontsize=10 if big else 6.5, fontweight="bold", zorder=3)
    ax.text(li * x_gap, -((max(layer_sizes) - 1) / 2) * y_gap - 1.0,
            layer_names[li], ha="center", va="top", fontsize=11)

legend = [
    Line2D([0], [0], color="#1f6fd6", lw=3, label="positive weight"),
    Line2D([0], [0], color="#d62728", lw=3, label="negative weight"),
]
ax.legend(handles=legend, loc="upper right", frameon=True)

ax.margins(0.08)
fig.tight_layout()

os.makedirs(os.path.join(HERE, "images"), exist_ok=True)
out = os.path.join(HERE, "images", "student_grade_network.png")
fig.savefig(out, dpi=150)
print(f"saved {out}")
