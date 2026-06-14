# Draw the ak_micrograd.py MLP architecture as neurons + fully-connected edges,
# and save it as a PNG.  Architecture: 2 -> 4 (tanh) -> 4 (tanh) -> 1  (37 parameters)
#
# Connection colors/widths are driven by the *actual trained weights*, so the
# picture reflects what the network learned (blue = positive, red = negative).

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")  # headless backend, just write a file
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.manual_seed(1337)

# -----------------------
# Rebuild + train the model (same as ak_micrograd.py) so edges show real weights
# -----------------------
X = torch.tensor([[2.0, 3.0], [1.0, 1.0], [2.0, 0.5], [0.5, 1.5]])
y = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]])

model = nn.Sequential(
    nn.Linear(2, 4), nn.Tanh(),
    nn.Linear(4, 4), nn.Tanh(),
    nn.Linear(4, 1),
)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)
for _ in range(1000):
    loss = loss_fn(model(X), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# pull out the weight matrices of the three Linear layers
linears = [m for m in model if isinstance(m, nn.Linear)]
weights = [lin.weight.detach() for lin in linears]  # shapes: (4,2), (4,4), (1,4)

# -----------------------
# Layout: one column of neurons per layer
# -----------------------
layer_sizes = [2, 4, 4, 1]
layer_names = ["input\n(2)", "hidden 1\ntanh (4)", "hidden 2\ntanh (4)", "output\n(1)"]
layer_colors = ["#cfe8ff", "#d6f5d6", "#d6f5d6", "#ffe0b3"]
# name shown inside each neuron, per layer
node_labels = [
    ["x1", "x2"],
    ["h1$_1$", "h1$_2$", "h1$_3$", "h1$_4$"],
    ["h2$_1$", "h2$_2$", "h2$_3$", "h2$_4$"],
    ["ŷ"],
]

x_gap = 3.0
y_gap = 1.4
positions = []  # positions[layer] = list of (x, y) per neuron
for li, n in enumerate(layer_sizes):
    x = li * x_gap
    ys = [(i - (n - 1) / 2) * y_gap for i in range(n)]
    positions.append([(x, yy) for yy in ys])

fig, ax = plt.subplots(figsize=(11, 6))
ax.axis("off")
ax.set_title("ak_micrograd.py — MLP architecture (edge color = trained weight)",
             fontsize=14, fontweight="bold")

# -----------------------
# Edges: connect every neuron in layer L to every neuron in layer L+1
# W[j, i] = weight from input neuron i -> output neuron j
# -----------------------
max_w = max(w.abs().max().item() for w in weights)
for li, W in enumerate(weights):
    src, dst = positions[li], positions[li + 1]
    for j, (x2, y2) in enumerate(dst):
        for i, (x1, y1) in enumerate(src):
            w = W[j, i].item()
            color = "#1f6fd6" if w >= 0 else "#d62728"   # blue / red
            lw = 0.3 + 3.0 * abs(w) / max_w              # thicker = larger |w|
            alpha = 0.25 + 0.6 * abs(w) / max_w
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=lw, alpha=alpha,
                    zorder=1)

# -----------------------
# Neurons: draw on top of the edges
# -----------------------
for li, layer in enumerate(positions):
    for ni, (x, yv) in enumerate(layer):
        ax.scatter(x, yv, s=900, color=layer_colors[li], edgecolors="black",
                   linewidths=1.5, zorder=2)
        # neuron name centered inside the circle
        ax.text(x, yv, node_labels[li][ni], ha="center", va="center",
                fontsize=9, fontweight="bold", zorder=3)
    # layer label under the column
    ax.text(li * x_gap, -((max(layer_sizes) - 1) / 2) * y_gap - 1.2,
            layer_names[li], ha="center", va="top", fontsize=11)

# legend for edge sign
legend = [
    Line2D([0], [0], color="#1f6fd6", lw=3, label="positive weight"),
    Line2D([0], [0], color="#d62728", lw=3, label="negative weight"),
]
ax.legend(handles=legend, loc="upper right", frameon=True)

ax.margins(0.1)
fig.tight_layout()
out = "ak_micrograd_network.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
