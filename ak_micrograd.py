# What this corresponds to (Andrej Karpathy mapping)
# torch.Tensor → micrograd Value
# loss.backward() → automatic backprop engine
# nn.Module → MLP / Layer abstraction
# optimizer.step() → manual SGD loop you wrote in micrograd
# zero_grad() → reset gradients each step

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 1. Toy dataset
# -----------------------
X = torch.tensor([
    [2.0, 3.0],
    [1.0, 1.0],
    [2.0, 0.5],
    [0.5, 1.5],
], dtype=torch.float32)

y = torch.tensor([[1.0], [-1.0], [1.0], [-1.0]], dtype=torch.float32)

# -----------------------
# 2. Model (MLP)
# -----------------------
model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

# log model size
n_params = sum(p.numel() for p in model.parameters())
total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
bytes_per_param = next(model.parameters()).element_size()
print(f"model: {n_params} params, {bytes_per_param} bytes/param, {total_bytes} bytes total")

# -----------------------
# 3. Loss + optimizer
# -----------------------
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

# -----------------------
# 4. Training loop
# -----------------------
# early-stopping config: stop once the loss stops improving meaningfully
best_loss = float("inf")
patience = 50      # how many steps to wait for an improvement before stopping
min_delta = 1e-4   # smallest loss drop that counts as a real improvement
wait = 0           # steps since the last meaningful improvement

for step in range(1000):

    # forward pass
    pred = model(X)
    loss = loss_fn(pred, y)

    # backward pass
    optimizer.zero_grad()
    loss.backward()

    # update weights
    optimizer.step()

    if step % 20 == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

    # early stopping: reset the counter on improvement, else count down patience
    if best_loss - loss.item() > min_delta:
        best_loss = loss.item()
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print(f"early stopping at step {step}, loss = {loss.item():.4f}")
            break