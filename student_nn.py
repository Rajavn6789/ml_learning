import pandas as pd
import torch
import torch.nn as nn

# -------------------------
# 0. Reproducibility
# -------------------------
# Fix the random seed so weight init and the train/test split are the same
# every run. Without this, accuracy jumps around and you can't tell whether a
# change actually helped.
torch.manual_seed(42)

# -------------------------
# 1. Load CSV
# -------------------------
# The data is already scaled to 0-1, so no normalization is needed here.
df = pd.read_csv("students.csv")

X = torch.tensor(df[['attendance', 'study_hours', 'previous_score']].values,
                 dtype=torch.float32)

y = torch.tensor(df[['pass']].values,
                 dtype=torch.float32)

# -------------------------
# 2. Train / test split
# -------------------------
# The original script trained and tested on ALL the data, so "accuracy" just
# measured how well the model memorized the training set. We hold out 20% of
# the rows so we can measure how well it generalizes to unseen students.
n = X.shape[0]
perm = torch.randperm(n)             # shuffled row indices
test_size = int(0.2 * n)

test_idx = perm[:test_size]
train_idx = perm[test_size:]

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

# -------------------------
# 3. Model (3 -> 2 -> 1)
# -------------------------
# Note: the final layer outputs a raw "logit" (no sigmoid). We apply the
# sigmoid inside the loss function (BCEWithLogitsLoss), which is more
# numerically stable than Sigmoid + BCELoss.
class PassFailNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)              # raw logit, no sigmoid here
        return x

model = PassFailNN()

# -------------------------
# 4. Loss & optimizer
# -------------------------
criterion = nn.BCEWithLogitsLoss()   # sigmoid + BCE combined (stable)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# -------------------------
# 5. Training loop (TRAIN DATA ONLY)
# -------------------------
epochs = 500

for epoch in range(epochs):
    logits = model(X_train)
    loss = criterion(logits, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# -------------------------
# 6. Evaluate on held-out test data
# -------------------------
def accuracy(logits, targets):
    # logits -> probabilities -> 0/1 predictions
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    return (preds == targets).float().mean().item()

with torch.no_grad():
    train_acc = accuracy(model(X_train), y_train)
    test_acc = accuracy(model(X_test), y_test)

print(f"\nTrain accuracy: {train_acc:.3f}")
print(f"Test accuracy:  {test_acc:.3f}   (on {test_size} unseen students)")

# -------------------------
# 7. Predict a new student
# -------------------------
# Inputs are on the same 0-1 scale as the CSV: attendance, study_hours,
# previous_score.
new_student = torch.tensor([[0.8, 0.6, 0.7]], dtype=torch.float32)

with torch.no_grad():
    prob = torch.sigmoid(model(new_student)).item()

print(f"\nPrediction probability: {prob:.3f}")
print("Result:", "PASS" if prob >= 0.5 else "FAIL")
