# Edutva Deep Learning, Session 2
# Lab 2B: the same network in PyTorch, then trained on MNIST.
#   Part A: rebuild the Lab 2A network with nn.Module and prove autograd gives the
#           same gradients and the same loss curve as our hand-written NumPy backprop.
#   Part B: narrate one training step (the five steps of every PyTorch loop).
#   Part C: train an MLP on MNIST with DataLoader and reach above 95% accuracy.
# Needs: torch, torchvision, numpy, matplotlib, scikit-learn
# Run: python lab2b_pytorch_mnist.py   (Colab: !pip install torch torchvision is already done;
#      remove the matplotlib.use line to show plots inline). MNIST downloads on first run.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.datasets import make_moons

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# =============================================================================
# Part A: PyTorch autograd versus our NumPy backprop (the Lab 2A network)
# =============================================================================
X, y = make_moons(n_samples=200, noise=0.2, random_state=SEED)
y = y.reshape(-1, 1)


# --- NumPy reference (copied from Lab 2A so this file runs on its own) -------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def init_params(n_in=2, n_hidden=8, seed=1):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 0.5, (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": r.normal(0, 0.5, (n_hidden, 1)), "b2": np.zeros(1)}


def np_forward(p, X):
    a1 = np.tanh(X @ p["W1"] + p["b1"])
    return sigmoid(a1 @ p["W2"] + p["b2"]), a1


def np_loss(p, X, y):
    a2, _ = np_forward(p, X)
    return -np.mean(y * np.log(a2 + 1e-12) + (1 - y) * np.log(1 - a2 + 1e-12))


def np_backward(p, X, y):
    a2, a1 = np_forward(p, X)
    dz2 = (a2 - y) / y.shape[0]
    dz1 = (dz2 @ p["W2"].T) * (1 - a1 ** 2)
    return {"W1": X.T @ dz1, "b1": dz1.sum(0), "W2": a1.T @ dz2, "b2": dz2.sum(0)}


# --- The same network as an nn.Module ----------------------------------------
class TwoLayerNet(nn.Module):
    """X -> Linear -> tanh -> Linear (logits). Sigmoid lives inside the loss."""

    def __init__(self, n_in=2, n_hidden=8):
        super().__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def load_numpy_params(model, p):
    """Copy NumPy weights in. nn.Linear stores weight as (out, in), so we transpose."""
    with torch.no_grad():
        model.fc1.weight.copy_(torch.from_numpy(p["W1"].T))
        model.fc1.bias.copy_(torch.from_numpy(p["b1"]))
        model.fc2.weight.copy_(torch.from_numpy(p["W2"].T))
        model.fc2.bias.copy_(torch.from_numpy(p["b2"]))


Xt = torch.from_numpy(X)                    # float64, matching NumPy
yt = torch.from_numpy(y.astype(np.float64))
criterion = nn.BCEWithLogitsLoss()           # sigmoid + binary cross-entropy, mean over N

# 1) Gradient comparison on one pass
p = init_params()
net = TwoLayerNet().double()
load_numpy_params(net, p)
loss = criterion(net(Xt), yt)
loss.backward()                              # autograd does what backward() did by hand
mine = np_backward(p, X, y)
pairs = {"W1": (net.fc1.weight.grad.T.numpy(), mine["W1"]),
         "b1": (net.fc1.bias.grad.numpy(), mine["b1"]),
         "W2": (net.fc2.weight.grad.T.numpy(), mine["W2"]),
         "b2": (net.fc2.bias.grad.numpy(), mine["b2"])}
print(f"\nPart A1: autograd versus hand-written backprop (loss {loss.item():.6f})")
ok = True
for name, (auto, hand) in pairs.items():
    err = np.linalg.norm(auto - hand) / (np.linalg.norm(auto) + np.linalg.norm(hand) + 1e-12)
    ok &= err < 1e-9
    print(f"  {name}: relative error {err:.2e}  {'PASS' if err < 1e-9 else 'FAIL'}")
assert ok, "Autograd and NumPy gradients disagree"

# 2) Same training, step for step: 500 full-batch steps, lr 0.5
lr, steps = 0.5, 500
p = init_params()
np_losses = []
for _ in range(steps):
    np_losses.append(np_loss(p, X, y))
    g = np_backward(p, X, y)
    for k in p:
        p[k] -= lr * g[k]

net = TwoLayerNet().double()
load_numpy_params(net, init_params())
opt = torch.optim.SGD(net.parameters(), lr=lr)
torch_losses = []
for _ in range(steps):
    opt.zero_grad()
    loss = criterion(net(Xt), yt)
    loss.backward()
    opt.step()
    torch_losses.append(loss.item())

gap = np.max(np.abs(np.array(np_losses) - np.array(torch_losses)))
print(f"Part A2: max gap between NumPy and PyTorch loss curves = {gap:.2e}")
print(f"         loss {torch_losses[0]:.3f} -> {torch_losses[-1]:.3f} in both")
assert gap < 1e-6, "Loss curves differ: check the weight copy or the learning rate"

# =============================================================================
# Part B: one training step, narrated (the five steps of a PyTorch loop)
# =============================================================================
tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST("data", train=True, download=True, transform=tfm)
test_ds = datasets.MNIST("data", train=False, download=True, transform=tfm)
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)


class MNISTNet(nn.Module):
    """784 pixels -> 128 tanh units -> 10 class scores (logits)."""

    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


def make_model():
    torch.manual_seed(SEED)
    return MNISTNet().to(device)


model = make_model()
ce = nn.CrossEntropyLoss()                   # softmax + cross-entropy, mean over the batch
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

xb, yb = next(iter(train_loader))
xb, yb = xb.to(device), yb.to(device)
w_before = model.net[1].weight.detach().clone()

print("\nPart B: one training step on a batch of", len(xb), "images")
opt.zero_grad()                              # step 1: clear old gradients
logits = model(xb)                           # step 2: forward pass
loss = ce(logits, yb)                        # step 3: loss
print(f"  forward + loss: {loss.item():.4f}  (a random 10-class guess is ln(10) = {np.log(10):.4f})")
loss.backward()                              # step 4: backward pass (autograd)
print(f"  backward: gradient norm of first layer = {model.net[1].weight.grad.norm():.4f}")
opt.step()                                   # step 5: update the weights
change = (model.net[1].weight.detach() - w_before).norm()
print(f"  update: first-layer weights moved by {change:.4f}")
with torch.no_grad():
    print(f"  same batch after the step: loss {ce(model(xb), yb).item():.4f}")

# =============================================================================
# Part C: full training on MNIST
# =============================================================================
def evaluate(model, loader):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss_sum += ce(logits, yb).item() * len(yb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += len(yb)
    return loss_sum / total, correct / total


model = make_model()                         # fresh weights, same seed
opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
EPOCHS = 10
history = {"train_loss": [], "test_loss": [], "test_acc": []}
print(f"\nPart C: training {EPOCHS} epochs on {len(train_ds)} images")
for epoch in range(1, EPOCHS + 1):
    model.train()
    running, seen = 0.0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = ce(model(xb), yb)
        loss.backward()
        opt.step()
        running += loss.item() * len(yb)
        seen += len(yb)
    test_loss, test_acc = evaluate(model, test_loader)
    history["train_loss"].append(running / seen)
    history["test_loss"].append(test_loss)
    history["test_acc"].append(test_acc)
    print(f"  epoch {epoch:2d}  train loss {running / seen:.4f}  "
          f"test loss {test_loss:.4f}  test acc {test_acc:.4f}")

final_acc = history["test_acc"][-1]
print(f"\nDone when: test accuracy above 95%? {final_acc:.4f}  "
      f"{'PASS' if final_acc > 0.95 else 'NOT YET (try more epochs or a lower learning rate)'}")
print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

# --- Plots --------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(np_losses, label="NumPy (Lab 2A, by hand)", lw=3, alpha=0.5)
axes[0].plot(torch_losses, "--", label="PyTorch autograd", lw=1.5)
axes[0].set_title("Part A: identical loss curves (moons)")
axes[0].set_xlabel("step")
axes[0].set_ylabel("BCE loss")
axes[0].legend()
axes[0].grid(alpha=0.3)
epochs = range(1, EPOCHS + 1)
axes[1].plot(epochs, history["train_loss"], label="train")
axes[1].plot(epochs, history["test_loss"], label="test")
axes[1].set_title("Part C: MNIST loss")
axes[1].set_xlabel("epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)
axes[2].plot(epochs, history["test_acc"], marker="o")
axes[2].axhline(0.95, color="r", ls="--", lw=1)
axes[2].set_title("Part C: MNIST test accuracy (red = 95% target)")
axes[2].set_xlabel("epoch")
axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lab2b_pytorch_mnist.png", dpi=130)
print("Saved lab2b_pytorch_mnist.png")

print("""
Questions to answer in your notebook:
1. Why do we call opt.zero_grad() every step? What happens if we forget?
2. nn.BCEWithLogitsLoss takes logits, not probabilities. Why is that more stable than
   sigmoid followed by log?
3. Narrate one training step in your own words, from input batch to weight update.
4. Our NumPy and PyTorch loss curves match to 1e-6. Name two things that could make them
   drift apart in float32 on a GPU.
""")
