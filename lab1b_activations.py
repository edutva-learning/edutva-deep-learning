# Edutva Deep Learning, Session 1
# Lab 1B: activation functions. Plot each activation and its derivative,
# then swap activations in the MLP and record what changes.
# Needs: numpy, matplotlib, scikit-learn
# Run: python lab1b_activations.py   (in Colab, remove the matplotlib.use line to show plots inline)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
X, y = make_moons(n_samples=600, noise=0.2, random_state=SEED)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=SEED)


# --- Activations and derivatives -----------------------------------------
def sigmoid(z):  return 1 / (1 + np.exp(-z))
def tanh(z):     return np.tanh(z)
def relu(z):     return np.maximum(0, z)
def identity(z): return z
def gelu(z):     return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z**3)))

def d_sigmoid(z):
    s = sigmoid(z)
    return s * (1 - s)
def d_tanh(z):   return 1 - np.tanh(z) ** 2
def d_relu(z):   return (z > 0).astype(float)
def d_gelu(z, h=1e-5):  # numerical derivative, fine for plotting
    return (gelu(z + h) - gelu(z - h)) / (2 * h)

ACTS = {"sigmoid": (sigmoid, d_sigmoid), "tanh": (tanh, d_tanh),
        "relu": (relu, d_relu), "gelu": (gelu, d_gelu)}

# --- Part 1: plot each activation and its derivative ---------------------
z = np.linspace(-6, 6, 400)
fig, axes = plt.subplots(2, 4, figsize=(14, 5), sharex=True)
for i, (name, (f, df)) in enumerate(ACTS.items()):
    axes[0, i].plot(z, f(z))
    axes[0, i].set_title(name)
    axes[0, i].grid(alpha=0.3)
    axes[1, i].plot(z, df(z), color="tab:red")
    axes[1, i].grid(alpha=0.3)
axes[0, 0].set_ylabel("f(z)")
axes[1, 0].set_ylabel("f'(z)")
plt.tight_layout()
plt.savefig("lab1b_activations.png", dpi=130)
plt.close()
print("Saved lab1b_activations.png")

print("\nPeak derivative (higher = stronger gradient signal):")
for name, (f, df) in ACTS.items():
    print(f"  {name:8s} max f'(z) = {df(z).max():.3f}")


# --- Part 2: swap activations in the MLP ---------------------------------
class MLP:
    """Input -> hidden (activation) -> output (sigmoid). Our own forward pass."""

    def __init__(self, W1, b1, W2, b2, activation):
        self.W1, self.b1, self.W2, self.b2, self.activation = W1, b1, W2, b2, activation

    def forward(self, X):
        h = self.activation(X @ self.W1 + self.b1)
        return sigmoid(h @ self.W2 + self.b2).ravel()

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


SK_TO_FN = {"identity": identity, "logistic": sigmoid, "tanh": tanh, "relu": relu}


def fit_and_port(name, hidden=16):
    # Backprop arrives in Session 2: borrow trained weights from scikit-learn.
    sk = MLPClassifier(hidden_layer_sizes=(hidden,), activation=name,
                       max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
    mlp = MLP(sk.coefs_[0], sk.intercepts_[0], sk.coefs_[1], sk.intercepts_[1], SK_TO_FN[name])
    assert np.allclose(mlp.forward(X_te), sk.predict_proba(X_te)[:, 1])
    return sk, mlp


def plot_boundary(ax, model_fn, title):
    xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
    zz = model_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, colors=["#3b82f6", "#ef4444"])
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_title(title)


rows = []
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, name in zip(axes, SK_TO_FN):
    sk, mlp = fit_and_port(name)
    acc_tr = (mlp.predict(X_tr) == y_tr).mean()
    acc_te = (mlp.predict(X_te) == y_te).mean()
    rows.append((name, sk.n_iter_, acc_tr, acc_te, sk.loss_))
    plot_boundary(ax, mlp.forward, f"{name}: test acc {acc_te:.2f}")
plt.tight_layout()
plt.savefig("lab1b_swap_activations.png", dpi=130)
plt.close()
print("Saved lab1b_swap_activations.png")

print("\nResults (hidden layer = 16 units)")
print(f"{'activation':10s} {'iters':>6s} {'train acc':>10s} {'test acc':>9s} {'final loss':>11s}")
for name, it, tr, te, loss in rows:
    print(f"{name:10s} {it:6d} {tr:10.3f} {te:9.3f} {loss:11.4f}")

print("""
Questions to answer in your notebook:
1. Why does the identity (no activation) model draw a straight line, whatever the width?
2. Sigmoid's max derivative is 0.25. What happens to a gradient across 10 sigmoid layers?
3. ReLU needed the most iterations here but scored highest. Why might more iterations
   still be worth it? Which activation gave the smoothest boundary, and why?
""")
