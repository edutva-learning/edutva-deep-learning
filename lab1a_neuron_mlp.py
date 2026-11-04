# Edutva Deep Learning, Session 1
# Lab 1A: a neuron from scratch, and a 2-layer MLP forward pass in NumPy.
# Needs: numpy, matplotlib, scikit-learn
# Run: python lab1a_neuron_mlp.py   (in Colab, remove the matplotlib.use line to show plots inline)

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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def tanh(z):
    return np.tanh(z)


# --- Part 1: a single neuron ---------------------------------------------
class Neuron:
    """One neuron: output = activation(w . x + b)."""

    def __init__(self, n_inputs, activation=sigmoid, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.normal(0, 1, n_inputs)
        self.b = 0.0
        self.activation = activation

    def forward(self, x):
        z = x @ self.w + self.b  # weighted sum plus bias
        return self.activation(z)


neuron = Neuron(2)
print("Neuron output on first 3 points:", neuron.forward(X[:3]).round(3))


# --- Part 2: a 2-layer MLP forward pass ----------------------------------
class MLP:
    """Input -> hidden (activation) -> output (sigmoid, P(class 1))."""

    def __init__(self, W1, b1, W2, b2, activation=tanh):
        self.W1, self.b1, self.W2, self.b2 = W1, b1, W2, b2
        self.activation = activation

    @classmethod
    def random(cls, n_in, n_hidden, activation=tanh, seed=0):
        rng = np.random.default_rng(seed)
        return cls(rng.normal(0, 1, (n_in, n_hidden)), np.zeros(n_hidden),
                   rng.normal(0, 1, (n_hidden, 1)), np.zeros(1), activation)

    def forward(self, X):
        h = self.activation(X @ self.W1 + self.b1)      # hidden layer
        return sigmoid(h @ self.W2 + self.b2).ravel()   # output probability

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)


# Backprop is taught in Session 2, so we borrow trained weights from scikit-learn
# and run them through OUR forward pass, then check the two agree.
sk = MLPClassifier(hidden_layer_sizes=(16,), activation="tanh",
                   max_iter=2000, random_state=SEED).fit(X_tr, y_tr)
trained = MLP(sk.coefs_[0], sk.intercepts_[0], sk.coefs_[1], sk.intercepts_[1], tanh)
random_net = MLP.random(2, 16, tanh, seed=1)

match = np.allclose(trained.forward(X_te), sk.predict_proba(X_te)[:, 1])
print("NumPy forward matches scikit-learn:", match)
assert match, "Forward pass disagrees with scikit-learn"


# --- Part 3: decision boundaries -----------------------------------------
def plot_boundary(ax, model_fn, title):
    xx, yy = np.meshgrid(np.linspace(-2, 3, 300), np.linspace(-1.5, 2, 300))
    zz = model_fn(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], alpha=0.3, colors=["#3b82f6", "#ef4444"])
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", edgecolor="k", s=20)
    ax.set_title(title)


fig, axes = plt.subplots(1, 2, figsize=(10, 4))
plot_boundary(axes[0], random_net.forward,
              f"Random weights (test acc {(random_net.predict(X_te) == y_te).mean():.2f})")
plot_boundary(axes[1], trained.forward,
              f"Trained weights, our forward pass (test acc {(trained.predict(X_te) == y_te).mean():.2f})")
plt.tight_layout()
plt.savefig("lab1a_decision_boundary.png", dpi=130)
print("Saved lab1a_decision_boundary.png")

print("""
Questions to answer in your notebook:
1. Why is a network with no activation function just one linear layer?
2. What does each weight and the bias do to the neuron's decision line?
3. The random network scores 0.15, far below chance, because its line is flipped.
   What does training change, and why is 0.15 as informative as 0.85?
""")
