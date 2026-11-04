# Edutva Deep Learning, Session 2
# Lab 2A: backprop by hand. Compute gradients for a 2-layer network in NumPy
# and verify them against numerical gradients (gradient checking).
# Needs: numpy, matplotlib, scikit-learn
# Run: python lab2a_backprop.py   (in Colab, remove the matplotlib.use line to show plots inline)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

SEED = 42
rng = np.random.default_rng(SEED)
X, y = make_moons(n_samples=200, noise=0.2, random_state=SEED)
y = y.reshape(-1, 1)  # shape (N, 1)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# --- The network: X -> Linear -> tanh -> Linear -> sigmoid -> BCE loss ----
def init_params(n_in=2, n_hidden=8, seed=0):
    r = np.random.default_rng(seed)
    return {"W1": r.normal(0, 0.5, (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": r.normal(0, 0.5, (n_hidden, 1)), "b2": np.zeros(1)}


def forward(p, X):
    """Returns the prediction and a cache of intermediate values for backprop."""
    z1 = X @ p["W1"] + p["b1"]      # (N, H)
    a1 = np.tanh(z1)                # (N, H)
    z2 = a1 @ p["W2"] + p["b2"]     # (N, 1)
    a2 = sigmoid(z2)                # (N, 1) = P(class 1)
    return a2, {"X": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2}


def loss_fn(p, X, y):
    """Mean binary cross-entropy."""
    a2, _ = forward(p, X)
    eps = 1e-12
    return -np.mean(y * np.log(a2 + eps) + (1 - y) * np.log(1 - a2 + eps))


# --- Backprop by hand: chain rule, one line per step ----------------------
def backward(p, cache, y):
    """Gradients of mean BCE loss w.r.t. every parameter."""
    N = y.shape[0]
    a2, a1, X = cache["a2"], cache["a1"], cache["X"]

    # Output layer. For sigmoid + BCE the two derivatives combine neatly:
    #   dL/dz2 = (a2 - y) / N
    dz2 = (a2 - y) / N                       # (N, 1)
    dW2 = a1.T @ dz2                         # (H, 1)
    db2 = dz2.sum(axis=0)                    # (1,)

    # Push the gradient back through W2, then through tanh: tanh'(z) = 1 - tanh(z)^2
    da1 = dz2 @ p["W2"].T                    # (N, H)
    dz1 = da1 * (1 - a1 ** 2)                # (N, H)
    dW1 = X.T @ dz1                          # (2, H)
    db1 = dz1.sum(axis=0)                    # (H,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


# --- Gradient checking: central differences -------------------------------
def numerical_gradients(p, X, y, h=1e-5):
    """(L(w + h) - L(w - h)) / (2h) for every single parameter."""
    grads = {}
    for name, arr in p.items():
        g = np.zeros_like(arr)
        it = np.nditer(arr, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            old = arr[idx]
            arr[idx] = old + h
            loss_plus = loss_fn(p, X, y)
            arr[idx] = old - h
            loss_minus = loss_fn(p, X, y)
            arr[idx] = old                    # restore
            g[idx] = (loss_plus - loss_minus) / (2 * h)
        grads[name] = g
    return grads


def relative_error(a, b):
    return np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12)


params = init_params(seed=1)
_, cache = forward(params, X)
analytic = backward(params, cache, y)
numeric = numerical_gradients(params, X, y)

print(f"{'param':6s} {'shape':10s} {'relative error':>16s}  status")
all_ok = True
for name in params:
    err = relative_error(analytic[name], numeric[name])
    ok = err < 1e-7
    all_ok &= ok
    print(f"{name:6s} {str(params[name].shape):10s} {err:16.2e}  {'PASS' if ok else 'FAIL'}")
print("\nGradient check:", "PASSED" if all_ok else "FAILED")
assert all_ok, "Analytic and numerical gradients disagree: check backward()"


# --- Show the checker catches a bug ---------------------------------------
buggy = {k: v.copy() for k, v in analytic.items()}
buggy["W1"] = buggy["W1"] * 1.01   # a 1% error, like a forgotten scale factor
print(f"\nDeliberate 1% bug in dW1: relative error = "
      f"{relative_error(buggy['W1'], numeric['W1']):.2e}  (above 1e-7, so the check catches it)")


# --- Use the gradients: a short training run -------------------------------
p = init_params(seed=1)
lr, losses = 0.5, []
for epoch in range(500):
    a2, cache = forward(p, X)
    losses.append(loss_fn(p, X, y))
    g = backward(p, cache, y)
    for k in p:
        p[k] -= lr * g[k]
acc = ((forward(p, X)[0] > 0.5).astype(int) == y).mean()
print(f"\nTrained 500 steps with our own gradients: loss {losses[0]:.3f} -> {losses[-1]:.3f}, "
      f"accuracy {acc:.3f}")

# --- Plot: per-parameter gradient comparison and loss curve ----------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
a_flat = np.concatenate([analytic[k].ravel() for k in params])
n_flat = np.concatenate([numeric[k].ravel() for k in params])
axes[0].scatter(n_flat, a_flat, s=18, alpha=0.7)
lim = [min(n_flat.min(), a_flat.min()), max(n_flat.max(), a_flat.max())]
axes[0].plot(lim, lim, "r--", lw=1)
axes[0].set_xlabel("numerical gradient")
axes[0].set_ylabel("backprop gradient")
axes[0].set_title("Gradient check: points sit on the line y = x")
axes[1].plot(losses)
axes[1].set_xlabel("step")
axes[1].set_ylabel("BCE loss")
axes[1].set_title("Training with our own backward()")
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig("lab2a_gradient_check.png", dpi=130)
print("Saved lab2a_gradient_check.png")

print("""
Questions to answer in your notebook:
1. Why does dL/dz2 simplify to (a2 - y) / N for sigmoid + cross-entropy?
2. Where does the factor (1 - a1**2) come from, and what happens to it when tanh saturates?
3. Gradient checking costs 2 forward passes per parameter. Why is it used only for debugging?
4. Why check with float64 and a relative error, rather than comparing raw differences?
""")
