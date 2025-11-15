import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------- Settings ----------------
DATA_PATH = os.path.join("data", "beam_cantilever_dataset.csv")
HIDDEN_SIZE = 16      # number of neurons in hidden layer
EPOCHS = 2000
LEARNING_RATE = 1e-3
TRAIN_SPLIT = 0.8     # 80% train, 20% test
SEED = 42

np.random.seed(SEED)

# ---------------- 1. Load dataset ----------------
df = pd.read_csv(DATA_PATH)

# Columns: L, E, I, P, delta_max
X = df[["L", "E", "I", "P"]].values    # shape (N, 4)
y = df[["delta_max"]].values          # shape (N, 1)

N, input_dim = X.shape
output_dim = 1

# ---------------- 2. Train / test split ----------------
indices = np.arange(N)
np.random.shuffle(indices)

train_size = int(TRAIN_SPLIT * N)
train_idx = indices[:train_size]
test_idx = indices[train_size:]

X_train = X[train_idx]
y_train = y[train_idx]
X_test = X[test_idx]
y_test = y[test_idx]

# ---------------- 3. Normalize data (standardization) ----------------
# Fit on train, apply to both train/test
x_mean = X_train.mean(axis=0, keepdims=True)
x_std = X_train.std(axis=0, keepdims=True) + 1e-12  # avoid /0

y_mean = y_train.mean(axis=0, keepdims=True)
y_std = y_train.std(axis=0, keepdims=True) + 1e-12

X_train_n = (X_train - x_mean) / x_std
X_test_n = (X_test - x_mean) / x_std

y_train_n = (y_train - y_mean) / y_std
y_test_n = (y_test - y_mean) / y_std

# ---------------- 4. Initialize network parameters ----------------
# Layer 1: 4 -> HIDDEN_SIZE
W1 = np.random.randn(input_dim, HIDDEN_SIZE) * 0.1
b1 = np.zeros((1, HIDDEN_SIZE))

# Layer 2: HIDDEN_SIZE -> 1
W2 = np.random.randn(HIDDEN_SIZE, output_dim) * 0.1
b2 = np.zeros((1, output_dim))

# ---------------- 5. Activation and its derivative ----------------
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    # derivative wrt input x (pre-activation)
    return 1.0 - np.tanh(x) ** 2

# ---------------- 6. Training loop ----------------
train_losses = []

for epoch in range(1, EPOCHS + 1):
    # ----- Forward pass -----
    # Layer 1
    z1 = X_train_n @ W1 + b1           # shape (N_train, HIDDEN_SIZE)
    a1 = tanh(z1)                      # shape (N_train, HIDDEN_SIZE)

    # Layer 2 (output)
    z2 = a1 @ W2 + b2                  # shape (N_train, 1)
    y_pred = z2                        # linear output for regression

    # ----- Loss (MSE) -----
    # Mean squared error on normalized targets
    diff = y_pred - y_train_n          # shape (N_train, 1)
    loss = np.mean(diff ** 2)
    train_losses.append(loss)

    # ----- Backward pass -----
    # dL/dy_pred
    dL_dy_pred = 2.0 * diff / len(X_train_n)   # shape (N_train, 1)

    # Gradients for W2, b2
    # y_pred = a1 @ W2 + b2
    dL_dW2 = a1.T @ dL_dy_pred                 # shape (HIDDEN_SIZE, 1)
    dL_db2 = np.sum(dL_dy_pred, axis=0, keepdims=True)  # shape (1, 1)

    # Backprop into a1
    dL_da1 = dL_dy_pred @ W2.T                 # shape (N_train, HIDDEN_SIZE)

    # Backprop through tanh: a1 = tanh(z1)
    dL_dz1 = dL_da1 * tanh_derivative(z1)      # shape (N_train, HIDDEN_SIZE)

    # Gradients for W1, b1
    dL_dW1 = X_train_n.T @ dL_dz1              # shape (4, HIDDEN_SIZE)
    dL_db1 = np.sum(dL_dz1, axis=0, keepdims=True)  # shape (1, HIDDEN_SIZE)

    # ----- Gradient descent update -----
    W2 -= LEARNING_RATE * dL_dW2
    b2 -= LEARNING_RATE * dL_db2
    W1 -= LEARNING_RATE * dL_dW1
    b1 -= LEARNING_RATE * dL_db1

    # ----- Print progress occasionally -----
    if epoch % 200 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d} | Train Loss (normalized): {loss:.6f}")

# ---------------- 7. Evaluate on test set ----------------
# Forward pass on test data
z1_test = X_test_n @ W1 + b1
a1_test = tanh(z1_test)
z2_test = a1_test @ W2 + b2
y_test_pred_n = z2_test

# Inverse transform to original scale
y_test_true = y_test
y_test_pred = y_test_pred_n * y_std + y_mean

# Compute metrics in original scale
mse = np.mean((y_test_pred - y_test_true) ** 2)
mae = np.mean(np.abs(y_test_pred - y_test_true))
# R^2
ss_res = np.sum((y_test_true - y_test_pred) ** 2)
ss_tot = np.sum((y_test_true - y_mean) ** 2)
r2 = 1 - ss_res / ss_tot

print("\nTest set performance (original scale):")
print(f"MSE: {mse:.6e}")
print(f"MAE: {mae:.6e}")
print(f"R^2: {r2:.4f}")

# ---------------- 8. Plots ----------------
os.makedirs("results", exist_ok=True)

# Loss curve
plt.figure()
plt.plot(train_losses)
plt.xlabel("Epoch")
plt.ylabel("Train MSE (normalized)")
plt.title("Training Loss (NumPy NN)")
plt.savefig("results/beam_numpy_loss_curve.png", dpi=150)

# True vs Predicted
plt.figure()
plt.scatter(y_test_true, y_test_pred, s=10, alpha=0.7)
min_val = min(y_test_true.min(), y_test_pred.min())
max_val = max(y_test_true.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect fit")
plt.xlabel("True delta_max")
plt.ylabel("Predicted delta_max")
plt.title("Beam Surrogate (NumPy NN): True vs Predicted")
plt.legend()
plt.savefig("results/beam_numpy_true_vs_pred.png", dpi=150)

plt.show()