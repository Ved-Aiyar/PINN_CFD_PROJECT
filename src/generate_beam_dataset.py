import numpy as np
import os
import csv

# ---------- Settings ----------
N_SAMPLES = 5000  # number of data points
OUTPUT_DIR = "data"
OUTPUT_FILE = "beam_cantilever_dataset.csv"

# Ranges for random sampling
L_MIN, L_MAX = 1.0, 5.0          # m
E_MIN, E_MAX = 1e10, 2.1e11      # Pa
I_MIN, I_MAX = 1e-6, 1e-4        # m^4
P_MIN, P_MAX = 100.0, 5000.0     # N

# ---------- Make sure output folder exists ----------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Sample random inputs ----------
# Uniform sampling in the given ranges
L = np.random.uniform(L_MIN, L_MAX, N_SAMPLES)
E = np.random.uniform(E_MIN, E_MAX, N_SAMPLES)
I = np.random.uniform(I_MIN, I_MAX, N_SAMPLES)
P = np.random.uniform(P_MIN, P_MAX, N_SAMPLES)

# ---------- Compute max deflection for cantilever with end load ----------
# Formula: delta_max = P * L^3 / (3 * E * I)
delta_max = P * L**3 / (3.0 * E * I)

# ---------- Write to CSV ----------
output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE)

with open(output_path, mode="w", newline="") as f:
    writer = csv.writer(f)
    # header
    writer.writerow(["L", "E", "I", "P", "delta_max"])
    # rows
    for l, e, i, p, d in zip(L, E, I, P, delta_max):
        writer.writerow([l, e, i, p, d])

print(f"Dataset saved to: {output_path}")
print(f"Total samples: {N_SAMPLES}")