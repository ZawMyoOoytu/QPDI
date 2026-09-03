from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_FILE = PROJECT_ROOT / "quantum_engine" / "qubo_matrix.npy"
CANDIDATE_FILE = PROJECT_ROOT / "data" / "quantum_candidates.csv"


Q = np.load(QUBO_FILE)
df = pd.read_csv(CANDIDATE_FILE)

N = len(df)
K = 5


def energy(indices):
    x = np.zeros(N, dtype=int)
    x[list(indices)] = 1
    return float(x @ Q @ x)


print()
print("========================================")
print(" QPDI — BENCHMARK CONSISTENCY CHECK")
print("========================================")

print()
print(f"QUBO shape : {Q.shape}")
print(f"Candidates : {N}")
print(f"Budget K   : {K}")

# ------------------------------------------------------------
# Direct check of the previously identified optimum
# ------------------------------------------------------------

known_bits = "00011111"

known_indices = tuple(
    i
    for i, bit in enumerate(known_bits[::-1])
    if bit == "1"
)

known_energy = energy(known_indices)

print()
print("----------------------------------------")
print(" DIRECT BITSTRING CHECK")
print("----------------------------------------")

print(
    f"Bitstring : {known_bits}"
)

print(
    f"Indices   : {known_indices}"
)

print(
    f"Energy    : {known_energy:.9f}"
)

print(
    "Regions   : "
    + ", ".join(
        str(df.iloc[i]["region"])
        for i in known_indices
    )
)


# ------------------------------------------------------------
# Exhaustive classical search
# ------------------------------------------------------------

best_energy = float("inf")
best_indices = None

evaluated = 0

for indices in combinations(
    range(N),
    K
):

    current_energy = energy(indices)

    evaluated += 1

    if current_energy < best_energy:

        best_energy = current_energy
        best_indices = indices


# ------------------------------------------------------------
# Build classical bitstring
# ------------------------------------------------------------

x = np.zeros(
    N,
    dtype=int
)

x[list(best_indices)] = 1

classical_bitstring = "".join(
    str(int(v))
    for v in x[::-1]
)

classical_regions = [
    str(df.iloc[i]["region"])
    for i in best_indices
]


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()
print("----------------------------------------")
print(" EXACT CLASSICAL SEARCH")
print("----------------------------------------")

print(
    f"Combinations : {evaluated}"
)

print(
    f"Bitstring    : {classical_bitstring}"
)

print(
    f"Energy       : {best_energy:.9f}"
)

print(
    "Regions      : "
    + ", ".join(classical_regions)
)


# ------------------------------------------------------------
# Consistency
# ------------------------------------------------------------

consistent = (
    classical_bitstring == known_bits
    and
    abs(best_energy - known_energy) < 1e-9
)


print()
print("----------------------------------------")
print(" CONSISTENCY")
print("----------------------------------------")

print(
    f"Known optimum energy : "
    f"{known_energy:.9f}"
)

print(
    f"Exact search energy  : "
    f"{best_energy:.9f}"
)

print(
    f"Consistent           : {consistent}"
)


print()
print("========================================")

if consistent:
    print(" ✓ BENCHMARK IS CONSISTENT")
else:
    print(" ✗ BENCHMARK INCONSISTENCY DETECTED")

print("========================================")