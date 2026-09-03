from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qubo_matrix.npy"
)

CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "quantum_candidates.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

BUDGET_UNITS = 5


# QAOA result from Step 27
QAOA_BITSTRING = "11100011"


# ============================================================
# LOAD DATA
# ============================================================

if not QUBO_FILE.exists():
    raise FileNotFoundError(
        f"QUBO file not found:\n{QUBO_FILE}"
    )

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate file not found:\n{CANDIDATE_FILE}"
    )


Q = np.load(QUBO_FILE)

df = pd.read_csv(
    CANDIDATE_FILE
)


N = Q.shape[0]


# ============================================================
# VALIDATION
# ============================================================

if Q.shape != (len(df), len(df)):
    raise ValueError(
        "QUBO dimensions do not match "
        "candidate count."
    )

if len(QAOA_BITSTRING) != N:
    raise ValueError(
        "QAOA bitstring length does not "
        "match QUBO size."
    )


# ============================================================
# QUBO ENERGY
# ============================================================

def qubo_energy(
    bitstring
):

    x = np.array(
        [
            int(bit)
            for bit in bitstring
        ],
        dtype=float
    )

    return float(
        x @ Q @ x
    )


# ============================================================
# QAOA BITSTRING → QUBO VARIABLE ORDER
# ============================================================

# Qiskit displays bitstrings in reverse
# relative to qubit indexing.

qaoa_qubit_bits = (
    QAOA_BITSTRING[::-1]
)


qaoa_selected_indices = [
    i
    for i, bit in enumerate(
        qaoa_qubit_bits
    )
    if bit == "1"
]


if len(qaoa_selected_indices) != BUDGET_UNITS:
    raise ValueError(
        "QAOA bitstring violates "
        "the exact-K constraint."
    )


qaoa_x = np.zeros(
    N,
    dtype=int
)

for i in qaoa_selected_indices:
    qaoa_x[i] = 1


qaoa_energy = float(
    qaoa_x @ Q @ qaoa_x
)


# ============================================================
# CLASSICAL EXACT SEARCH
# ============================================================

best_energy = float("inf")

best_indices = None

evaluated = 0


for selected_indices in combinations(
    range(N),
    BUDGET_UNITS
):

    x = np.zeros(
        N,
        dtype=int
    )

    for i in selected_indices:
        x[i] = 1

    energy = float(
        x @ Q @ x
    )

    evaluated += 1

    if energy < best_energy:

        best_energy = energy

        best_indices = (
            selected_indices
        )


# ============================================================
# CLASSICAL OPTIMAL BITSTRING
# ============================================================

classical_x = np.zeros(
    N,
    dtype=int
)

for i in best_indices:
    classical_x[i] = 1


classical_bitstring = "".join(
    str(int(value))
    for value in classical_x[::-1]
)


# ============================================================
# SELECTED REGIONS
# ============================================================

qaoa_regions = [
    df.iloc[i]["region"]
    for i in qaoa_selected_indices
]


classical_regions = [
    df.iloc[i]["region"]
    for i in best_indices
]


# ============================================================
# ENERGY COMPARISON
# ============================================================

energy_gap = (
    qaoa_energy
    - best_energy
)


if abs(best_energy) > 1e-12:

    relative_gap = (
        abs(energy_gap)
        / abs(best_energy)
    ) * 100

else:

    relative_gap = 0.0


same_solution = (
    set(qaoa_selected_indices)
    == set(best_indices)
)


# ============================================================
# OVERLAP
# ============================================================

qaoa_set = set(
    qaoa_selected_indices
)

classical_set = set(
    best_indices
)

intersection = (
    qaoa_set
    & classical_set
)

union = (
    qaoa_set
    | classical_set
)

jaccard_similarity = (
    len(intersection)
    / len(union)
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("========================================")
print(" QPDI — CLASSICAL EXACT BENCHMARK")
print("========================================")

print(
    f"Candidates        : {N}"
)

print(
    f"Budget K          : {BUDGET_UNITS}"
)

print(
    f"Possible solutions: "
    f"{evaluated}"
)


# ============================================================
# QAOA RESULT
# ============================================================

print()
print("========================================")
print(" QAOA SOLUTION")
print("========================================")

print(
    f"Bitstring : {QAOA_BITSTRING}"
)

print(
    f"Energy    : {qaoa_energy:.6f}"
)

print(
    f"Selected  : {len(qaoa_regions)}"
)

print(
    "Regions   : "
    + ", ".join(qaoa_regions)
)


# ============================================================
# CLASSICAL RESULT
# ============================================================

print()
print("========================================")
print(" CLASSICAL EXACT OPTIMUM")
print("========================================")

print(
    f"Bitstring : {classical_bitstring}"
)

print(
    f"Energy    : {best_energy:.6f}"
)

print(
    f"Selected  : {len(classical_regions)}"
)

print(
    "Regions   : "
    + ", ".join(classical_regions)
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("========================================")
print(" QAOA vs CLASSICAL")
print("========================================")

print(
    f"Energy gap      : "
    f"{energy_gap:.6f}"
)

print(
    f"Relative gap    : "
    f"{relative_gap:.4f}%"
)

print(
    f"Same solution   : "
    f"{same_solution}"
)

print(
    f"Jaccard overlap : "
    f"{jaccard_similarity:.4f}"
)


# ============================================================
# FINAL ASSESSMENT
# ============================================================

print()
print("========================================")
print(" BENCHMARK ASSESSMENT")
print("========================================")


if same_solution:

    print(
        "✓ QAOA recovered the exact "
        "classical optimum."
    )

elif energy_gap >= 0:

    print(
        "QAOA returned a valid solution "
        "but not the exact classical optimum."
    )

else:

    print(
        "WARNING: QAOA energy is lower "
        "than the exact benchmark."
    )


print()
print(
    "Important:"
)

print(
    "This benchmark evaluates solution "
    "quality on the current synthetic "
    "8-candidate problem."
)

print(
    "It does NOT demonstrate quantum "
    "advantage."
)