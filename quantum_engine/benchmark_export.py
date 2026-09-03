from pathlib import Path
from itertools import combinations
import json

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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "benchmark_results.json"
)


# ============================================================
# CONFIGURATION
# ============================================================

QAOA_BITSTRING = "11100011"

QAOA_PROBABILITY = 0.201140

BUDGET_K = 5


# ============================================================
# LOAD
# ============================================================

Q = np.load(QUBO_FILE)

df = pd.read_csv(CANDIDATE_FILE)

N = len(df)


# ============================================================
# QUBO ENERGY
# ============================================================

def energy_from_indices(indices):
    x = np.zeros(N, dtype=int)

    for i in indices:
        x[i] = 1

    return float(x @ Q @ x)


# ============================================================
# QAOA DECODE
# ============================================================

# Qiskit bitstring → qubit order
qaoa_qubit_bits = QAOA_BITSTRING[::-1]

qaoa_indices = tuple(
    i
    for i, bit in enumerate(qaoa_qubit_bits)
    if bit == "1"
)

qaoa_energy = energy_from_indices(
    qaoa_indices
)

qaoa_regions = [
    str(df.iloc[i]["region"])
    for i in qaoa_indices
]


# ============================================================
# CLASSICAL EXACT SEARCH
# ============================================================

best_energy = float("inf")
best_indices = None

evaluated = 0

for indices in combinations(
    range(N),
    BUDGET_K
):

    energy = energy_from_indices(
        indices
    )

    evaluated += 1

    if energy < best_energy:

        best_energy = energy
        best_indices = indices


# ============================================================
# CLASSICAL RESULT
# ============================================================

classical_indices = tuple(
    best_indices
)

classical_x = np.zeros(
    N,
    dtype=int
)

for i in classical_indices:
    classical_x[i] = 1


classical_bitstring = "".join(
    str(int(v))
    for v in classical_x[::-1]
)

classical_regions = [
    str(df.iloc[i]["region"])
    for i in classical_indices
]


# ============================================================
# COMPARISON
# ============================================================

energy_gap = (
    qaoa_energy
    - best_energy
)

relative_gap_percent = (
    abs(energy_gap)
    / abs(best_energy)
    * 100
)

same_solution = (
    set(qaoa_indices)
    == set(classical_indices)
)

intersection = (
    set(qaoa_indices)
    & set(classical_indices)
)

union = (
    set(qaoa_indices)
    | set(classical_indices)
)

jaccard = (
    len(intersection)
    / len(union)
)


# ============================================================
# BUILD RESULT
# ============================================================

result = {

    "project": "QPDI",

    "experiment": {
        "type": "synthetic_policy_optimization",
        "candidate_count": N,
        "selection_budget": BUDGET_K,
        "classical_search_space": evaluated,
    },

    "qaoa": {
        "bitstring": QAOA_BITSTRING,
        "probability": QAOA_PROBABILITY,
        "energy": round(qaoa_energy, 9),
        "selected_regions": qaoa_regions,
    },

    "classical_exact": {
        "bitstring": classical_bitstring,
        "energy": round(best_energy, 9),
        "selected_regions": classical_regions,
    },

    "comparison": {
        "same_solution": same_solution,
        "energy_gap": round(
            energy_gap,
            9
        ),
        "relative_gap_percent": round(
            relative_gap_percent,
            6
        ),
        "jaccard_similarity": round(
            jaccard,
            6
        ),
    },

    "validation": {
        "qaoa_selected_count": len(
            qaoa_indices
        ),
        "classical_selected_count": len(
            classical_indices
        ),
        "exact_k_constraint": (
            len(qaoa_indices)
            == BUDGET_K
            and
            len(classical_indices)
            == BUDGET_K
        ),
    },

    "research_interpretation": (
        "For this synthetic 8-candidate instance, "
        "the QAOA most-probable solution matches "
        "the exact classical optimum. This result "
        "does not demonstrate quantum advantage."
    ),
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        result,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# DISPLAY
# ============================================================

print()
print("========================================")
print(" QPDI — FINAL BENCHMARK EXPORT")
print("========================================")

print()
print("QAOA")
print("----------------------------------------")
print(
    f"Bitstring : {QAOA_BITSTRING}"
)
print(
    f"Energy    : {qaoa_energy:.6f}"
)
print(
    f"Probability: {QAOA_PROBABILITY:.6f}"
)
print(
    "Regions   : "
    + ", ".join(qaoa_regions)
)

print()
print("CLASSICAL EXACT")
print("----------------------------------------")
print(
    f"Bitstring : {classical_bitstring}"
)
print(
    f"Energy    : {best_energy:.6f}"
)
print(
    "Regions   : "
    + ", ".join(classical_regions)
)

print()
print("COMPARISON")
print("----------------------------------------")
print(
    f"Same solution : {same_solution}"
)
print(
    f"Energy gap    : {energy_gap:.6f}"
)
print(
    f"Relative gap  : "
    f"{relative_gap_percent:.6f}%"
)
print(
    f"Jaccard       : {jaccard:.6f}"
)

print()
print("VALIDATION")
print("----------------------------------------")
print(
    f"Combinations evaluated : {evaluated}"
)
print(
    f"Exact-K valid          : "
    f"{result['validation']['exact_k_constraint']}"
)

print()
print("========================================")
print(" BENCHMARK RESULT SAVED")
print("========================================")

print(
    OUTPUT_FILE
)