from pathlib import Path
import json


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "final_qaoa_benchmark.json"
)


# ============================================================
# LOAD
# ============================================================

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:

    result = json.load(f)


classical = result["classical_exact"]
qaoa = result["qaoa"]
comparison = result["comparison"]
optimum = result["optimum_probability"]

most_probable = qaoa["most_probable_valid_state"]

classical_energy = float(
    classical["energy"]
)

qaoa_energy = float(
    most_probable["energy"]
)


# ============================================================
# APPROXIMATION QUALITY
# ============================================================

# For a minimization problem, define the absolute
# energy approximation error.

absolute_error = (
    qaoa_energy - classical_energy
)

relative_error = (
    abs(absolute_error)
    / abs(classical_energy)
    * 100
)


# A simple energy-quality score:
#
#   1.0 = exact optimum
#   lower = farther from optimum
#
# Since energies are negative, use the magnitudes
# relative to the classical optimum.

approximation_ratio = (
    abs(classical_energy)
    / abs(qaoa_energy)
)


# ============================================================
# QAOA PROBABILITY
# ============================================================

most_probable_probability = float(
    most_probable["probability"]
)

classical_optimum_probability = float(
    optimum["probability"]
)


# ============================================================
# JACCARD
# ============================================================

jaccard = float(
    comparison["jaccard_similarity"]
)


# ============================================================
# TOP STATES
# ============================================================

# The current benchmark JSON stores the most probable
# valid state, but not the complete valid-state list.
#
# Therefore this step reports the metrics available
# from the final benchmark.


# ============================================================
# PRINT
# ============================================================

print()
print("========================================")
print(" QPDI — QAOA PERFORMANCE ANALYSIS")
print("========================================")


print()
print("PROBLEM")
print("----------------------------------------")

print(
    f"Candidate count       : "
    f"{result['experiment']['candidate_count']}"
)

print(
    f"Selection budget      : "
    f"{result['experiment']['selection_budget']}"
)

print(
    f"Classical combinations: "
    f"{result['experiment']['classical_combinations']}"
)

print(
    f"QAOA reps             : "
    f"{result['experiment']['qaoa_reps']}"
)


print()
print("CLASSICAL OPTIMUM")
print("----------------------------------------")

print(
    f"Bitstring : "
    f"{classical['qiskit_bitstring']}"
)

print(
    f"Energy    : "
    f"{classical_energy:.9f}"
)

print(
    "Regions   : "
    + ", ".join(classical["regions"])
)


print()
print("QAOA MOST PROBABLE VALID STATE")
print("----------------------------------------")

print(
    f"Qiskit bitstring : "
    f"{most_probable['qiskit_bitstring']}"
)

print(
    f"QUBO bitstring   : "
    f"{most_probable['qubo_bitstring']}"
)

print(
    f"Energy           : "
    f"{qaoa_energy:.9f}"
)

print(
    f"Probability      : "
    f"{most_probable_probability:.9f}"
)

print(
    f"Probability (%)  : "
    f"{most_probable_probability * 100:.6f}%"
)

print(
    "Regions          : "
    + ", ".join(most_probable["regions"])
)


print()
print("OPTIMUM PROBABILITY")
print("----------------------------------------")

print(
    f"Classical optimum probability : "
    f"{classical_optimum_probability:.9f}"
)

print(
    f"Percentage                    : "
    f"{classical_optimum_probability * 100:.6f}%"
)


print()
print("ENERGY QUALITY")
print("----------------------------------------")

print(
    f"Classical energy : "
    f"{classical_energy:.9f}"
)

print(
    f"QAOA energy      : "
    f"{qaoa_energy:.9f}"
)

print(
    f"Absolute error   : "
    f"{absolute_error:.9f}"
)

print(
    f"Relative error   : "
    f"{relative_error:.6f}%"
)

print(
    f"Approximation ratio : "
    f"{approximation_ratio:.6f}"
)


print()
print("SOLUTION OVERLAP")
print("----------------------------------------")

print(
    f"Same solution : "
    f"{comparison['same_solution']}"
)

print(
    f"Jaccard       : "
    f"{jaccard:.6f}"
)


print()
print("========================================")
print(" INTERPRETATION")
print("========================================")

print(
    "The depth-1 QAOA result does not reach "
    "the exact classical optimum as its most "
    "probable valid solution."
)

print(
    "The classical optimum is present in the "
    "QAOA probability distribution, but with "
    "very low probability."
)

print(
    "The approximation ratio is a descriptive "
    "benchmark metric for this minimization "
    "experiment; it is not evidence of quantum "
    "advantage."
)

print(
    "The experiment uses a small synthetic "
    "8-variable policy-selection problem and "
    "does not represent real political prediction."
)