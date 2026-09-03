from pathlib import Path
import json
import numpy as np


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qaoa_distribution.json"
)


# ============================================================
# LOAD DATA
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Distribution file not found:\n{INPUT_FILE}\n\n"
        "Run qaoa_distribution.py first."
    )


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


states = data["all_valid_states"]

classical = data["classical_optimum"]

classical_energy = float(
    classical["energy"]
)

classical_qubo_bitstring = (
    classical["qubo_bitstring"]
)

classical_rank = (
    classical["rank_in_qaoa_valid_distribution"]
)

classical_probability = (
    classical["probability"]
)


# ============================================================
# BASIC INFORMATION
# ============================================================

candidate_count = data[
    "experiment"
]["candidate_count"]

selection_budget = data[
    "experiment"
]["selection_budget"]

total_valid_states = len(states)


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_top_n(n):

    subset = states[:n]

    probabilities = [
        float(state["probability"])
        for state in subset
    ]

    energies = [
        float(state["energy"])
        for state in subset
    ]

    cumulative_probability = sum(
        probabilities
    )

    average_energy = (
        float(np.mean(energies))
        if energies
        else None
    )

    best_energy_state = min(
        subset,
        key=lambda state: state["energy"]
    )

    best_probability_state = max(
        subset,
        key=lambda state: state["probability"]
    )

    return {
        "n": n,

        "cumulative_probability":
            cumulative_probability,

        "cumulative_probability_percent":
            cumulative_probability * 100,

        "average_energy":
            average_energy,

        "best_energy":
            float(
                best_energy_state["energy"]
            ),

        "best_energy_bitstring":
            best_energy_state[
                "qubo_bitstring"
            ],

        "best_energy_regions":
            best_energy_state[
                "regions"
            ],

        "highest_probability":
            float(
                best_probability_state[
                    "probability"
                ]
            ),

        "highest_probability_bitstring":
            best_probability_state[
                "qubo_bitstring"
            ],

        "highest_probability_regions":
            best_probability_state[
                "regions"
            ]
    }


# ============================================================
# TOP ANALYSIS
# ============================================================

top_1 = analyze_top_n(1)
top_5 = analyze_top_n(5)
top_10 = analyze_top_n(
    min(10, total_valid_states)
)


# ============================================================
# ENERGY GAP
# ============================================================

top_1_energy_gap = (
    top_1["best_energy"]
    - classical_energy
)

top_5_energy_gap = (
    top_5["best_energy"]
    - classical_energy
)

top_10_energy_gap = (
    top_10["best_energy"]
    - classical_energy
)


# ============================================================
# RELATIVE ENERGY GAP
# ============================================================

def relative_gap(energy):

    denominator = abs(
        classical_energy
    )

    if denominator < 1e-12:
        return None

    return (
        (energy - classical_energy)
        / denominator
    )


top_1_relative_gap = relative_gap(
    top_1["best_energy"]
)

top_5_relative_gap = relative_gap(
    top_5["best_energy"]
)

top_10_relative_gap = relative_gap(
    top_10["best_energy"]
)


# ============================================================
# CLASSICAL OPTIMUM PRESENCE
# ============================================================

def find_state(bitstring):

    for rank, state in enumerate(
        states,
        start=1
    ):

        if (
            state["qubo_bitstring"]
            == bitstring
        ):

            return rank, state

    return None, None


optimum_rank, optimum_state = (
    find_state(
        classical_qubo_bitstring
    )
)


# ============================================================
# TOP-N SUCCESS FLAGS
# ============================================================

success_top_1 = (
    optimum_rank is not None
    and optimum_rank <= 1
)

success_top_5 = (
    optimum_rank is not None
    and optimum_rank <= 5
)

success_top_10 = (
    optimum_rank is not None
    and optimum_rank <= 10
)


# ============================================================
# BEST ENERGY RANK
# ============================================================

best_energy_state = min(
    states,
    key=lambda state: state["energy"]
)

best_energy_rank = (
    states.index(
        best_energy_state
    ) + 1
)


# ============================================================
# OUTPUT
# ============================================================

print()
print("========================================")
print(" QPDI — QAOA DISTRIBUTION ANALYSIS")
print("========================================")

print()
print("PROBLEM")
print("----------------------------------------")

print(
    f"Candidate count      : "
    f"{candidate_count}"
)

print(
    f"Selection budget     : "
    f"{selection_budget}"
)

print(
    f"Exact-K states       : "
    f"{total_valid_states}"
)


print()
print("CLASSICAL OPTIMUM")
print("----------------------------------------")

print(
    f"QUBO bitstring       : "
    f"{classical_qubo_bitstring}"
)

print(
    f"Energy               : "
    f"{classical_energy:.9f}"
)

print(
    f"QAOA distribution rank: "
    f"{classical_rank}"
)

print(
    f"Probability          : "
    f"{classical_probability:.12f}"
)

print(
    f"Probability (%)      : "
    f"{classical_probability * 100:.8f}%"
)


print()
print("TOP-1")
print("----------------------------------------")

print(
    f"Cumulative probability : "
    f"{top_1['cumulative_probability_percent']:.6f}%"
)

print(
    f"Average energy         : "
    f"{top_1['average_energy']:.9f}"
)

print(
    f"Best energy            : "
    f"{top_1['best_energy']:.9f}"
)

print(
    f"Energy gap             : "
    f"{top_1_energy_gap:.9f}"
)

print(
    f"Relative gap           : "
    f"{top_1_relative_gap * 100:.6f}%"
)

print(
    f"Classical optimum in Top-1: "
    f"{success_top_1}"
)


print()
print("TOP-5")
print("----------------------------------------")

print(
    f"Cumulative probability : "
    f"{top_5['cumulative_probability_percent']:.6f}%"
)

print(
    f"Average energy         : "
    f"{top_5['average_energy']:.9f}"
)

print(
    f"Best energy            : "
    f"{top_5['best_energy']:.9f}"
)

print(
    f"Energy gap             : "
    f"{top_5_energy_gap:.9f}"
)

print(
    f"Relative gap           : "
    f"{top_5_relative_gap * 100:.6f}%"
)

print(
    f"Classical optimum in Top-5: "
    f"{success_top_5}"
)


print()
print("TOP-10")
print("----------------------------------------")

print(
    f"Cumulative probability : "
    f"{top_10['cumulative_probability_percent']:.6f}%"
)

print(
    f"Average energy         : "
    f"{top_10['average_energy']:.9f}"
)

print(
    f"Best energy            : "
    f"{top_10['best_energy']:.9f}"
)

print(
    f"Energy gap             : "
    f"{top_10_energy_gap:.9f}"
)

print(
    f"Relative gap           : "
    f"{top_10_relative_gap * 100:.6f}%"
)

print(
    f"Classical optimum in Top-10: "
    f"{success_top_10}"
)


print()
print("BEST ENERGY STATE")
print("----------------------------------------")

print(
    f"Rank                  : "
    f"{best_energy_rank}"
)

print(
    f"QUBO bitstring        : "
    f"{best_energy_state['qubo_bitstring']}"
)

print(
    f"Energy                : "
    f"{best_energy_state['energy']:.9f}"
)

print(
    f"Probability           : "
    f"{best_energy_state['probability']:.12f}"
)

print(
    f"Regions               : "
    f"{', '.join(best_energy_state['regions'])}"
)


print()
print("========================================")
print(" INTERPRETATION")
print("========================================")

if success_top_1:

    print(
        "The classical optimum is the "
        "most probable QAOA solution."
    )

else:

    print(
        "The classical optimum is NOT the "
        "most probable QAOA solution."
    )


if success_top_10:

    print(
        "The classical optimum appears "
        "within the QAOA Top-10 distribution."
    )

else:

    print(
        "The classical optimum does not "
        "appear within the QAOA Top-10."
    )


print()
print(
    "Note: Top-N probability is a "
    "distribution metric, not evidence "
    "of quantum advantage."
)

print(
    "This experiment uses a small "
    "synthetic policy-selection problem."
)

print()
print("========================================")
print(" ANALYSIS COMPLETE")
print("========================================")