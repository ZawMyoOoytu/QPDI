from pathlib import Path
import json


# ============================================================
# PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qaoa_distribution.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "benchmark_report.json"
)


# ============================================================
# LOAD
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}\n\n"
        "Run qaoa_distribution.py first."
    )


with open(
    INPUT_FILE,
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)


# ============================================================
# DATA
# ============================================================

experiment = data["experiment"]

classical = data["classical_optimum"]

states = data["all_valid_states"]

classical_energy = float(
    classical["energy"]
)

classical_bitstring = (
    classical["qubo_bitstring"]
)

classical_regions = (
    classical["regions"]
)

classical_probability = float(
    classical["probability"]
)

classical_rank = int(
    classical["rank_in_qaoa_valid_distribution"]
)


# ============================================================
# HELPER
# ============================================================

def analyze_top_n(n):

    subset = states[:n]

    probabilities = [
        float(s["probability"])
        for s in subset
    ]

    energies = [
        float(s["energy"])
        for s in subset
    ]

    best_state = min(
        subset,
        key=lambda s: s["energy"]
    )

    cumulative_probability = sum(
        probabilities
    )

    average_energy = (
        sum(
            p * e
            for p, e in zip(
                probabilities,
                energies
            )
        )
        / cumulative_probability
        if cumulative_probability > 0
        else None
    )

    energy_gap = (
        best_state["energy"]
        - classical_energy
    )

    relative_gap = (
        energy_gap
        / abs(classical_energy)
    )

    return {
        "count": n,

        "cumulative_probability":
            cumulative_probability,

        "cumulative_probability_percent":
            cumulative_probability * 100,

        "probability_weighted_average_energy":
            average_energy,

        "best_energy":
            float(
                best_state["energy"]
            ),

        "best_energy_gap":
            float(energy_gap),

        "best_energy_relative_gap":
            float(relative_gap),

        "best_energy_relative_gap_percent":
            float(relative_gap * 100),

        "best_state":
            {
                "qiskit_bitstring":
                    best_state[
                        "qiskit_bitstring"
                    ],

                "qubo_bitstring":
                    best_state[
                        "qubo_bitstring"
                    ],

                "probability":
                    float(
                        best_state[
                            "probability"
                        ]
                    ),

                "energy":
                    float(
                        best_state[
                            "energy"
                        ]
                    ),

                "regions":
                    best_state[
                        "regions"
                    ]
            }
    }


# ============================================================
# TOP-N
# ============================================================

top_1 = analyze_top_n(1)

top_5 = analyze_top_n(5)

top_10 = analyze_top_n(
    min(10, len(states))
)


# ============================================================
# QAOA MOST PROBABLE STATE
# ============================================================

most_probable = states[0]

most_probable_energy = float(
    most_probable["energy"]
)

most_probable_gap = (
    most_probable_energy
    - classical_energy
)

most_probable_relative_gap = (
    most_probable_gap
    / abs(classical_energy)
)


# ============================================================
# EXPECTED ENERGY
# ============================================================

expected_energy = float(
    data[
        "qaoa_parameters"
    ][
        "expected_qubo_energy"
    ]
)

expected_energy_gap = (
    expected_energy
    - classical_energy
)

expected_relative_gap = (
    expected_energy_gap
    / abs(classical_energy)
)


# ============================================================
# QAOA EXACT OPTIMUM PROBABILITY
# ============================================================

optimum_probability = (
    classical_probability
)

optimum_probability_percent = (
    optimum_probability * 100
)


# ============================================================
# BENCHMARK RESULT
# ============================================================

benchmark = {

    "problem": {

        "candidate_count":
            experiment[
                "candidate_count"
            ],

        "selection_budget":
            experiment[
                "selection_budget"
            ],

        "exact_k_states":
            len(states),

        "qaoa_reps":
            experiment[
                "qaoa_reps"
            ],

        "grid_points":
            experiment[
                "grid_points"
            ],

        "evaluations":
            experiment[
                "evaluations"
            ]
    },

    "classical_exact": {

        "bitstring":
            classical_bitstring,

        "energy":
            classical_energy,

        "regions":
            classical_regions,

        "rank_in_qaoa_distribution":
            classical_rank,

        "qaoa_probability":
            optimum_probability,

        "qaoa_probability_percent":
            optimum_probability_percent
    },

    "qaoa_most_probable": {

        "bitstring":
            most_probable[
                "qubo_bitstring"
            ],

        "qiskit_bitstring":
            most_probable[
                "qiskit_bitstring"
            ],

        "energy":
            most_probable_energy,

        "probability":
            most_probable[
                "probability"
            ],

        "probability_percent":
            most_probable[
                "probability"
            ] * 100,

        "energy_gap":
            most_probable_gap,

        "relative_energy_gap":
            most_probable_relative_gap,

        "relative_energy_gap_percent":
            most_probable_relative_gap * 100,

        "regions":
            most_probable[
                "regions"
            ]
    },

    "qaoa_expected_energy": {

        "energy":
            expected_energy,

        "energy_gap":
            expected_energy_gap,

        "relative_gap":
            expected_relative_gap,

        "relative_gap_percent":
            expected_relative_gap * 100
    },

    "top_1": top_1,

    "top_5": top_5,

    "top_10": top_10,

    "conclusion": {

        "classical_optimum_is_qaoa_top_1":
            classical_rank == 1,

        "classical_optimum_is_qaoa_top_5":
            classical_rank <= 5,

        "classical_optimum_is_qaoa_top_10":
            classical_rank <= 10,

        "quantum_advantage_demonstrated":
            False,

        "benchmark_scope":
            "small synthetic policy-selection "
            "research problem",

        "interpretation":
            "The QAOA simulation does not reach "
            "the classical optimum as its most "
            "probable solution. Classical exact "
            "search remains the reference optimum."
    }
}


# ============================================================
# SAVE
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        benchmark,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PRINT REPORT
# ============================================================

print()
print("========================================")
print(" QPDI — CLASSICAL vs QAOA BENCHMARK")
print("========================================")


print()
print("PROBLEM")
print("----------------------------------------")

print(
    f"Candidates          : "
    f"{experiment['candidate_count']}"
)

print(
    f"Selection budget    : "
    f"{experiment['selection_budget']}"
)

print(
    f"Exact combinations  : "
    f"{len(states)}"
)

print(
    f"QAOA repetitions     : "
    f"{experiment['qaoa_reps']}"
)


print()
print("CLASSICAL EXACT")
print("----------------------------------------")

print(
    f"Bitstring           : "
    f"{classical_bitstring}"
)

print(
    f"Energy              : "
    f"{classical_energy:.9f}"
)

print(
    f"Regions             : "
    f"{', '.join(classical_regions)}"
)


print()
print("QAOA MOST PROBABLE")
print("----------------------------------------")

print(
    f"Bitstring           : "
    f"{most_probable['qubo_bitstring']}"
)

print(
    f"Energy              : "
    f"{most_probable_energy:.9f}"
)

print(
    f"Probability         : "
    f"{most_probable['probability'] * 100:.6f}%"
)

print(
    f"Energy gap          : "
    f"{most_probable_gap:.9f}"
)

print(
    f"Relative gap        : "
    f"{most_probable_relative_gap * 100:.6f}%"
)

print(
    f"Regions             : "
    f"{', '.join(most_probable['regions'])}"
)


print()
print("QAOA EXPECTED ENERGY")
print("----------------------------------------")

print(
    f"Expected energy     : "
    f"{expected_energy:.9f}"
)

print(
    f"Energy gap          : "
    f"{expected_energy_gap:.9f}"
)

print(
    f"Relative gap        : "
    f"{expected_relative_gap * 100:.6f}%"
)


print()
print("TOP-N BENCHMARK")
print("----------------------------------------")

print(
    f"Top-1 probability   : "
    f"{top_1['cumulative_probability_percent']:.6f}%"
)

print(
    f"Top-5 probability   : "
    f"{top_5['cumulative_probability_percent']:.6f}%"
)

print(
    f"Top-10 probability  : "
    f"{top_10['cumulative_probability_percent']:.6f}%"
)

print()

print(
    f"Top-1 best gap      : "
    f"{top_1['best_energy_relative_gap_percent']:.6f}%"
)

print(
    f"Top-5 best gap      : "
    f"{top_5['best_energy_relative_gap_percent']:.6f}%"
)

print(
    f"Top-10 best gap     : "
    f"{top_10['best_energy_relative_gap_percent']:.6f}%"
)


print()
print("CLASSICAL OPTIMUM IN QAOA")
print("----------------------------------------")

print(
    f"Distribution rank   : "
    f"{classical_rank} / {len(states)}"
)

print(
    f"Probability         : "
    f"{optimum_probability_percent:.8f}%"
)


print()
print("CONCLUSION")
print("----------------------------------------")

if classical_rank == 1:

    print(
        "Classical optimum is the "
        "QAOA most-probable solution."
    )

else:

    print(
        "Classical optimum is NOT the "
        "QAOA most-probable solution."
    )


if classical_rank <= 10:

    print(
        "Classical optimum appears "
        "within QAOA Top-10."
    )

else:

    print(
        "Classical optimum is outside "
        "QAOA Top-10."
    )


print()
print(
    "Quantum advantage demonstrated : False"
)

print(
    "Benchmark scope               : "
    "small synthetic research problem"
)


print()
print("========================================")
print(" BENCHMARK REPORT SAVED")
print("========================================")

print(OUTPUT_FILE)