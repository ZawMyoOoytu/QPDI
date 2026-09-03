from pathlib import Path
from itertools import combinations
import json

import numpy as np
import pandas as pd

from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_FILE = PROJECT_ROOT / "quantum_engine" / "qubo_matrix.npy"
CANDIDATE_FILE = PROJECT_ROOT / "data" / "quantum_candidates.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "final_qaoa_benchmark.json"
)


# ============================================================
# CONFIG
# ============================================================

REPS = 1
GRID_POINTS = 21
K = 5


# ============================================================
# LOAD
# ============================================================

Q = np.load(QUBO_FILE)
df = pd.read_csv(CANDIDATE_FILE)

N = Q.shape[0]

if Q.shape != (len(df), len(df)):
    raise ValueError(
        "QUBO dimensions do not match candidate count."
    )


# ============================================================
# QUBO ENERGY
# ============================================================

def qubo_energy(x):
    """
    E(x) = x^T Q x
    """

    x = np.asarray(x, dtype=float)

    return float(x @ Q @ x)


# ============================================================
# QUBO → ISING
# ============================================================

def qubo_to_ising(Q):
    """
    Convert:

        E(x) = x^T Q x

    using:

        x_i = (1 - Z_i) / 2

    For upper-triangular QUBO convention:

        q_ij x_i x_j
        =
        q_ij/4 *
        (1 - Z_i - Z_j + Z_i Z_j)
    """

    n = Q.shape[0]

    linear = np.zeros(n)
    coupling = np.zeros((n, n))
    constant = 0.0

    # --------------------------------------------------------
    # Diagonal terms
    # --------------------------------------------------------

    for i in range(n):

        qii = Q[i, i]

        constant += qii / 2.0
        linear[i] -= qii / 2.0

    # --------------------------------------------------------
    # Off-diagonal terms
    # --------------------------------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            qij = Q[i, j]

            if abs(qij) < 1e-12:
                continue

            coefficient = qij / 4.0

            constant += coefficient

            linear[i] -= coefficient
            linear[j] -= coefficient

            coupling[i, j] += coefficient

    return linear, coupling, constant


# ============================================================
# BUILD ISING HAMILTONIAN
# ============================================================

linear, coupling, constant = qubo_to_ising(Q)

pauli_terms = []


# ------------------------------------------------------------
# Z terms
# ------------------------------------------------------------

for i in range(N):

    if abs(linear[i]) < 1e-12:
        continue

    label = ["I"] * N

    # Qiskit Pauli label uses highest qubit index on left.
    label[N - 1 - i] = "Z"

    pauli_terms.append(
        (
            "".join(label),
            float(linear[i])
        )
    )


# ------------------------------------------------------------
# ZZ terms
# ------------------------------------------------------------

for i in range(N):

    for j in range(i + 1, N):

        if abs(coupling[i, j]) < 1e-12:
            continue

        label = ["I"] * N

        label[N - 1 - i] = "Z"
        label[N - 1 - j] = "Z"

        pauli_terms.append(
            (
                "".join(label),
                float(coupling[i, j])
            )
        )


cost_hamiltonian = SparsePauliOp.from_list(
    pauli_terms
)


# ============================================================
# QAOA CIRCUIT
# ============================================================

qaoa = QAOAAnsatz(
    cost_operator=cost_hamiltonian,
    reps=REPS
)

circuit = qaoa.decompose()


# ============================================================
# EXPLICIT QAOA PARAMETERS
# ============================================================

parameters = list(circuit.parameters)

if len(parameters) != 2 * REPS:
    raise RuntimeError(
        f"Expected {2 * REPS} QAOA parameters, "
        f"found {len(parameters)}."
    )

# Qiskit QAOA ansatz parameters are ordered as:
#
#   beta parameters first
#   gamma parameters second
#
beta_parameters = parameters[:REPS]
gamma_parameters = parameters[REPS:]


# ============================================================
# STATEVECTOR → QUBO EXPECTED ENERGY
# ============================================================

def state_expected_qubo_energy(state):

    probabilities = state.probabilities()

    total_energy = 0.0

    for index, probability in enumerate(probabilities):

        if probability < 1e-15:
            continue

        # ----------------------------------------------------
        # Qiskit display order → QUBO variable order
        # ----------------------------------------------------

        qiskit_bitstring = format(
            index,
            f"0{N}b"
        )

        qubo_bitstring = qiskit_bitstring[::-1]

        x = np.array(
            [int(bit) for bit in qubo_bitstring],
            dtype=float
        )

        energy = qubo_energy(x)

        total_energy += probability * energy

    return total_energy


# ============================================================
# EXACT CLASSICAL BENCHMARK
# ============================================================

best_classical_energy = float("inf")
best_classical_indices = None


for indices in combinations(range(N), K):

    x = np.zeros(N, dtype=int)

    x[list(indices)] = 1

    energy = qubo_energy(x)

    if energy < best_classical_energy:

        best_classical_energy = energy
        best_classical_indices = indices


# ------------------------------------------------------------
# Classical solution
# ------------------------------------------------------------

classical_x = np.zeros(
    N,
    dtype=int
)

classical_x[list(best_classical_indices)] = 1


# Qiskit display bitstring convention:
# reverse QUBO variable order.

classical_bitstring = "".join(
    str(int(v))
    for v in classical_x[::-1]
)


classical_regions = [
    str(df.iloc[i]["region"])
    for i in best_classical_indices
]


# ============================================================
# GRID SEARCH
# ============================================================

gamma_values = np.linspace(
    0,
    2 * np.pi,
    GRID_POINTS
)

beta_values = np.linspace(
    0,
    np.pi,
    GRID_POINTS
)


best_expected_energy = float("inf")
best_gamma = None
best_beta = None
best_state = None


for gamma in gamma_values:

    for beta in beta_values:

        parameter_values = {}

        for parameter in beta_parameters:
            parameter_values[parameter] = beta

        for parameter in gamma_parameters:
            parameter_values[parameter] = gamma

        parameterized = circuit.assign_parameters(
            parameter_values
        )

        state = Statevector.from_instruction(
            parameterized
        )

        expected_energy = (
            state_expected_qubo_energy(state)
        )

        if expected_energy < best_expected_energy:

            best_expected_energy = expected_energy
            best_gamma = gamma
            best_beta = beta
            best_state = state


# ============================================================
# STATE DISTRIBUTION
# ============================================================

probabilities = best_state.probabilities()

valid_states = []


for index, probability in enumerate(probabilities):

    if probability < 1e-12:
        continue

    # Qiskit display bitstring
    qiskit_bitstring = format(
        index,
        f"0{N}b"
    )

    # Reverse ONCE:
    # Qiskit display → QUBO variable order
    qubo_bitstring = qiskit_bitstring[::-1]

    x = np.array(
        [int(bit) for bit in qubo_bitstring],
        dtype=int
    )

    selected = int(np.sum(x))

    if selected != K:
        continue

    energy = qubo_energy(x)

    indices = tuple(
        np.where(x == 1)[0]
    )

    regions = [
        str(df.iloc[i]["region"])
        for i in indices
    ]

    valid_states.append(
        {
            "qiskit_bitstring": qiskit_bitstring,
            "qubo_bitstring": qubo_bitstring,
            "probability": float(probability),
            "energy": float(energy),
            "selected_count": selected,
            "indices": list(map(int, indices)),
            "regions": regions
        }
    )


valid_states.sort(
    key=lambda item: item["probability"],
    reverse=True
)


# ============================================================
# VALIDATION
# ============================================================

if not valid_states:
    raise RuntimeError(
        "No valid exact-K QAOA states found."
    )


# ============================================================
# CLASSICAL OPTIMUM IN QAOA DISTRIBUTION
# ============================================================

classical_state = None


for state in valid_states:

    if (
        state["qubo_bitstring"]
        == "".join(
            str(int(v))
            for v in classical_x
        )
    ):

        classical_state = state
        break


if classical_state is None:

    classical_probability = 0.0

else:

    classical_probability = (
        classical_state["probability"]
    )


# ============================================================
# MOST PROBABLE VALID QAOA STATE
# ============================================================

most_probable = valid_states[0]


# ============================================================
# MOST PROBABLE QAOA ENERGY GAP
# ============================================================

energy_gap = (
    most_probable["energy"]
    - best_classical_energy
)


relative_gap = (
    abs(energy_gap)
    / abs(best_classical_energy)
    * 100
)


same_solution = (
    most_probable["qubo_bitstring"]
    == "".join(
        str(int(v))
        for v in classical_x
    )
)


# ============================================================
# JACCARD SIMILARITY
# ============================================================

qaoa_set = set(
    most_probable["indices"]
)

classical_set = set(
    best_classical_indices
)

intersection = (
    qaoa_set & classical_set
)

union = (
    qaoa_set | classical_set
)

jaccard = (
    len(intersection)
    / len(union)
)


# ============================================================
# QAOA EXPECTED ENERGY GAP
# ============================================================

expected_energy_gap = (
    best_expected_energy
    - best_classical_energy
)

expected_relative_gap = (
    abs(expected_energy_gap)
    / abs(best_classical_energy)
    * 100
)


# ============================================================
# RESULT
# ============================================================

result = {

    "experiment": {
        "name": "QPDI Final QAOA Benchmark",
        "type": "synthetic_policy_optimization",
        "candidate_count": N,
        "selection_budget": K,
        "classical_combinations": len(
            list(combinations(range(N), K))
        ),
        "qaoa_reps": REPS,
        "grid_points": GRID_POINTS,
        "qaoa_evaluations": (
            GRID_POINTS * GRID_POINTS
        ),
        "bitstring_convention": (
            "Qiskit display bitstring is reversed "
            "once before QUBO variable mapping."
        )
    },

    "classical_exact": {

        "qiskit_bitstring":
            classical_bitstring,

        "qubo_bitstring":
            "".join(
                str(int(v))
                for v in classical_x
            ),

        "energy":
            round(
                best_classical_energy,
                9
            ),

        "indices":
            list(
                map(
                    int,
                    best_classical_indices
                )
            ),

        "regions":
            classical_regions
    },

    "qaoa": {

        "gamma":
            round(
                float(best_gamma),
                9
            ),

        "beta":
            round(
                float(best_beta),
                9
            ),

        "expected_energy":
            round(
                float(best_expected_energy),
                9
            ),

        "expected_energy_gap":
            round(
                float(expected_energy_gap),
                9
            ),

        "expected_relative_gap_percent":
            round(
                float(expected_relative_gap),
                6
            ),

        "most_probable_valid_state":
            most_probable
    },

    "optimum_probability": {

        "classical_optimum_qiskit_bitstring":
            classical_bitstring,

        "probability":
            round(
                classical_probability,
                9
            ),

        "percentage":
            round(
                classical_probability * 100,
                6
            ),

        "present_in_distribution":
            (
                classical_state is not None
            )
    },

    "comparison": {

        "same_solution":
            same_solution,

        "energy_gap":
            round(
                energy_gap,
                9
            ),

        "relative_gap_percent":
            round(
                relative_gap,
                6
            ),

        "jaccard_similarity":
            round(
                jaccard,
                6
            )
    },

    "interpretation": (
        "This experiment compares depth-1 QAOA "
        "simulation against exhaustive classical "
        "optimization for a small synthetic "
        "8-variable policy-selection problem. "
        "The result is a research demonstration "
        "and does not demonstrate quantum advantage."
    )
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
        result,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PRINT
# ============================================================

print()
print("========================================")
print(" QPDI — FINAL QAOA BENCHMARK")
print("========================================")

print()
print("CLASSICAL EXACT")
print("----------------------------------------")

print(
    f"Qiskit bitstring : "
    f"{classical_bitstring}"
)

print(
    f"QUBO bitstring   : "
    f"{''.join(str(int(v)) for v in classical_x)}"
)

print(
    f"Energy           : "
    f"{best_classical_energy:.9f}"
)

print(
    "Regions          : "
    + ", ".join(classical_regions)
)


print()
print("QAOA")
print("----------------------------------------")

print(
    f"Gamma            : "
    f"{best_gamma:.9f}"
)

print(
    f"Beta             : "
    f"{best_beta:.9f}"
)

print(
    f"Expected energy  : "
    f"{best_expected_energy:.9f}"
)

print(
    f"Most probable Qiskit state : "
    f"{most_probable['qiskit_bitstring']}"
)

print(
    f"Most probable QUBO state    : "
    f"{most_probable['qubo_bitstring']}"
)

print(
    f"Probability      : "
    f"{most_probable['probability']:.9f}"
)

print(
    "Regions          : "
    + ", ".join(most_probable["regions"])
)


print()
print("CLASSICAL OPTIMUM IN QAOA DISTRIBUTION")
print("----------------------------------------")

print(
    f"Qiskit bitstring : "
    f"{classical_bitstring}"
)

print(
    f"Probability      : "
    f"{classical_probability:.9f}"
)

print(
    f"Percentage       : "
    f"{classical_probability * 100:.6f}%"
)


print()
print("COMPARISON")
print("----------------------------------------")

print(
    f"Same solution    : "
    f"{same_solution}"
)

print(
    f"Energy gap       : "
    f"{energy_gap:.9f}"
)

print(
    f"Relative gap     : "
    f"{relative_gap:.6f}%"
)

print(
    f"Jaccard          : "
    f"{jaccard:.6f}"
)


print()
print("QAOA EXPECTED ENERGY")
print("----------------------------------------")

print(
    f"Expected gap     : "
    f"{expected_energy_gap:.9f}"
)

print(
    f"Expected gap (%) : "
    f"{expected_relative_gap:.6f}%"
)


print()
print("========================================")
print(" FINAL BENCHMARK SAVED")
print("========================================")

print(OUTPUT_FILE)