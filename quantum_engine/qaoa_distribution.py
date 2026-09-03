from pathlib import Path
import json

import numpy as np
import pandas as pd

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector


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

BENCHMARK_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "final_qaoa_benchmark.json"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qaoa_distribution.json"
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

with open(
    BENCHMARK_FILE,
    "r",
    encoding="utf-8"
) as f:
    benchmark = json.load(f)

N = Q.shape[0]


if Q.shape != (len(df), len(df)):
    raise ValueError(
        "QUBO dimensions do not match candidate count."
    )


# ============================================================
# QUBO ENERGY
# ============================================================

def qubo_energy(x):

    x = np.asarray(x, dtype=float)

    return float(
        x @ Q @ x
    )


# ============================================================
# QUBO → ISING
# ============================================================

def qubo_to_ising(Q):

    n = Q.shape[0]

    linear = np.zeros(n)
    coupling = np.zeros((n, n))
    constant = 0.0

    # Diagonal terms
    for i in range(n):

        qii = Q[i, i]

        constant += qii / 2.0
        linear[i] -= qii / 2.0

    # Off-diagonal terms
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


for i in range(N):

    if abs(linear[i]) < 1e-12:
        continue

    label = ["I"] * N

    label[N - 1 - i] = "Z"

    pauli_terms.append(
        (
            "".join(label),
            float(linear[i])
        )
    )


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
# PARAMETERS
# ============================================================

parameters = list(circuit.parameters)

if len(parameters) != 2 * REPS:

    raise RuntimeError(
        f"Expected {2 * REPS} parameters, "
        f"found {len(parameters)}."
    )


beta_parameters = parameters[:REPS]
gamma_parameters = parameters[REPS:]


# ============================================================
# EXPECTED QUBO ENERGY
# ============================================================

def state_expected_qubo_energy(state):

    probabilities = state.probabilities()

    total_energy = 0.0

    for index, probability in enumerate(probabilities):

        if probability < 1e-15:
            continue

        qiskit_bitstring = format(
            index,
            f"0{N}b"
        )

        # Qiskit display order → QUBO order
        qubo_bitstring = (
            qiskit_bitstring[::-1]
        )

        x = np.array(
            [int(bit) for bit in qubo_bitstring],
            dtype=float
        )

        total_energy += (
            probability
            * qubo_energy(x)
        )

    return total_energy


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

        parameterized = (
            circuit.assign_parameters(
                parameter_values
            )
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
# FULL VALID DISTRIBUTION
# ============================================================

probabilities = best_state.probabilities()

valid_states = []


for index, probability in enumerate(probabilities):

    if probability < 1e-15:
        continue

    qiskit_bitstring = format(
        index,
        f"0{N}b"
    )

    qubo_bitstring = (
        qiskit_bitstring[::-1]
    )

    x = np.array(
        [int(bit) for bit in qubo_bitstring],
        dtype=int
    )

    selected_count = int(
        np.sum(x)
    )

    # Keep only exact-K solutions
    if selected_count != K:
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
            "qiskit_bitstring":
                qiskit_bitstring,

            "qubo_bitstring":
                qubo_bitstring,

            "probability":
                float(probability),

            "energy":
                float(energy),

            "selected_count":
                selected_count,

            "indices":
                list(
                    map(
                        int,
                        indices
                    )
                ),

            "regions":
                regions
        }
    )


# ============================================================
# SORT
# ============================================================

valid_states.sort(
    key=lambda item: item["probability"],
    reverse=True
)


# ============================================================
# CLASSICAL OPTIMUM
# ============================================================

classical_qiskit_bitstring = (
    benchmark["classical_exact"]
    ["qiskit_bitstring"]
)

classical_qubo_bitstring = (
    benchmark["classical_exact"]
    ["qubo_bitstring"]
)


# ============================================================
# CLASSICAL OPTIMUM RANK
# ============================================================

classical_rank = None
classical_probability = 0.0


for rank, state in enumerate(
    valid_states,
    start=1
):

    if (
        state["qubo_bitstring"]
        == classical_qubo_bitstring
    ):

        classical_rank = rank

        classical_probability = (
            state["probability"]
        )

        break


# ============================================================
# TOP STATES
# ============================================================

top_1 = valid_states[:1]
top_5 = valid_states[:5]
top_10 = valid_states[:10]


# ============================================================
# CUMULATIVE PROBABILITY
# ============================================================

top_1_probability = sum(
    state["probability"]
    for state in top_1
)

top_5_probability = sum(
    state["probability"]
    for state in top_5
)

top_10_probability = sum(
    state["probability"]
    for state in top_10
)

all_valid_probability = sum(
    state["probability"]
    for state in valid_states
)


# ============================================================
# RESULT
# ============================================================

result = {

    "experiment": {

        "name":
            "QPDI QAOA Full Distribution",

        "candidate_count":
            N,

        "selection_budget":
            K,

        "qaoa_reps":
            REPS,

        "grid_points":
            GRID_POINTS,

        "evaluations":
            GRID_POINTS * GRID_POINTS,

        "valid_exact_k_states":
            len(valid_states),

        "bitstring_convention":
            "Qiskit display bitstring is reversed once "
            "before QUBO variable mapping."
    },

    "qaoa_parameters": {

        "gamma":
            float(best_gamma),

        "beta":
            float(best_beta),

        "expected_qubo_energy":
            float(best_expected_energy)
    },

    "classical_optimum": {

        "qiskit_bitstring":
            classical_qiskit_bitstring,

        "qubo_bitstring":
            classical_qubo_bitstring,

        "energy":
            benchmark["classical_exact"]["energy"],

        "regions":
            benchmark["classical_exact"]["regions"],

        "rank_in_qaoa_valid_distribution":
            classical_rank,

        "probability":
            classical_probability,

        "percentage":
            classical_probability * 100
    },

    "probability_summary": {

        "top_1_probability":
            top_1_probability,

        "top_5_probability":
            top_5_probability,

        "top_10_probability":
            top_10_probability,

        "all_exact_k_probability":
            all_valid_probability
    },

    "top_1":
        top_1,

    "top_5":
        top_5,

    "top_10":
        top_10,

    "all_valid_states":
        valid_states
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
print(" QPDI — FULL QAOA DISTRIBUTION")
print("========================================")

print()
print("QAOA PARAMETERS")
print("----------------------------------------")

print(
    f"Gamma          : "
    f"{best_gamma:.9f}"
)

print(
    f"Beta           : "
    f"{best_beta:.9f}"
)

print(
    f"Expected energy: "
    f"{best_expected_energy:.9f}"
)


print()
print("DISTRIBUTION")
print("----------------------------------------")

print(
    f"Total states           : "
    f"{2 ** N}"
)

print(
    f"Exact-K valid states   : "
    f"{len(valid_states)}"
)

print(
    f"Exact-K probability    : "
    f"{all_valid_probability:.9f}"
)


print()
print("TOP 10 VALID STATES")
print("----------------------------------------")

for rank, state in enumerate(
    valid_states[:10],
    start=1
):

    print(
        f"{rank:2d}. "
        f"{state['qiskit_bitstring']} "
        f"P={state['probability']:.9f} "
        f"E={state['energy']:.9f} "
        f"Regions={','.join(state['regions'])}"
    )


print()
print("CLASSICAL OPTIMUM")
print("----------------------------------------")

print(
    f"Qiskit bitstring : "
    f"{classical_qiskit_bitstring}"
)

print(
    f"QUBO bitstring   : "
    f"{classical_qubo_bitstring}"
)

print(
    f"Rank             : "
    f"{classical_rank}"
)

print(
    f"Probability      : "
    f"{classical_probability:.12f}"
)

print(
    f"Percentage       : "
    f"{classical_probability * 100:.8f}%"
)


print()
print("CUMULATIVE PROBABILITY")
print("----------------------------------------")

print(
    f"Top-1  : "
    f"{top_1_probability:.9f}"
)

print(
    f"Top-5  : "
    f"{top_5_probability:.9f}"
)

print(
    f"Top-10 : "
    f"{top_10_probability:.9f}"
)


print()
print("========================================")
print(" DISTRIBUTION SAVED")
print("========================================")

print(OUTPUT_FILE)