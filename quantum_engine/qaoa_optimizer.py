from pathlib import Path

import numpy as np

from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import (
    SparsePauliOp,
    Statevector,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qubo_matrix.npy"
)


# ============================================================
# CONFIGURATION
# ============================================================

QAOA_REPS = 1

GRID_POINTS = 21


# ============================================================
# LOAD QUBO
# ============================================================

if not QUBO_FILE.exists():

    raise FileNotFoundError(
        f"QUBO file not found:\n"
        f"{QUBO_FILE}\n\n"
        "Run qubo_builder.py first."
    )


Q = np.load(
    QUBO_FILE
)


if Q.ndim != 2:

    raise ValueError(
        "QUBO must be a 2D matrix."
    )


if Q.shape[0] != Q.shape[1]:

    raise ValueError(
        "QUBO must be square."
    )


N = Q.shape[0]


# ============================================================
# QUBO → ISING
# ============================================================

h = np.zeros(
    N,
    dtype=float
)

J = {}

constant = 0.0


for i in range(N):

    qii = Q[i, i]

    constant += qii / 2

    h[i] -= qii / 2


for i in range(N):

    for j in range(i + 1, N):

        qij = Q[i, j]

        if abs(qij) < 1e-12:
            continue

        constant += qij / 4

        h[i] -= qij / 4

        h[j] -= qij / 4

        J[(i, j)] = qij / 4


# ============================================================
# BUILD COST HAMILTONIAN
# ============================================================

pauli_terms = []

coefficients = []


for i in range(N):

    if abs(h[i]) < 1e-12:
        continue

    label = ["I"] * N

    label[N - 1 - i] = "Z"

    pauli_terms.append(
        "".join(label)
    )

    coefficients.append(
        h[i]
    )


for (i, j), coupling in J.items():

    if abs(coupling) < 1e-12:
        continue

    label = ["I"] * N

    label[N - 1 - i] = "Z"

    label[N - 1 - j] = "Z"

    pauli_terms.append(
        "".join(label)
    )

    coefficients.append(
        coupling
    )


cost_hamiltonian = SparsePauliOp(
    pauli_terms,
    coeffs=coefficients,
).simplify()


# ============================================================
# QAOA CIRCUIT
# ============================================================

qaoa = QAOAAnsatz(
    cost_operator=cost_hamiltonian,
    reps=QAOA_REPS,
)


qaoa = qaoa.decompose()


parameters = list(
    qaoa.parameters
)


if len(parameters) != 2:

    raise RuntimeError(
        f"Expected 2 QAOA parameters, "
        f"found {len(parameters)}."
    )


# Qiskit normally exposes the two parameters
# as beta/gamma parameters.
#
# We explicitly identify them by name.

parameter_by_name = {
    str(parameter): parameter
    for parameter in parameters
}


beta_parameter = None
gamma_parameter = None


for parameter in parameters:

    name = str(parameter).lower()

    if "beta" in name:

        beta_parameter = parameter

    elif "gamma" in name:

        gamma_parameter = parameter


# Fallback to parameter ordering if names
# are not descriptive.

if beta_parameter is None or gamma_parameter is None:

    beta_parameter = parameters[0]

    gamma_parameter = parameters[1]


# ============================================================
# EXPECTATION VALUE
# ============================================================

def expectation_value(
    gamma,
    beta,
):

    bound_circuit = qaoa.assign_parameters(
        {
            gamma_parameter: gamma,
            beta_parameter: beta,
        }
    )

    state = Statevector.from_instruction(
        bound_circuit
    )

    energy = np.real(
        state.expectation_value(
            cost_hamiltonian
        )
    )

    return float(energy)


# ============================================================
# PARAMETER GRID
# ============================================================

gammas = np.linspace(
    0,
    2 * np.pi,
    GRID_POINTS
)

betas = np.linspace(
    0,
    np.pi,
    GRID_POINTS
)


# ============================================================
# GRID SEARCH
# ============================================================

best_energy = float("inf")

best_gamma = None
best_beta = None


results = []


print()
print("========================================")
print(" QPDI — QAOA PARAMETER OPTIMIZATION")
print("========================================")

print(
    f"QUBO shape       : {Q.shape}"
)

print(
    f"Qubits           : {N}"
)

print(
    f"QAOA repetitions : {QAOA_REPS}"
)

print(
    f"Grid points      : {GRID_POINTS}"
)

print(
    f"Total evaluations: "
    f"{GRID_POINTS ** 2}"
)


print()
print("Searching gamma / beta ...")


for gamma in gammas:

    for beta in betas:

        energy = expectation_value(
            gamma,
            beta,
        )

        results.append(
            (
                energy,
                gamma,
                beta,
            )
        )

        if energy < best_energy:

            best_energy = energy

            best_gamma = gamma

            best_beta = beta


# ============================================================
# SORT RESULTS
# ============================================================

results.sort(
    key=lambda item: item[0]
)


# ============================================================
# BEST RESULT
# ============================================================

print()
print("========================================")
print(" BEST QAOA PARAMETERS")
print("========================================")

print(
    f"Gamma : {best_gamma:.6f}"
)

print(
    f"Beta  : {best_beta:.6f}"
)

print(
    f"Energy: {best_energy:.6f}"
)


# ============================================================
# TOP PARAMETER RESULTS
# ============================================================

print()
print("========================================")
print(" TOP 10 PARAMETER COMBINATIONS")
print("========================================")

print(
    f"{'Rank':<6}"
    f"{'Energy':<16}"
    f"{'Gamma':<16}"
    f"{'Beta':<16}"
)


for rank, (
    energy,
    gamma,
    beta,
) in enumerate(
    results[:10],
    start=1,
):

    print(
        f"{rank:<6}"
        f"{energy:<16.6f}"
        f"{gamma:<16.6f}"
        f"{beta:<16.6f}"
    )


# ============================================================
# FINAL STATE
# ============================================================

best_circuit = qaoa.assign_parameters(
    {
        gamma_parameter: best_gamma,
        beta_parameter: best_beta,
    }
)


best_state = Statevector.from_instruction(
    best_circuit
)


probabilities = best_state.probabilities()


# ============================================================
# TOP MEASUREMENT STATES
# ============================================================

top_indices = np.argsort(
    probabilities
)[::-1][:10]


print()
print("========================================")
print(" TOP QAOA STATES")
print("========================================")

print(
    f"{'State':<15}"
    f"{'Probability':<15}"
)


for index in top_indices:

    bitstring = format(
        index,
        f"0{N}b"
    )

    probability = probabilities[
        index
    ]

    print(
        f"{bitstring:<15}"
        f"{probability:<15.6f}"
    )


print()
print("========================================")
print(" QAOA OPTIMIZATION COMPLETE")
print("========================================")

print(
    "Next:"
)

print(
    "QAOA state"
)

print(
    "    ↓"
)

print(
    "Measurement"
)

print(
    "    ↓"
)

print(
    "Valid bitstring"
)

print(
    "    ↓"
)

print(
    "Selected regions"
)