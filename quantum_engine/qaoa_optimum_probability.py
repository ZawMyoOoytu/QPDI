from pathlib import Path

import numpy as np

from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import QAOAAnsatz


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

REPS = 1
GRID_POINTS = 21
BUDGET_K = 5

# Exact classical optimum from Step 28
CLASSICAL_OPTIMUM = "00011111"


# ============================================================
# LOAD QUBO
# ============================================================

if not QUBO_FILE.exists():
    raise FileNotFoundError(
        f"QUBO file not found:\n{QUBO_FILE}"
    )

Q = np.load(QUBO_FILE)

N = Q.shape[0]


# ============================================================
# QUBO → ISING
# ============================================================

def qubo_to_ising(Q):

    n = Q.shape[0]

    linear = np.zeros(n)
    coupling = np.zeros((n, n))

    constant = 0.0

    # x_i = (1 - z_i) / 2
    #
    # x^T Q x
    #
    # Diagonal:
    # Q_ii x_i
    #
    # Off-diagonal:
    # 2 Q_ij x_i x_j
    # for i < j

    for i in range(n):

        qii = Q[i, i]

        constant += qii / 2.0
        linear[i] -= qii / 2.0

    for i in range(n):

        for j in range(i + 1, n):

            qij = Q[i, j]

            if abs(qij) < 1e-12:
                continue

            # 2 Qij xi xj
            #
            # xi xj =
            # (1 - zi - zj + zi zj) / 4
            #
            # therefore:
            #
            # Qij / 2 *
            # (1 - zi - zj + zi zj)

            constant += qij / 2.0

            linear[i] -= qij / 2.0
            linear[j] -= qij / 2.0

            coupling[i, j] += qij / 2.0

    return (
        linear,
        coupling,
        constant,
    )


# ============================================================
# BUILD ISING HAMILTONIAN
# ============================================================

linear, coupling, constant = (
    qubo_to_ising(Q)
)

pauli_terms = []


for i in range(N):

    if abs(linear[i]) < 1e-12:
        continue

    label = ["I"] * N

    label[N - 1 - i] = "Z"

    pauli_terms.append(
        (
            "".join(label),
            float(linear[i]),
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
                float(coupling[i, j]),
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
    reps=REPS,
)

qaoa_circuit = qaoa.decompose()


# ============================================================
# GRID SEARCH
# ============================================================

best_energy = float("inf")
best_gamma = None
best_beta = None
best_state = None


gamma_values = np.linspace(
    0,
    2 * np.pi,
    GRID_POINTS,
)

beta_values = np.linspace(
    0,
    np.pi,
    GRID_POINTS,
)


for gamma in gamma_values:

    for beta in beta_values:

        parameter_values = {
            qaoa_circuit.parameters[0]: gamma,
            qaoa_circuit.parameters[1]: beta,
        }

        circuit = (
            qaoa_circuit.assign_parameters(
                parameter_values
            )
        )

        state = Statevector.from_instruction(
            circuit
        )

        probabilities = state.probabilities()

        energy = 0.0

        for index, probability in enumerate(
            probabilities
        ):

            if probability < 1e-15:
                continue

            # Qiskit basis state:
            # index → binary bitstring
            #
            # Reverse to obtain
            # QUBO variable order.

            bitstring = format(
                index,
                f"0{N}b"
            )[::-1]

            x = np.array(
                [
                    int(bit)
                    for bit in bitstring
                ],
                dtype=float,
            )

            energy += (
                probability
                * float(x @ Q @ x)
            )

        if energy < best_energy:

            best_energy = energy
            best_gamma = gamma
            best_beta = beta
            best_state = state


# ============================================================
# PROBABILITY OF CLASSICAL OPTIMUM
# ============================================================

classical_index = int(
    CLASSICAL_OPTIMUM[::-1],
    2,
)

classical_probability = (
    best_state.probabilities()[
        classical_index
    ]
)


# ============================================================
# QAOA MOST PROBABLE STATE
# ============================================================

probabilities = (
    best_state.probabilities()
)

most_probable_index = int(
    np.argmax(probabilities)
)

most_probable_bitstring = format(
    most_probable_index,
    f"0{N}b"
)

most_probable_qubo_bitstring = (
    most_probable_bitstring[::-1]
)

most_probable_probability = (
    probabilities[
        most_probable_index
    ]
)


# ============================================================
# CLASSICAL OPTIMUM ENERGY
# ============================================================

classical_x = np.array(
    [
        int(bit)
        for bit in CLASSICAL_OPTIMUM
    ],
    dtype=float,
)

classical_energy = float(
    classical_x @ Q @ classical_x
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("========================================")
print(" QPDI — QAOA OPTIMUM PROBABILITY")
print("========================================")

print(
    f"Qubits        : {N}"
)

print(
    f"QAOA reps     : {REPS}"
)

print(
    f"Grid points   : {GRID_POINTS}"
)

print(
    f"Evaluations   : "
    f"{GRID_POINTS ** 2}"
)


print()
print("========================================")
print(" BEST QAOA PARAMETERS")
print("========================================")

print(
    f"Gamma         : "
    f"{best_gamma:.6f}"
)

print(
    f"Beta          : "
    f"{best_beta:.6f}"
)

print(
    f"Expected QUBO : "
    f"{best_energy:.6f}"
)


print()
print("========================================")
print(" MOST PROBABLE QAOA STATE")
print("========================================")

print(
    f"Bitstring     : "
    f"{most_probable_qubo_bitstring}"
)

print(
    f"Probability   : "
    f"{most_probable_probability:.6f}"
)


print()
print("========================================")
print(" CLASSICAL EXACT OPTIMUM")
print("========================================")

print(
    f"Bitstring     : "
    f"{CLASSICAL_OPTIMUM}"
)

print(
    f"Energy        : "
    f"{classical_energy:.6f}"
)

print(
    f"QAOA Probability: "
    f"{classical_probability:.6f}"
)

print(
    f"QAOA Probability (%): "
    f"{classical_probability * 100:.4f}%"
)


# ============================================================
# INTERPRETATION
# ============================================================

print()
print("========================================")
print(" INTERPRETATION")
print("========================================")

if classical_probability > 0:

    print(
        "✓ The exact classical optimum "
        "appears in the QAOA output distribution."
    )

else:

    print(
        "✗ The exact classical optimum "
        "does not appear in the QAOA output distribution."
    )


if (
    most_probable_qubo_bitstring
    == CLASSICAL_OPTIMUM
):

    print(
        "✓ The most probable QAOA state "
        "is the classical optimum."
    )

else:

    print(
        "QAOA's most probable state is "
        "different from the classical optimum."
    )


print()
print(
    "Note: probability is not solution "
    "quality and does not demonstrate "
    "quantum advantage."
)