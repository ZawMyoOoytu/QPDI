from pathlib import Path

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp


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


# ------------------------------------------------------------
# Linear terms
# ------------------------------------------------------------

for i in range(N):

    qii = Q[i, i]

    constant += qii / 2

    h[i] -= qii / 2


# ------------------------------------------------------------
# Quadratic terms
# ------------------------------------------------------------

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
# BUILD ISING HAMILTONIAN
# ============================================================

pauli_terms = []

coefficients = []


# ------------------------------------------------------------
# Z terms
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# ZZ terms
# ------------------------------------------------------------

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
# QAOA ANSATZ
# ============================================================

QAOA_REPS = 1

qaoa = QAOAAnsatz(
    cost_operator=cost_hamiltonian,
    reps=QAOA_REPS,
)


qaoa_circuit = qaoa.decompose()


# ============================================================
# ADD MEASUREMENT
# ============================================================

measured_circuit = QuantumCircuit(
    N,
    N
)

measured_circuit.compose(
    qaoa_circuit,
    inplace=True
)

measured_circuit.measure(
    range(N),
    range(N)
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("========================================")
print(" QPDI — QAOA CIRCUIT")
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
    f"Z terms          : {len(h)}"
)

print(
    f"ZZ interactions  : {len(J)}"
)

print(
    f"Constant offset  : {constant:.6f}"
)


print()
print("========================================")
print(" CIRCUIT INFORMATION")
print("========================================")

print(
    f"Depth            : "
    f"{qaoa_circuit.depth()}"
)

print(
    f"Gate count       : "
    f"{qaoa_circuit.size()}"
)

print(
    f"Parameters       : "
    f"{qaoa_circuit.num_parameters}"
)


print()
print("========================================")
print(" QAOA PARAMETERS")
print("========================================")

for parameter in qaoa_circuit.parameters:

    print(
        f"Parameter: {parameter}"
    )


print()
print("========================================")
print(" QAOA CIRCUIT")
print("========================================")

print(
    qaoa_circuit.draw(
        output="text",
        fold=120
    )
)


print()
print("========================================")
print(" QAOA CIRCUIT READY")
print("========================================")

print(
    "Next:"
)

print(
    "QAOA parameter optimization"
)

print(
    "      ↓"
)

print(
    "Quantum state preparation"
)

print(
    "      ↓"
)

print(
    "Measurement"
)

print(
    "      ↓"
)

print(
    "Candidate bitstrings"
)