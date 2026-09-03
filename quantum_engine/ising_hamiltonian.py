from pathlib import Path

import numpy as np
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


# ============================================================
# VALIDATION
# ============================================================

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
#
# QUBO:
#
#     E(x) = Σ Qii xi
#          + Σ(i<j) Qij xi xj
#
# with:
#
#     xi = (1 - zi) / 2
#
# Therefore:
#
#     xi       → (1 - zi) / 2
#
#     xi*xj    → (1 - zi - zj + zi*zj) / 4
#
# The constant energy offset does not affect
# the optimal bitstring.
# ============================================================

h = np.zeros(
    N,
    dtype=float
)

J = {}

constant = 0.0


# ============================================================
# LINEAR TERMS
# ============================================================

for i in range(N):

    qii = Q[i, i]

    constant += qii / 2

    h[i] -= qii / 2


# ============================================================
# QUADRATIC TERMS
# ============================================================

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
# BUILD PAULI TERMS
#
# H = Σ hi Zi + Σ Jij Zi Zj
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


# ============================================================
# CREATE SPARSE PAULI OP
# ============================================================

cost_hamiltonian = SparsePauliOp(
    pauli_terms,
    coeffs=coefficients,
)


cost_hamiltonian = (
    cost_hamiltonian.simplify()
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("========================================")
print(" QPDI — QUBO → ISING")
print("========================================")

print(
    f"QUBO file : {QUBO_FILE}"
)

print(
    f"QUBO shape: {Q.shape}"
)

print(
    f"Qubits    : {N}"
)

print()
print(
    f"Linear Z terms : {len(h)}"
)

print(
    f"ZZ couplings   : {len(J)}"
)

print(
    f"Pauli terms    : "
    f"{len(cost_hamiltonian)}"
)

print(
    f"Constant offset: {constant:.6f}"
)


print()
print("========================================")
print(" ISING HAMILTONIAN")
print("========================================")

print(
    cost_hamiltonian
)


print()
print("========================================")
print(" CONVERSION COMPLETE")
print("========================================")

print("QUBO")
print("  ↓")
print("Binary variables xᵢ")
print("  ↓")
print("xᵢ = (1 - zᵢ) / 2")
print("  ↓")
print("Ising variables zᵢ ∈ {-1,+1}")
print("  ↓")
print("SparsePauliOp")
print("  ↓")
print("QAOA Cost Hamiltonian")