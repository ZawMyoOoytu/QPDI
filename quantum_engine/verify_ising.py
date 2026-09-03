from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qubo_matrix.npy"
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
# ENERGY FUNCTIONS
# ============================================================

def qubo_energy(x):

    return float(
        x @ Q @ x
    )


def ising_energy(x, linear, coupling, constant):

    # x_i = (1 - z_i) / 2
    z = 1 - 2 * x

    energy = constant

    for i in range(N):

        energy += (
            linear[i] * z[i]
        )

    for i in range(N):

        for j in range(i + 1, N):

            energy += (
                coupling[i, j]
                * z[i]
                * z[j]
            )

    return float(energy)


# ============================================================
# BUILD ISING
# ============================================================

linear, coupling, constant = qubo_to_ising(Q)


# ============================================================
# TEST STATES
# ============================================================

test_states = [

    np.array(
        [0, 0, 0, 0, 0, 0, 0, 0],
        dtype=float
    ),

    np.array(
        [1, 0, 0, 0, 0, 0, 0, 0],
        dtype=float
    ),

    np.array(
        [1, 1, 1, 1, 1, 0, 0, 0],
        dtype=float
    ),

    np.array(
        [0, 0, 0, 1, 1, 1, 1, 1],
        dtype=float
    ),

    np.array(
        [1, 1, 0, 1, 0, 1, 0, 1],
        dtype=float
    ),

    np.array(
        [1, 1, 1, 1, 1, 1, 1, 1],
        dtype=float
    ),
]


# ============================================================
# VERIFICATION
# ============================================================

print()
print("========================================")
print(" QPDI — QUBO / ISING VERIFICATION")
print("========================================")

print()
print(f"QUBO size       : {N} x {N}")
print(f"Ising constant  : {constant:.12f}")
print()

max_error = 0.0


for x in test_states:

    q_energy = qubo_energy(x)

    i_energy = ising_energy(
        x,
        linear,
        coupling,
        constant
    )

    error = abs(
        q_energy - i_energy
    )

    max_error = max(
        max_error,
        error
    )

    bitstring = "".join(
        str(int(v))
        for v in x
    )

    print("----------------------------------------")
    print(f"x              : {bitstring}")
    print(f"QUBO energy    : {q_energy:.12f}")
    print(f"Ising energy   : {i_energy:.12f}")
    print(f"Absolute error : {error:.12e}")


print()
print("========================================")
print(" VERIFICATION RESULT")
print("========================================")

print(
    f"Maximum error : {max_error:.12e}"
)

if max_error < 1e-9:

    print(
        "STATUS         : PASS"
    )

    print(
        "QUBO and Ising energies are "
        "mathematically consistent."
    )

else:

    print(
        "STATUS         : FAIL"
    )

    print(
        "QUBO → Ising conversion is inconsistent."
    )