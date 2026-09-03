from pathlib import Path
from itertools import combinations

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


# ============================================================
# CONFIGURATION
# ============================================================

BUDGET_UNITS = 5

QAOA_REPS = 1

GRID_POINTS = 21


# ============================================================
# LOAD DATA
# ============================================================

if not QUBO_FILE.exists():

    raise FileNotFoundError(
        f"QUBO file not found:\n"
        f"{QUBO_FILE}\n\n"
        "Run qubo_builder.py first."
    )


if not CANDIDATE_FILE.exists():

    raise FileNotFoundError(
        f"Candidate file not found:\n"
        f"{CANDIDATE_FILE}\n\n"
        "Run candidate_selector.py first."
    )


Q = np.load(
    QUBO_FILE
)

df = pd.read_csv(
    CANDIDATE_FILE
)


N = Q.shape[0]


if Q.shape != (len(df), len(df)):

    raise ValueError(
        "QUBO size does not match "
        "candidate count."
    )


if BUDGET_UNITS > N:

    raise ValueError(
        "Budget cannot exceed "
        "candidate count."
    )


# ============================================================
# QUBO → ISING
# ============================================================

h = np.zeros(
    N,
    dtype=float
)

J = {}


for i in range(N):

    qii = Q[i, i]

    h[i] -= qii / 2


for i in range(N):

    for j in range(i + 1, N):

        qij = Q[i, j]

        if abs(qij) < 1e-12:
            continue

        h[i] -= qij / 4

        h[j] -= qij / 4

        J[(i, j)] = qij / 4


# ============================================================
# BUILD ISING HAMILTONIAN
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
).decompose()


parameters = list(
    qaoa.parameters
)


if len(parameters) != 2:

    raise RuntimeError(
        f"Expected 2 QAOA parameters, "
        f"found {len(parameters)}."
    )


# Qiskit parameter names can vary slightly,
# so identify them by name when possible.

beta_parameter = None
gamma_parameter = None


for parameter in parameters:

    name = str(parameter).lower()

    if "beta" in name:
        beta_parameter = parameter

    elif "gamma" in name:
        gamma_parameter = parameter


if beta_parameter is None:

    beta_parameter = parameters[0]


if gamma_parameter is None:

    gamma_parameter = parameters[1]


# ============================================================
# EXPECTATION VALUE
# ============================================================

def get_energy(
    gamma,
    beta,
):

    circuit = qaoa.assign_parameters(
        {
            gamma_parameter: gamma,
            beta_parameter: beta,
        }
    )

    state = Statevector.from_instruction(
        circuit
    )

    energy = np.real(
        state.expectation_value(
            cost_hamiltonian
        )
    )

    return float(energy)


# ============================================================
# QAOA PARAMETER SEARCH
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


best_energy = float("inf")

best_gamma = None
best_beta = None


print()
print("========================================")
print(" QPDI — QAOA POLICY DECODER")
print("========================================")

print(
    f"Candidates : {N}"
)

print(
    f"Budget K   : {BUDGET_UNITS}"
)

print(
    f"Qubits     : {N}"
)

print(
    f"Grid       : "
    f"{GRID_POINTS} × {GRID_POINTS}"
)

print()
print("Optimizing QAOA parameters...")


for gamma in gammas:

    for beta in betas:

        energy = get_energy(
            gamma,
            beta,
        )

        if energy < best_energy:

            best_energy = energy

            best_gamma = gamma

            best_beta = beta


# ============================================================
# FINAL QAOA STATE
# ============================================================

best_circuit = qaoa.assign_parameters(
    {
        gamma_parameter: best_gamma,
        beta_parameter: best_beta,
    }
)


state = Statevector.from_instruction(
    best_circuit
)


probabilities = state.probabilities()


# ============================================================
# VALIDATE BITSTRING
# ============================================================

def decode_bitstring(
    bitstring
):

    # Qiskit displays classical bitstrings
    # in reverse qubit-order relative to
    # our qubit indexing.

    qubit_bits = bitstring[::-1]

    selected_indices = [
        i
        for i, bit in enumerate(
            qubit_bits
        )
        if bit == "1"
    ]

    return selected_indices


# ============================================================
# GET TOP STATES
# ============================================================

sorted_indices = np.argsort(
    probabilities
)[::-1]


valid_states = []


for index in sorted_indices:

    bitstring = format(
        index,
        f"0{N}b"
    )

    selected_indices = decode_bitstring(
        bitstring
    )

    selected_count = len(
        selected_indices
    )

    if selected_count == BUDGET_UNITS:

        valid_states.append(
            (
                bitstring,
                probabilities[index],
                selected_indices,
            )
        )

    if len(valid_states) >= 10:

        break


# ============================================================
# DISPLAY QAOA RESULT
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


print()
print("========================================")
print(" TOP VALID QAOA STATES")
print("========================================")


if not valid_states:

    print(
        "WARNING: No state with exact "
        f"K={BUDGET_UNITS} found among "
        "the highest-probability states."
    )

else:

    print(
        f"{'Bitstring':<15}"
        f"{'Probability':<15}"
        f"{'Selected':<10}"
    )

    for (
        bitstring,
        probability,
        selected_indices,
    ) in valid_states:

        print(
            f"{bitstring:<15}"
            f"{probability:<15.6f}"
            f"{len(selected_indices):<10}"
        )


# ============================================================
# SELECT BEST VALID STATE
# ============================================================

if valid_states:

    best_bitstring = valid_states[0][0]

    best_probability = valid_states[0][1]

    selected_indices = valid_states[0][2]

else:

    raise RuntimeError(
        "QAOA did not produce a valid "
        "exact-K state in the inspected states."
    )


# ============================================================
# MAP TO ACTUAL REGIONS
# ============================================================

selected_regions = [
    df.iloc[i]["region"]
    for i in selected_indices
]


# ============================================================
# DISPLAY SELECTED REGIONS
# ============================================================

print()
print("========================================")
print(" QAOA SELECTED REGIONS")
print("========================================")

print(
    f"Bitstring   : {best_bitstring}"
)

print(
    f"Probability : {best_probability:.6f}"
)

print(
    f"Selected    : {len(selected_regions)}"
)

print(
    f"Required K  : {BUDGET_UNITS}"
)


print()

for index in selected_indices:

    row = df.iloc[index]

    print(
        f"Qubit {index} → "
        f"{row['region']} | "
        f"AI rank={int(row['ai_rank'])} | "
        f"objective="
        f"{row['optimization_score']:.4f}"
    )


# ============================================================
# EXACT-K VALIDATION
# ============================================================

constraint_valid = (
    len(selected_regions)
    == BUDGET_UNITS
)


print()
print("========================================")
print(" CONSTRAINT VALIDATION")
print("========================================")

print(
    f"Selected count : "
    f"{len(selected_regions)}"
)

print(
    f"Required count : "
    f"{BUDGET_UNITS}"
)

print(
    f"Exact-K valid  : "
    f"{constraint_valid}"
)


if not constraint_valid:

    raise RuntimeError(
        "QAOA solution violates "
        "the exact-K constraint."
    )


print()
print("========================================")
print(" POLICY DECODING COMPLETE")
print("========================================")

print(
    "QAOA bitstring"
)

print(
    "      ↓"
)

print(
    "Qubit indices"
)

print(
    "      ↓"
)

print(
    "Actual region names"
)

print(
    "      ↓"
)

print(
    f"Exactly {BUDGET_UNITS} selected regions"
)

print(
    "      ↓"
)

print(
    "Classical benchmark"
)