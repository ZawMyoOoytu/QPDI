from pathlib import Path
from itertools import combinations
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

OUTPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qaoa_parameter_scan.json"
)


# ============================================================
# CONFIG
# ============================================================

REPS = 1

# endpoint=False prevents duplicate gamma=0 and gamma=2*pi
GRID_POINTS = 41

K = 5


# ============================================================
# LOAD INPUTS
# ============================================================

if not QUBO_FILE.exists():
    raise FileNotFoundError(
        f"QUBO file not found:\n{QUBO_FILE}\n\n"
        "Run qubo_builder.py first."
    )

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate file not found:\n{CANDIDATE_FILE}\n\n"
        "Run candidate_selector.py first."
    )


Q = np.load(QUBO_FILE)

df = pd.read_csv(
    CANDIDATE_FILE
)

N = int(Q.shape[0])


# ============================================================
# VALIDATE INPUTS
# ============================================================

if Q.ndim != 2:
    raise ValueError(
        "QUBO matrix must be 2-dimensional."
    )

if Q.shape[0] != Q.shape[1]:
    raise ValueError(
        "QUBO matrix must be square."
    )

if len(df) != N:
    raise ValueError(
        "QUBO size does not match candidate count."
    )

if K > N:
    raise ValueError(
        "Selection budget K cannot exceed "
        "candidate count."
    )


# ============================================================
# QUBO ENERGY
# ============================================================

def qubo_energy(x):

    x = np.asarray(
        x,
        dtype=float
    )

    return float(
        x @ Q @ x
    )


# ============================================================
# QUBO → ISING
# ============================================================

def qubo_to_ising(Q):

    n = Q.shape[0]

    linear = np.zeros(
        n,
        dtype=float
    )

    coupling = np.zeros(
        (n, n),
        dtype=float
    )

    # --------------------------------------------------------
    # Diagonal QUBO terms
    #
    # qii * xi
    #
    # xi = (1 - Zi) / 2
    # --------------------------------------------------------

    for i in range(n):

        qii = float(
            Q[i, i]
        )

        linear[i] -= (
            qii / 2.0
        )

    # --------------------------------------------------------
    # Off-diagonal QUBO terms
    #
    # qij * xi * xj
    #
    # xi*xj =
    # 1/4 * (1 - Zi - Zj + ZiZj)
    #
    # Therefore:
    #
    # ZZ coefficient = qij / 4
    # --------------------------------------------------------

    for i in range(n):

        for j in range(i + 1, n):

            qij = float(
                Q[i, j]
            )

            if abs(qij) < 1e-12:
                continue

            coefficient = (
                qij / 4.0
            )

            linear[i] -= (
                coefficient
            )

            linear[j] -= (
                coefficient
            )

            coupling[i, j] += (
                coefficient
            )

    return linear, coupling


# ============================================================
# BUILD ISING HAMILTONIAN
# ============================================================

linear, coupling = (
    qubo_to_ising(Q)
)

pauli_terms = []


# ------------------------------------------------------------
# Z TERMS
# ------------------------------------------------------------

for i in range(N):

    if abs(linear[i]) < 1e-12:
        continue

    label = ["I"] * N

    # Qiskit Pauli label ordering
    label[N - 1 - i] = "Z"

    pauli_terms.append(
        (
            "".join(label),
            float(linear[i])
        )
    )


# ------------------------------------------------------------
# ZZ TERMS
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


cost_hamiltonian = (
    SparsePauliOp.from_list(
        pauli_terms
    )
)


# ============================================================
# BUILD QAOA CIRCUIT
# ============================================================

qaoa = QAOAAnsatz(
    cost_operator=cost_hamiltonian,
    reps=REPS
)

circuit = (
    qaoa.decompose()
)


# ============================================================
# PARAMETERS
# ============================================================

parameters = list(
    circuit.parameters
)

expected_parameter_count = (
    2 * REPS
)

if len(parameters) != expected_parameter_count:

    raise RuntimeError(
        f"Expected "
        f"{expected_parameter_count} "
        f"QAOA parameters, but found "
        f"{len(parameters)}."
    )


# Qiskit QAOAAnsatz ordering
beta_parameters = (
    parameters[:REPS]
)

gamma_parameters = (
    parameters[REPS:]
)


# ============================================================
# EXPECTED QUBO ENERGY
# ============================================================

def expected_energy(state):

    probabilities = (
        state.probabilities()
    )

    total = 0.0

    for index, probability in enumerate(
        probabilities
    ):

        probability = float(
            probability
        )

        if probability < 1e-15:
            continue

        # Qiskit computational-basis index
        qiskit_bitstring = format(
            index,
            f"0{N}b"
        )

        # IMPORTANT:
        #
        # Qiskit displayed bitstring
        # → reverse once
        # → QUBO variable order
        #
        qubo_bitstring = (
            qiskit_bitstring[::-1]
        )

        x = np.array(
            [
                int(bit)
                for bit in qubo_bitstring
            ],
            dtype=float
        )

        total += (
            probability
            * qubo_energy(x)
        )

    return float(total)


# ============================================================
# PARAMETER GRID
# ============================================================

gamma_values = np.linspace(
    0.0,
    2.0 * np.pi,
    GRID_POINTS,
    endpoint=False
)

beta_values = np.linspace(
    0.0,
    np.pi,
    GRID_POINTS
)


# ============================================================
# SCAN
# ============================================================

results = []

best_energy = float(
    "inf"
)

best_gamma = None
best_beta = None


for gamma in gamma_values:

    for beta in beta_values:

        parameter_values = {}

        # beta parameters
        for parameter in beta_parameters:

            parameter_values[
                parameter
            ] = float(beta)

        # gamma parameters
        for parameter in gamma_parameters:

            parameter_values[
                parameter
            ] = float(gamma)

        bound_circuit = (
            circuit.assign_parameters(
                parameter_values
            )
        )

        state = (
            Statevector.from_instruction(
                bound_circuit
            )
        )

        energy = expected_energy(
            state
        )

        result = {
            "gamma": float(gamma),

            "beta": float(beta),

            "expected_energy":
                float(energy)
        }

        results.append(
            result
        )

        if energy < best_energy:

            best_energy = float(
                energy
            )

            best_gamma = float(
                gamma
            )

            best_beta = float(
                beta
            )


# ============================================================
# SAFETY CHECK
# ============================================================

if best_gamma is None or best_beta is None:

    raise RuntimeError(
        "Parameter scan failed to find "
        "a valid QAOA parameter point."
    )


# ============================================================
# SORT RESULTS
# ============================================================

results_sorted = sorted(
    results,
    key=lambda item: item[
        "expected_energy"
    ]
)


# ============================================================
# CLASSICAL EXACT OPTIMUM
# ============================================================

valid_energies = []


for indices in combinations(
    range(N),
    K
):

    x = np.zeros(
        N,
        dtype=int
    )

    for i in indices:

        x[i] = 1

    energy = qubo_energy(
        x
    )

    valid_energies.append(
        float(energy)
    )


classical_optimum = float(
    min(valid_energies)
)


# ============================================================
# ENERGY GAP
# ============================================================

energy_gap = float(
    best_energy
    - classical_optimum
)


if abs(classical_optimum) > 1e-12:

    relative_gap = float(
        energy_gap
        / abs(classical_optimum)
    )

else:

    relative_gap = None


# ============================================================
# BOUNDARY ANALYSIS
# ============================================================

gamma_step = float(
    2.0 * np.pi
    / GRID_POINTS
)

beta_step = float(
    np.pi
    / (GRID_POINTS - 1)
)


# ------------------------------------------------------------
# Because gamma endpoint is excluded, there is no actual
# gamma = 2*pi point in the scan.
#
# We therefore check whether the best point is close to
# gamma = 0 or close to the upper periodic boundary.
# ------------------------------------------------------------

gamma_is_near_zero = bool(
    abs(best_gamma)
    <= gamma_step
)


gamma_distance_to_two_pi = abs(
    best_gamma
    - 2.0 * np.pi
)


gamma_is_near_two_pi = bool(
    gamma_distance_to_two_pi
    <= gamma_step
)


beta_is_boundary = bool(
    (
        abs(best_beta)
        <= beta_step
    )
    or
    (
        abs(
            best_beta - np.pi
        )
        <= beta_step
    )
)


boundary_warning = bool(
    gamma_is_near_zero
    or
    gamma_is_near_two_pi
    or
    beta_is_boundary
)


# ============================================================
# BEST POINTS
# ============================================================

top_10_parameter_points = (
    results_sorted[:10]
)


# ============================================================
# JSON OUTPUT
# ============================================================

output = {

    "experiment": {

        "name":
            "QPDI QAOA Parameter Sensitivity Scan",

        "candidate_count":
            int(N),

        "selection_budget":
            int(K),

        "qaoa_reps":
            int(REPS),

        "grid_points":
            int(GRID_POINTS),

        "evaluations":
            int(len(results)),

        "gamma_endpoint_excluded":
            True,

        "gamma_range":
            [
                0.0,
                float(2.0 * np.pi)
            ],

        "beta_range":
            [
                0.0,
                float(np.pi)
            ]
    },


    "best_parameters": {

        "gamma":
            float(best_gamma),

        "beta":
            float(best_beta),

        "expected_energy":
            float(best_energy)
    },


    "classical_reference": {

        "optimal_energy":
            float(classical_optimum)
    },


    "quality": {

        "energy_gap":
            float(energy_gap),

        "relative_gap":
            (
                float(relative_gap)
                if relative_gap is not None
                else None
            ),

        "relative_gap_percent":
            (
                float(
                    relative_gap * 100.0
                )
                if relative_gap is not None
                else None
            )
    },


    "boundary_analysis": {

        "gamma_near_zero":
            bool(gamma_is_near_zero),

        "gamma_near_two_pi":
            bool(gamma_is_near_two_pi),

        "beta_at_boundary":
            bool(beta_is_boundary),

        "boundary_warning":
            bool(boundary_warning)
    },


    "top_10_parameter_points":
        top_10_parameter_points
}


# ============================================================
# SAVE JSON
# ============================================================

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# PRINT
# ============================================================

print()

print(
    "========================================"
)

print(
    " QPDI — QAOA PARAMETER SENSITIVITY"
)

print(
    "========================================"
)


print()
print("SCAN")
print("----------------------------------------")

print(
    f"Grid points       : "
    f"{GRID_POINTS}"
)

print(
    f"Evaluations       : "
    f"{len(results)}"
)

print(
    "Gamma endpoint    : excluded"
)


print()
print("BEST PARAMETERS")
print("----------------------------------------")

print(
    f"Gamma             : "
    f"{best_gamma:.9f}"
)

print(
    f"Beta              : "
    f"{best_beta:.9f}"
)

print(
    f"Expected energy   : "
    f"{best_energy:.9f}"
)


print()
print("CLASSICAL REFERENCE")
print("----------------------------------------")

print(
    f"Optimal energy    : "
    f"{classical_optimum:.9f}"
)


print()
print("QUALITY")
print("----------------------------------------")

print(
    f"Energy gap        : "
    f"{energy_gap:.9f}"
)

if relative_gap is not None:

    print(
        f"Relative gap      : "
        f"{relative_gap * 100:.6f}%"
    )

else:

    print(
        "Relative gap      : N/A"
    )


print()
print("BOUNDARY ANALYSIS")
print("----------------------------------------")

print(
    f"Gamma near 0      : "
    f"{gamma_is_near_zero}"
)

print(
    f"Gamma near 2π     : "
    f"{gamma_is_near_two_pi}"
)

print(
    f"Beta at boundary  : "
    f"{beta_is_boundary}"
)

print(
    f"Boundary warning  : "
    f"{boundary_warning}"
)


print()
print("TOP 10 PARAMETER POINTS")
print("----------------------------------------")

for rank, item in enumerate(
    results_sorted[:10],
    start=1
):

    print(
        f"{rank:2d}. "
        f"gamma={item['gamma']:.6f} "
        f"beta={item['beta']:.6f} "
        f"energy="
        f"{item['expected_energy']:.9f}"
    )


print()
print(
    "========================================"
)

print(
    " SCAN COMPLETE"
)

print(
    "========================================"
)

print(
    f"Saved to:\n{OUTPUT_FILE}"
)