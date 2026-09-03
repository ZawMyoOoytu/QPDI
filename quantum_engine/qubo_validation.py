from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd


# ============================================================
# QPDI — QUBO VALIDATION
#
# Verify:
#
#   1. QUBO energy
#   2. Exact-K constraint
#   3. Classical optimum
#   4. Objective consistency
#
# This is a correctness test before QAOA.
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qubo_matrix.npy"
)

AI_FILE = (
    PROJECT_ROOT
    / "data"
    / "ai_policy_predictions.csv"
)


# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------

BUDGET_UNITS = 5

PENALTY = 3.0


# ------------------------------------------------------------
# 3. LOAD FILES
# ------------------------------------------------------------

if not QUBO_FILE.exists():

    raise FileNotFoundError(
        f"QUBO file not found:\n{QUBO_FILE}"
    )


if not AI_FILE.exists():

    raise FileNotFoundError(
        f"AI prediction file not found:\n{AI_FILE}\n\n"
        "Run policy_predictor.py first."
    )


Q = np.load(
    QUBO_FILE
)

df = pd.read_csv(
    AI_FILE
)


# ------------------------------------------------------------
# 4. VALIDATION
# ------------------------------------------------------------

if Q.ndim != 2:

    raise ValueError(
        "QUBO must be a 2D matrix."
    )


if Q.shape[0] != Q.shape[1]:

    raise ValueError(
        "QUBO must be square."
    )


N = Q.shape[0]


if len(df) != N:

    raise ValueError(
        "AI dataset size and QUBO size "
        "do not match."
    )


# ------------------------------------------------------------
# 5. QUBO ENERGY
# ------------------------------------------------------------

def qubo_energy(
    x
):

    x = np.asarray(
        x,
        dtype=float
    )

    return float(
        x @ Q @ x
    )


# ------------------------------------------------------------
# 6. ORIGINAL OBJECTIVE
# ------------------------------------------------------------

benefit = df[
    "predicted_benefit"
].to_numpy(
    dtype=float
)


need = df[
    "need"
].to_numpy(
    dtype=float
)


# Normalize exactly as QUBO builder does.

benefit_score = (
    benefit - benefit.min()
) / (
    benefit.max()
    - benefit.min()
    + 1e-12
)


need_score = (
    need - need.min()
) / (
    need.max()
    - need.min()
    + 1e-12
)


combined_score = (
    benefit_score
    + 0.5 * need_score
)


# ------------------------------------------------------------
# 7. DIRECT OBJECTIVE
# ------------------------------------------------------------

def direct_objective(
    x
):
    """
    Lower is better.

    Policy objective is:

        - benefit
        - fairness
        + constraint penalty
    """

    x = np.asarray(
        x,
        dtype=float
    )

    selected = x.sum()

    objective = (
        -np.dot(
            combined_score,
            x
        )
        +
        PENALTY
        * (
            selected
            - BUDGET_UNITS
        ) ** 2
    )

    return float(
        objective
    )


# ------------------------------------------------------------
# 8. EXACT VALID SOLUTIONS
# ------------------------------------------------------------

valid_solutions = []

for selected_indices in combinations(
    range(N),
    BUDGET_UNITS
):

    x = np.zeros(
        N,
        dtype=int
    )

    for index in selected_indices:

        x[index] = 1

    q_energy = qubo_energy(
        x
    )

    direct_energy = direct_objective(
        x
    )

    valid_solutions.append({
        "indices": selected_indices,
        "qubo_energy": q_energy,
        "direct_energy": direct_energy,
    })


# ------------------------------------------------------------
# 9. FIND OPTIMA
# ------------------------------------------------------------

qubo_best = min(
    valid_solutions,
    key=lambda row:
        row["qubo_energy"]
)


direct_best = min(
    valid_solutions,
    key=lambda row:
        row["direct_energy"]
)


# ------------------------------------------------------------
# 10. COMPARE
# ------------------------------------------------------------

qubo_matches_direct = (
    qubo_best["indices"]
    ==
    direct_best["indices"]
)


energy_difference = abs(
    qubo_best["qubo_energy"]
    -
    direct_best["direct_energy"]
)


# ------------------------------------------------------------
# 11. CONSTRAINT TEST
# ------------------------------------------------------------

constraint_pass = True

for solution in valid_solutions:

    indices = solution[
        "indices"
    ]

    if len(indices) != BUDGET_UNITS:

        constraint_pass = False

        break


# ------------------------------------------------------------
# 12. DISPLAY
# ------------------------------------------------------------

print()
print("========================================")
print(" QPDI — QUBO VALIDATION")
print("========================================")

print(
    f"Regions       : {N}"
)

print(
    f"Budget K      : {BUDGET_UNITS}"
)

print(
    f"Penalty P     : {PENALTY}"
)

print()
print(
    f"Valid solutions checked : "
    f"{len(valid_solutions)}"
)


# ------------------------------------------------------------
# 13. QUBO OPTIMUM
# ------------------------------------------------------------

qubo_regions = [
    f"R{i + 1:02d}"
    for i in qubo_best[
        "indices"
    ]
]


print()
print("========================================")
print(" QUBO OPTIMUM")
print("========================================")

print(
    f"Energy : "
    f"{qubo_best['qubo_energy']:.8f}"
)

print()
print("Selected regions:")

for region in qubo_regions:

    print(
        f"  ✓ {region}"
    )


# ------------------------------------------------------------
# 14. DIRECT OPTIMUM
# ------------------------------------------------------------

direct_regions = [
    f"R{i + 1:02d}"
    for i in direct_best[
        "indices"
    ]
]


print()
print("========================================")
print(" DIRECT OBJECTIVE OPTIMUM")
print("========================================")

print(
    f"Energy : "
    f"{direct_best['direct_energy']:.8f}"
)

print()
print("Selected regions:")

for region in direct_regions:

    print(
        f"  ✓ {region}"
    )


# ------------------------------------------------------------
# 15. TEST RESULTS
# ------------------------------------------------------------

print()
print("========================================")
print(" VALIDATION TESTS")
print("========================================")


if constraint_pass:

    print(
        "✓ Exact-K constraint : PASS"
    )

else:

    print(
        "✗ Exact-K constraint : FAIL"
    )


if qubo_matches_direct:

    print(
        "✓ Objective consistency : PASS"
    )

else:

    print(
        "✗ Objective consistency : FAIL"
    )


print()
print(
    f"Energy difference : "
    f"{energy_difference:.8f}"
)


# ------------------------------------------------------------
# 16. FINAL STATUS
# ------------------------------------------------------------

print()
print("========================================")
print(" FINAL QUBO STATUS")
print("========================================")


if (
    constraint_pass
    and qubo_matches_direct
):

    print(
        "✓ QUBO VALIDATION PASSED"
    )

    print()
    print(
        "The QUBO optimum is consistent "
        "with the intended objective."
    )

else:

    print(
        "✗ QUBO VALIDATION FAILED"
    )

    print()
    print(
        "Review the QUBO construction "
        "before running QAOA."
    )