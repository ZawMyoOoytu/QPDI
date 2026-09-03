from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATE_FILE = (
    PROJECT_ROOT
    / "data"
    / "quantum_candidates.csv"
)

QUBO_OUTPUT_FILE = (
    PROJECT_ROOT
    / "quantum_engine"
    / "qubo_matrix.npy"
)


# ============================================================
# CONFIGURATION
# ============================================================

BUDGET_UNITS = 5

BENEFIT_WEIGHT = 1.0
NEED_WEIGHT = 0.5

CONSTRAINT_PENALTY = 3.0


# ============================================================
# LOAD AI CANDIDATES
# ============================================================

if not CANDIDATE_FILE.exists():
    raise FileNotFoundError(
        f"Candidate file not found:\n"
        f"{CANDIDATE_FILE}\n\n"
        "Run candidate_selector.py first."
    )


df = pd.read_csv(
    CANDIDATE_FILE
)


# ============================================================
# VALIDATION
# ============================================================

required_columns = [
    "ai_rank",
    "region",
    "need",
    "predicted_benefit",
    "optimization_score",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + ", ".join(missing_columns)
    )


N = len(df)

if BUDGET_UNITS > N:
    raise ValueError(
        "Budget cannot exceed "
        "number of candidates."
    )


# ============================================================
# NORMALIZE BENEFIT
# ============================================================

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


# ============================================================
# COMBINED OBJECTIVE
# ============================================================

combined_score = (
    BENEFIT_WEIGHT
    * benefit_score
    +
    NEED_WEIGHT
    * need_score
)


df["qubo_benefit_score"] = (
    benefit_score
)

df["qubo_need_score"] = (
    need_score
)

df["qubo_objective_score"] = (
    combined_score
)


# ============================================================
# BUILD QUBO
#
# Minimize:
#
#     - objective
#     + P * (sum(x_i) - K)^2
#
# x_i ∈ {0,1}
#
# K = exactly 5 selected regions
# ============================================================

Q = np.zeros(
    (N, N),
    dtype=float
)


# Objective:
# maximize combined_score
# therefore minimize -combined_score

for i in range(N):
    Q[i, i] -= combined_score[i]


# Exact-K constraint

K = BUDGET_UNITS
P = CONSTRAINT_PENALTY


# Diagonal contribution

for i in range(N):
    Q[i, i] += (
        P * (1 - 2 * K)
    )


# Pairwise contribution

for i in range(N):
    for j in range(i + 1, N):
        Q[i, j] += (
            2 * P
        )


# ============================================================
# SAVE QUBO
# ============================================================

np.save(
    QUBO_OUTPUT_FILE,
    Q
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("========================================")
print(" QPDI — AI → QUBO")
print("========================================")

print(
    f"Input file       : {CANDIDATE_FILE}"
)

print(
    f"Candidate count  : {N}"
)

print(
    f"Budget K         : {BUDGET_UNITS}"
)

print(
    f"Benefit weight   : {BENEFIT_WEIGHT}"
)

print(
    f"Need weight      : {NEED_WEIGHT}"
)

print(
    f"Penalty P        : {CONSTRAINT_PENALTY}"
)


print()
print("========================================")
print(" QUBIT → REGION MAPPING")
print("========================================")

for qubit, region in enumerate(
    df["region"]
):
    print(
        f"Qubit {qubit} → {region}"
    )


print()
print("========================================")
print(" QUANTUM CANDIDATES")
print("========================================")

print(
    df[
        [
            "ai_rank",
            "region",
            "need",
            "predicted_benefit",
            "qubo_objective_score",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


print()
print("========================================")
print(" QUBO CREATED")
print("========================================")

print(
    f"QUBO shape : {Q.shape}"
)

print(
    f"Qubits     : {N}"
)

print(
    f"Selected K : {BUDGET_UNITS}"
)

print(
    f"Saved to   : {QUBO_OUTPUT_FILE}"
)


print()
print("========================================")
print(" QUANTUM PIPELINE")
print("========================================")

print("100 synthetic regions")
print("        ↓")
print("Random Forest prediction")
print("        ↓")
print("AI ranking")
print("        ↓")
print("Top 8 candidates")
print("        ↓")
print("8 × 8 QUBO")
print("        ↓")
print("8 qubits")
print("        ↓")
print("Ising Hamiltonian")
print("        ↓")
print("QAOA")