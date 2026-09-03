from pathlib import Path

import pandas as pd


# ============================================================
# QPDI — AI TOP-K CANDIDATE SELECTOR
#
# 100 regions
#      ↓
# AI optimization score
#      ↓
# Top-K candidates
#      ↓
# Quantum optimization
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

AI_INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ai_policy_predictions.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "quantum_candidates.csv"
)


# ------------------------------------------------------------
# 2. SETTINGS
# ------------------------------------------------------------

TOP_K = 8

BUDGET_UNITS = 5


# ------------------------------------------------------------
# 3. LOAD AI RESULTS
# ------------------------------------------------------------

if not AI_INPUT_FILE.exists():

    raise FileNotFoundError(
        f"AI prediction file not found:\n"
        f"{AI_INPUT_FILE}\n\n"
        "Run policy_predictor.py first."
    )


df = pd.read_csv(
    AI_INPUT_FILE
)


# ------------------------------------------------------------
# 4. VALIDATE
# ------------------------------------------------------------

required_columns = [
    "region",
    "need",
    "predicted_benefit",
    "optimization_score",
]


missing = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing:

    raise ValueError(
        "Missing columns: "
        + ", ".join(missing)
    )


if TOP_K > len(df):

    raise ValueError(
        "TOP_K cannot exceed "
        "the number of regions."
    )


if BUDGET_UNITS > TOP_K:

    raise ValueError(
        "Budget cannot exceed "
        "candidate count."
    )


# ------------------------------------------------------------
# 5. SORT BY AI SCORE
# ------------------------------------------------------------

ranked_df = df.sort_values(
    "optimization_score",
    ascending=False,
).reset_index(
    drop=True
)


# ------------------------------------------------------------
# 6. SELECT TOP-K
# ------------------------------------------------------------

candidates = ranked_df.head(
    TOP_K
).copy()


# ------------------------------------------------------------
# 7. ADD RANK
# ------------------------------------------------------------

candidates.insert(
    0,
    "ai_rank",
    range(
        1,
        TOP_K + 1
    )
)


# ------------------------------------------------------------
# 8. SAVE
# ------------------------------------------------------------

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 9. DISPLAY
# ------------------------------------------------------------

print()
print("========================================")
print(" QPDI — AI CANDIDATE SELECTOR")
print("========================================")

print(
    f"Original regions : {len(df)}"
)

print(
    f"Quantum candidates: {TOP_K}"
)

print(
    f"Budget units     : {BUDGET_UNITS}"
)


print()
print("========================================")
print(" TOP AI CANDIDATES")
print("========================================")


print(
    candidates[
        [
            "ai_rank",
            "region",
            "need",
            "predicted_benefit",
            "optimization_score",
        ]
    ]
    .round(4)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 10. OUTPUT
# ------------------------------------------------------------

print()
print("========================================")
print(" CANDIDATE SELECTION COMPLETE")
print("========================================")

print(
    f"Saved to:\n{OUTPUT_FILE}"
)


print()
print("Pipeline:")

print(
    "100 regions"
)

print(
    "    ↓"
)

print(
    "AI ranking"
)

print(
    "    ↓"
)

print(
    f"Top {TOP_K} candidates"
)

print(
    "    ↓"
)

print(
    "Quantum optimization"
)