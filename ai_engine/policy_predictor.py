from pathlib import Path
import sys

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "ai_policy_predictions.csv"
)

sys.path.append(
    str(PROJECT_ROOT)
)

from data.synthetic_policy_data import (
    create_policy_dataset
)


# ============================================================
# CREATE DATASET
# ============================================================

df = create_policy_dataset(
    n_regions=100
)


# ============================================================
# FEATURES
# ============================================================

features = [
    "health",
    "education",
    "poverty",
    "infrastructure",
    "disaster_risk",
    "internet_access",
]


X = df[features]
y = df["observed_benefit"]


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )
)


# ============================================================
# TRAIN MODEL
# ============================================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
)

model.fit(
    X_train,
    y_train,
)


# ============================================================
# EVALUATION
# ============================================================

test_predictions = model.predict(
    X_test
)

mae = mean_absolute_error(
    y_test,
    test_predictions,
)

r2 = r2_score(
    y_test,
    test_predictions,
)


# ============================================================
# PREDICT ALL REGIONS
# ============================================================

df["predicted_benefit"] = model.predict(
    X
)


# ============================================================
# NORMALIZE BENEFIT
# ============================================================

benefit_min = df[
    "predicted_benefit"
].min()

benefit_max = df[
    "predicted_benefit"
].max()

df["predicted_benefit_score"] = (
    df["predicted_benefit"]
    - benefit_min
) / (
    benefit_max
    - benefit_min
    + 1e-12
)


# ============================================================
# NORMALIZE NEED
# ============================================================

need_min = df["need"].min()

need_max = df["need"].max()

df["need_score"] = (
    df["need"]
    - need_min
) / (
    need_max
    - need_min
    + 1e-12
)


# ============================================================
# AI OPTIMIZATION SCORE
# ============================================================

df["optimization_score"] = (
    1.0
    * df["predicted_benefit_score"]
    +
    0.5
    * df["need_score"]
)


# ============================================================
# RANK REGIONS
# ============================================================

df = df.sort_values(
    "optimization_score",
    ascending=False,
).reset_index(
    drop=True
)


# ============================================================
# SAVE AI OUTPUT
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# DISPLAY
# ============================================================

print()
print("====================================")
print(" QPDI — AI POLICY INTELLIGENCE")
print("====================================")

print(
    f"Dataset size : {len(df)} regions"
)

print(
    f"Training     : {len(X_train)} regions"
)

print(
    f"Testing      : {len(X_test)} regions"
)

print()
print(
    f"MAE          : {mae:.4f}"
)

print(
    f"R²           : {r2:.4f}"
)


print()
print("====================================")
print(" TOP 10 PREDICTED POLICY PRIORITIES")
print("====================================")

print(
    df[
        [
            "region",
            "need",
            "predicted_benefit",
            "optimization_score",
        ]
    ]
    .head(10)
    .round(4)
    .to_string(index=False)
)


print()
print("====================================")
print(" AI OUTPUT SAVED")
print("====================================")

print(
    OUTPUT_FILE
)