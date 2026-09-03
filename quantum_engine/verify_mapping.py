from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUBO_FILE = PROJECT_ROOT / "quantum_engine" / "qubo_matrix.npy"
CANDIDATE_FILE = PROJECT_ROOT / "data" / "quantum_candidates.csv"


Q = np.load(QUBO_FILE)
df = pd.read_csv(CANDIDATE_FILE)


BITSTRING = "00011111"


def energy_from_x(x):
    return float(x @ Q @ x)


print()
print("========================================")
print(" QPDI — BITSTRING / REGION MAPPING")
print("========================================")

print()
print("Candidate order:")
print("----------------------------------------")

for i, row in df.iterrows():
    print(
        f"Qubit {i} → "
        f"{row['region']}"
    )


print()
print("========================================")
print(" BITSTRING TEST")
print("========================================")

print(
    f"Bitstring: {BITSTRING}"
)


# ------------------------------------------------------------
# Convention A
# ------------------------------------------------------------

x_normal = np.array(
    [int(bit) for bit in BITSTRING],
    dtype=int
)

indices_normal = tuple(
    np.where(x_normal == 1)[0]
)

regions_normal = [
    str(df.iloc[i]["region"])
    for i in indices_normal
]

energy_normal = energy_from_x(
    x_normal
)


print()
print("Convention A: left-to-right")
print("----------------------------------------")
print(
    f"Indices : {indices_normal}"
)
print(
    f"Regions : {regions_normal}"
)
print(
    f"Energy  : {energy_normal:.9f}"
)


# ------------------------------------------------------------
# Convention B
# ------------------------------------------------------------

x_reversed = np.array(
    [int(bit) for bit in BITSTRING[::-1]],
    dtype=int
)

indices_reversed = tuple(
    np.where(x_reversed == 1)[0]
)

regions_reversed = [
    str(df.iloc[i]["region"])
    for i in indices_reversed
]

energy_reversed = energy_from_x(
    x_reversed
)


print()
print("Convention B: reversed")
print("----------------------------------------")
print(
    f"Indices : {indices_reversed}"
)
print(
    f"Regions : {regions_reversed}"
)
print(
    f"Energy  : {energy_reversed:.9f}"
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print()
print("========================================")
print(" MAPPING SUMMARY")
print("========================================")

print(
    "Use exactly ONE convention throughout "
    "the QAOA pipeline."
)

print()
print(
    "For Qiskit measurement output, remember:"
)

print(
    "Qiskit display bitstrings are written "
    "with the highest-index qubit on the left."
)

print()
print(
    "Therefore QAOA decoding should reverse "
    "the measured bitstring ONCE before mapping "
    "to qubit indices."
)