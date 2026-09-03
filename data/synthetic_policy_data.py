import numpy as np
import pandas as pd


SEED = 42

rng = np.random.default_rng(SEED)


def create_policy_dataset(n_regions=20):

    data = []

    for i in range(n_regions):

        population = rng.integers(
            50_000,
            500_000
        )

        health = rng.uniform(
            0.35,
            0.90
        )

        education = rng.uniform(
            0.30,
            0.90
        )

        poverty = rng.uniform(
            0.05,
            0.60
        )

        infrastructure = rng.uniform(
            0.30,
            0.90
        )

        disaster_risk = rng.uniform(
            0.05,
            0.80
        )

        internet_access = rng.uniform(
            0.30,
            0.95
        )

        # Synthetic policy-need score
        need = (
            0.25 * (1 - health)
            + 0.20 * (1 - education)
            + 0.25 * poverty
            + 0.20 * (1 - infrastructure)
            + 0.10 * disaster_risk
        )

        # Synthetic expected policy benefit
        benefit = (
            0.35 * need
            + 0.20 * (1 - health)
            + 0.15 * (1 - education)
            + 0.15 * poverty
            + 0.10 * disaster_risk
            + 0.05 * internet_access
        )

        # Synthetic noise
        benefit += rng.normal(
            0,
            0.02
        )

        data.append({
            "region": f"R{i + 1:02d}",
            "population": population,
            "health": health,
            "education": education,
            "poverty": poverty,
            "infrastructure": infrastructure,
            "disaster_risk": disaster_risk,
            "internet_access": internet_access,
            "need": need,
            "observed_benefit": benefit,
        })

    return pd.DataFrame(data)


if __name__ == "__main__":

    df = create_policy_dataset(
        n_regions=20
    )

    print("\n=== SYNTHETIC POLICY DATASET ===")

    print(
        df.round(3).to_string(
            index=False
        )
    )