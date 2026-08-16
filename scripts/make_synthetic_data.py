"""Generates a stand-in development/validation dataset so the pipeline is
buildable and testable before real bank data is wired up. Shape mirrors what
the original SAS pipeline expected from inp.development_base /
inp.validation_base: one row per application, a CUST_ID, a Good_Bad flag,
a sample_weight, and a mix of numeric/categorical predictor columns --
including some with missingness, a sentinel "dummy" value, outliers, and
both real signal and pure noise, so every Phase 1 stage has something to do.

Run: python scripts/make_synthetic_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.common.io import DATA_RAW_DIR  # noqa: E402

RNG_SEED = 42
BAD_RATE = 0.08


def _make_sample(n: int, rng: np.random.Generator, id_start: int) -> pd.DataFrame:
    # Latent risk score drives Good_Bad, so the "signal" columns below are
    # genuinely predictive and the "noise" columns genuinely aren't.
    latent_risk = rng.normal(0, 1, n)
    bad_prob = 1 / (1 + np.exp(-(latent_risk - 2.0)))  # keeps bad rate low
    good_bad = rng.binomial(1, np.clip(bad_prob, 0, 1))

    df = pd.DataFrame(
        {
            "CUST_ID": np.arange(id_start, id_start + n),
            "Good_Bad": good_bad,
            "sample_weight": 1.0,
            # numeric, correlated with risk -> should pass univariate screen
            "bureau_score": (700 - latent_risk * 60 + rng.normal(0, 20, n)).round(0),
            "num_30dpd_6m": np.clip((latent_risk * 1.5 + rng.normal(0, 1, n)), 0, None).round(0),
            "util_ratio_pct": np.clip(50 + latent_risk * 15 + rng.normal(0, 10, n), 0, 100).round(1),
            # numeric, pure noise -> should fail univariate screen
            "noise_num_1": rng.normal(500, 50, n).round(1),
            "noise_num_2": rng.uniform(0, 1, n).round(3),
            # numeric with a sentinel "dummy" code (e.g. -1 = not applicable)
            "months_on_book": np.where(rng.uniform(0, 1, n) < 0.15, -1, rng.integers(1, 240, n)),
            # numeric with heavy missingness (tests fill-rate screening)
            "collections_bal": np.where(
                rng.uniform(0, 1, n) < 0.75, np.nan, np.clip(latent_risk * 500 + rng.normal(0, 200, n), 0, None)
            ).round(2),
            # numeric with a handful of extreme outliers (tests outlier stage)
            "annual_income": np.clip(rng.normal(600000, 150000, n), 50000, None),
            # categorical, correlated with risk -> should pass univariate screen
            "employment_type": np.select(
                [latent_risk < -0.5, latent_risk < 0.5],
                ["salaried", "self_employed"],
                default="unemployed",
            ),
            # categorical, pure noise
            "state": rng.choice(["MH", "DL", "KA", "TN", "UP", "WB"], size=n),
            # categorical with some missing values
            "channel": np.where(
                rng.uniform(0, 1, n) < 0.10, None, rng.choice(["branch", "online", "dsa"], size=n)
            ),
        }
    )

    # sprinkle a few extreme outliers into annual_income
    outlier_idx = rng.choice(n, size=max(1, n // 200), replace=False)
    df.loc[outlier_idx, "annual_income"] = rng.uniform(5_000_000, 20_000_000, len(outlier_idx))

    return df


def main():
    rng = np.random.default_rng(RNG_SEED)
    dev = _make_sample(5000, rng, id_start=100000)
    val = _make_sample(2000, rng, id_start=900000)

    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    dev.to_parquet(DATA_RAW_DIR / "development_base.parquet", index=False)
    val.to_parquet(DATA_RAW_DIR / "validation_base.parquet", index=False)

    print(f"wrote {len(dev)} development rows, {len(val)} validation rows to {DATA_RAW_DIR}")
    print(f"development bad rate: {dev['Good_Bad'].mean():.3%}")
    print(f"validation bad rate:  {val['Good_Bad'].mean():.3%}")


if __name__ == "__main__":
    main()
