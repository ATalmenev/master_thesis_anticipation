"""Closed-form calibration of tail-loss intensity lambda/r."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
HORSERACE_FILE = PROCESSED / "wallet_horserace_extended_summary.csv"
OUT = PROCESSED / "calibration_lambda_r_summary.csv"

PRESS_MIN = 0.10
PRESS_MAX = 0.85
SPEC_LABEL = "(5) + qz-linear trend"


def implied_lambda_r(beta_hat: float, press_min: float = PRESS_MIN,
                     press_max: float = PRESS_MAX) -> float:
    """Solve log[(1 - press_max * x) / (1 - press_min * x)] = beta_hat."""
    def f(x: float) -> float:
        return math.log((1.0 - press_max * x) / (1.0 - press_min * x)) - beta_hat
    return brentq(f, 1e-6, 1.0 / press_max - 1e-6)


def main() -> None:
    """Calibrate lambda/r from the ALO coefficient."""
    if not HORSERACE_FILE.exists():
        sys.exit(f"missing {HORSERACE_FILE.name}: run horserace_ladder.py first")
    df = pd.read_csv(HORSERACE_FILE)
    head_row = df[df["col"].str.contains("qz-linear trend", regex=False)]
    if head_row.empty:
        sys.exit(f"linear-trend spec not found in {HORSERACE_FILE.name}")
    beta_hat = float(head_row["beta"].iloc[0])

    lambda_r = implied_lambda_r(beta_hat)
    semi_elasticity = math.exp(beta_hat) - 1.0

    print("Cohort                                : full tier-1+")
    print(f"beta_hat                              : {beta_hat:+.4f}")
    print(f"Semi-elasticity exp(beta_hat) - 1     : {semi_elasticity:+.4f} "
          f"({100 * semi_elasticity:+.1f}%)")
    print(f"Implied lambda/r                      : {lambda_r:.4f}")
    print(f"  with press_min = {PRESS_MIN}, press_max = {PRESS_MAX}")

    pd.DataFrame([{
        "cohort": "full_tier1plus",
        "beta_hat": beta_hat,
        "semi_elasticity": semi_elasticity,
        "lambda_r": lambda_r,
        "press_min": PRESS_MIN,
        "press_max": PRESS_MAX,
    }]).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
