"""Studentized block bootstrap-t inference for two-way FE regressions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bootstrap_utils import iter_demean, slope_se

BOOTSTRAP_REPLICATES = 3000
SEED = 42


def _residualize(values: np.ndarray, controls: np.ndarray | None) -> np.ndarray:
    """OLS-residualize values on controls (no-op if controls is None or empty)."""
    if controls is None or controls.shape[1] == 0:
        return values
    return values - controls @ np.linalg.lstsq(controls, values, rcond=None)[0]


def run(df: pd.DataFrame, regressor_col: str, control_cols: list[str],
        fixed_effects: list[str], block_length: int = 8,
        bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
        seed: int = SEED, fast: bool = True,
        b: int | None = None) -> dict:
    """Run a two-way FE regression with studentized block bootstrap-t inference.

    `b` is an alias for `bootstrap_replicates` kept for backwards compatibility
    with entry scripts that pass `b=` as a keyword argument.
    """
    if b is not None:
        bootstrap_replicates = b
    required_cols = ["log1p_alo", regressor_col] + control_cols
    panel = (df.replace([np.inf, -np.inf], np.nan)
               .dropna(subset=required_cols + ["user"])
               .reset_index(drop=True))
    fe0_codes = pd.factorize(panel[fixed_effects[0]].to_numpy())[0]
    fe1_codes = pd.factorize(panel[fixed_effects[1]].to_numpy())[0]
    n_fe0, n_fe1 = fe0_codes.max() + 1, fe1_codes.max() + 1
    cluster_codes, unique_clusters = pd.factorize(panel["user"].to_numpy())
    n_clusters = len(unique_clusters)
    outcome_raw = panel["log1p_alo"].to_numpy(float)
    regressor_raw = panel[regressor_col].to_numpy(float)
    controls_raw = panel[control_cols].to_numpy(float) if control_cols else None

    def residualize_sample(sample_rows: np.ndarray):
        outcome_demeaned = iter_demean(outcome_raw[sample_rows],
                                       fe0_codes[sample_rows], fe1_codes[sample_rows],
                                       n_fe0, n_fe1)
        regressor_demeaned = iter_demean(regressor_raw[sample_rows],
                                         fe0_codes[sample_rows], fe1_codes[sample_rows],
                                         n_fe0, n_fe1)
        controls_demeaned = None
        if control_cols:
            controls_demeaned = np.column_stack([
                iter_demean(controls_raw[sample_rows, j],
                            fe0_codes[sample_rows], fe1_codes[sample_rows],
                            n_fe0, n_fe1)
                for j in range(len(control_cols))
            ])
        return (_residualize(regressor_demeaned, controls_demeaned),
                _residualize(outcome_demeaned, controls_demeaned))

    full_rows = np.arange(len(panel))
    regressor_full, outcome_full = residualize_sample(full_rows)
    beta_hat, se_hat = slope_se(regressor_full, outcome_full,
                                cluster_codes, n_clusters)
    t_observed = beta_hat / se_hat
    p_normal = float(2 * stats.norm.sf(abs(t_observed)))

    time_codes = pd.factorize(panel["tkey"].to_numpy())[0]
    n_periods = time_codes.max() + 1
    rows_by_period = [np.where(time_codes == i)[0] for i in range(n_periods)]
    n_blocks = int(np.ceil(n_periods / block_length))
    start_max = max(n_periods - block_length + 1, 1)
    rng = np.random.default_rng(seed)
    t_bootstrap = np.empty(bootstrap_replicates)

    for k in range(bootstrap_replicates):
        starts = rng.integers(0, start_max, n_blocks)
        sequence = np.concatenate([np.arange(s, s + block_length)
                                   for s in starts])[:n_periods]
        sequence = sequence[sequence < n_periods]
        sample_rows = np.concatenate([rows_by_period[i] for i in sequence])
        if fast:
            regressor_boot = regressor_full[sample_rows]
            outcome_boot = outcome_full[sample_rows]
        else:
            regressor_boot, outcome_boot = residualize_sample(sample_rows)
        beta_star, se_star = slope_se(regressor_boot, outcome_boot,
                                      cluster_codes[sample_rows], n_clusters)
        t_bootstrap[k] = ((beta_star - beta_hat) / se_star
                          if np.isfinite(se_star) and se_star != 0 else np.nan)

    t_bootstrap = t_bootstrap[np.isfinite(t_bootstrap)]
    p_bootstrap = float(np.mean(np.abs(t_bootstrap) >= abs(t_observed)))
    q_upper, q_lower = np.percentile(t_bootstrap, [97.5, 2.5])
    return {"beta": beta_hat, "se": se_hat, "t_obs": t_observed,
            "p_cluster_normal": p_normal, "p_bootstrap_t": p_bootstrap,
            "boot_t_ci_lo": beta_hat - q_upper * se_hat,
            "boot_t_ci_hi": beta_hat - q_lower * se_hat,
            "B": int(t_bootstrap.size)}
