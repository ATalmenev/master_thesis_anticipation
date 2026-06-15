"""Numerical helpers for the block bootstrap-t (demeaning, clustered slope SE)."""

from __future__ import annotations

import numpy as np
import pandas as pd

BOOTSTRAP_REPLICATES = 3000
SEED = 42


def iter_demean(values: np.ndarray, fe0_codes: np.ndarray, fe1_codes: np.ndarray,
                n_fe0: int, n_fe1: int,
                tol: float = 1e-10, maxit: int = 200) -> np.ndarray:
    """Two-way fixed-effect demeaning via alternating Gauss-Seidel sweeps."""
    counts_fe0 = np.bincount(fe0_codes, minlength=n_fe0)
    counts_fe0 = np.where(counts_fe0 > 0, counts_fe0, 1)
    counts_fe1 = np.bincount(fe1_codes, minlength=n_fe1)
    counts_fe1 = np.where(counts_fe1 > 0, counts_fe1, 1)
    current = values.copy()
    for _ in range(maxit):
        current = current - (np.bincount(fe0_codes, current, n_fe0) / counts_fe0)[fe0_codes]
        updated = current - (np.bincount(fe1_codes, current, n_fe1) / counts_fe1)[fe1_codes]
        if np.max(np.abs(updated - current)) < tol:
            return updated
        current = updated
    return current


def slope_se(regressor: np.ndarray, outcome: np.ndarray,
             cluster_codes: np.ndarray, n_clusters: int):
    """OLS slope and cluster-robust standard error."""
    regressor_ss = float(regressor @ regressor)
    if regressor_ss <= 0:
        return np.nan, np.nan
    beta = float(regressor @ outcome) / regressor_ss
    score = regressor * (outcome - beta * regressor)
    cluster_scores = np.bincount(cluster_codes, weights=score, minlength=n_clusters)
    std_err = np.sqrt(float(cluster_scores @ cluster_scores)) / regressor_ss
    return beta, (std_err if std_err > 0 else np.nan)


def bootstrap_t(df: pd.DataFrame, outcome_col: str, regressor_col: str,
                fixed_effects: list[str], block_length: int = 8,
                bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
                seed: int = SEED) -> dict:
    """Studentized block bootstrap-t for a clustered FE regression."""
    panel = df.dropna(subset=[outcome_col, regressor_col]).reset_index(drop=True)
    fe0_codes = pd.factorize(panel[fixed_effects[0]].to_numpy())[0]
    fe1_codes = pd.factorize(panel[fixed_effects[1]].to_numpy())[0]
    n_fe0, n_fe1 = fe0_codes.max() + 1, fe1_codes.max() + 1
    cluster_codes, unique_clusters = pd.factorize(panel["user"].to_numpy())
    n_clusters = len(unique_clusters)
    regressor = panel[regressor_col].to_numpy(float)
    outcome = panel[outcome_col].to_numpy(float)

    regressor_demeaned = iter_demean(regressor, fe0_codes, fe1_codes, n_fe0, n_fe1)
    outcome_demeaned = iter_demean(outcome, fe0_codes, fe1_codes, n_fe0, n_fe1)
    beta_hat, se_hat = slope_se(regressor_demeaned, outcome_demeaned,
                                cluster_codes, n_clusters)
    t_observed = beta_hat / se_hat

    time_codes = pd.factorize(panel["tkey"].to_numpy())[0]
    n_periods = time_codes.max() + 1
    rows_by_period = [np.where(time_codes == i)[0] for i in range(n_periods)]
    n_blocks = int(np.ceil(n_periods / block_length))
    start_max = max(n_periods - block_length + 1, 1)
    rng = np.random.default_rng(seed)

    t_bootstrap = np.empty(bootstrap_replicates)
    for k in range(bootstrap_replicates):
        starts = rng.integers(0, start_max, n_blocks)
        sequence = np.concatenate([np.arange(s, s + block_length) for s in starts])[:n_periods]
        sequence = sequence[sequence < n_periods]
        sample_rows = np.concatenate([rows_by_period[i] for i in sequence])
        regressor_boot = iter_demean(regressor[sample_rows], fe0_codes[sample_rows],
                                     fe1_codes[sample_rows], n_fe0, n_fe1)
        outcome_boot = iter_demean(outcome[sample_rows], fe0_codes[sample_rows],
                                   fe1_codes[sample_rows], n_fe0, n_fe1)
        beta_star, se_star = slope_se(regressor_boot, outcome_boot,
                                      cluster_codes[sample_rows], n_clusters)
        t_bootstrap[k] = ((beta_star - beta_hat) / se_star
                          if np.isfinite(se_star) and se_star != 0 else np.nan)
    t_bootstrap = t_bootstrap[np.isfinite(t_bootstrap)]
    p_two_sided = float(np.mean(np.abs(t_bootstrap) >= abs(t_observed)))

    def ci(alpha: float):
        q_lo, q_hi = np.percentile(t_bootstrap,
                                   [100 * (1 - alpha / 2), 100 * (alpha / 2)])
        return beta_hat - q_lo * se_hat, beta_hat - q_hi * se_hat

    lo95, hi95 = ci(0.05)
    lo99, hi99 = ci(0.01)
    lo999, hi999 = ci(0.001)
    sig_level = ("0.001" if hi999 < 0 or lo999 > 0
                 else "0.01" if hi99 < 0 or lo99 > 0
                 else "0.05" if hi95 < 0 or lo95 > 0 else "ns")
    return {"beta_hat": beta_hat, "se_hat": se_hat, "t_obs": t_observed,
            "boot_t_p_two": p_two_sided, "B": int(t_bootstrap.size),
            "ci95": (lo95, hi95), "ci99": (lo99, hi99), "ci999": (lo999, hi999),
            "sig_level": sig_level}


