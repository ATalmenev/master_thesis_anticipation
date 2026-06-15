"""8-column ALO horse-race ladder on the full tier-1+ cohort.

Coefficient on per-coin signed pressure interacted with predetermined
exposure, log(1+ALO) outcome, as fixed effects, the lagged outcome, the
exposure-time trends, and the five channel controls are added one
specification at a time. Studentized block bootstrap-t clustered on the
21 fund masters.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

ROOT: Final = Path(__file__).resolve().parents[1]
SRC: Final = ROOT / "src"
sys.path.insert(0, str(SRC))

from bootstrap_utils import iter_demean, slope_se
from pressure_signal import (
    PANEL_FILE,
    build_per_coin_pressure,
    load_sub_to_entity,
)
from cohort import (
    aggregate_to_wallet,
    load_tier_cohort,
)

PROCESSED: Final = ROOT / "data" / "processed"
L2_CONTROLS_FILE: Final = PROCESSED / "l2_controls_minute_4h_20251010.csv"
OUTPUT_FILE: Final = PROCESSED / "wallet_horserace_extended_summary.csv"

BOOTSTRAP_REPLICATES: Final = int(os.environ.get("BOOT_B", "3000"))
BLOCK_LENGTH: Final = 8  # 8 x 5s = 40s, one half-life at rho_hat_1 = 0.9
SEED: Final = 42
CHANNELS: Final = ("vol", "spread", "forced", "depth", "funding")


def _residualize(values: np.ndarray, controls: np.ndarray | None) -> np.ndarray:
    """OLS-residualize values on controls (no-op if controls is None)."""
    if controls is None or controls.shape[1] == 0:
        return values
    beta, *_ = np.linalg.lstsq(controls, values, rcond=None)
    return values - controls @ beta


def run_specification(
    panel: pd.DataFrame,
    fixed_effects: list[str],
    controls: list[str],
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, float]:
    """Estimate beta on press_x_qz under given FE and controls; return boot-t p."""
    if controls:
        panel = panel.dropna(subset=controls).reset_index(drop=True)
    outcome = panel["log1p_alo"].to_numpy(float)
    regressor = panel["press_x_qz"].to_numpy(float)
    n_obs = len(panel)

    if not fixed_effects:
        outcome_d = outcome - outcome.mean()
        regressor_d = regressor - regressor.mean()
        controls_d = (
            panel[controls].to_numpy(float) - panel[controls].to_numpy(float).mean(0)
            if controls else None
        )
    else:
        fe0 = pd.factorize(panel[fixed_effects[0]].to_numpy())[0]
        fe1 = (
            pd.factorize(panel[fixed_effects[1]].to_numpy())[0]
            if len(fixed_effects) == 2 else np.zeros(n_obs, dtype=int)
        )
        n0, n1 = int(fe0.max()) + 1, int(fe1.max()) + 1
        outcome_d = iter_demean(outcome, fe0, fe1, n0, n1)
        regressor_d = iter_demean(regressor, fe0, fe1, n0, n1)
        controls_d = (
            np.column_stack([
                iter_demean(panel[c].to_numpy(float), fe0, fe1, n0, n1)
                for c in controls
            ])
            if controls else None
        )

    regressor_r = _residualize(regressor_d, controls_d)
    outcome_r = _residualize(outcome_d, controls_d)

    cluster_ids, unique = pd.factorize(panel["user"].to_numpy())
    n_clusters = len(unique)
    beta, std_err = slope_se(regressor_r, outcome_r, cluster_ids, n_clusters)
    t_observed = beta / std_err

    residual = outcome_r - beta * regressor_r
    total_ss = float(((outcome - outcome.mean()) ** 2).sum())
    residual_ss = float((residual ** 2).sum())
    r_squared = 1.0 - residual_ss / total_ss

    time_codes = pd.factorize(panel["tkey"].to_numpy())[0]
    n_periods = int(time_codes.max()) + 1
    rows_by_period = [np.where(time_codes == t)[0] for t in range(n_periods)]
    n_blocks = int(np.ceil(n_periods / BLOCK_LENGTH))
    start_max = max(n_periods - BLOCK_LENGTH + 1, 1)

    rng = np.random.default_rng(SEED)
    t_bootstrap = np.empty(bootstrap_replicates)
    for k in range(bootstrap_replicates):
        starts = rng.integers(0, start_max, n_blocks)
        sequence = np.concatenate([np.arange(s, s + BLOCK_LENGTH) for s in starts])[:n_periods]
        sequence = sequence[sequence < n_periods]
        rows = np.concatenate([rows_by_period[t] for t in sequence])
        beta_star, se_star = slope_se(
            regressor_r[rows], outcome_r[rows], cluster_ids[rows], n_clusters,
        )
        t_bootstrap[k] = (
            (beta_star - beta) / se_star
            if np.isfinite(se_star) and se_star != 0 else np.nan
        )

    t_bootstrap = t_bootstrap[np.isfinite(t_bootstrap)]
    p_bootstrap = float(np.mean(np.abs(t_bootstrap) >= abs(t_observed)))

    return {
        "beta": beta,
        "se": std_err,
        "p_boot": p_bootstrap,
        "n": n_obs,
        "r2": r_squared,
    }


def build_panel() -> pd.DataFrame:
    """Build the full tier-1+ wallet-coin-5s panel with channel controls."""
    sub_to_entity = load_sub_to_entity()
    tier_info = load_tier_cohort()
    tier_wallets = {wallet for wallet, info in tier_info.items() if info["tier"] >= 1}

    per_coin_pressure = build_per_coin_pressure()
    raw_panel = pd.read_csv(PANEL_FILE)
    wallet_panel = aggregate_to_wallet(raw_panel, sub_to_entity, per_coin_pressure)
    wallet_panel = wallet_panel[wallet_panel["wallet_id"].isin(tier_wallets)].copy()

    coin_bucket_state = (
        raw_panel.assign(coin=raw_panel["coin"].astype(str))
        .groupby(["coin", "bucket"], as_index=False)
        .agg(
            vol=("max_abs_mid_return_1s", "first"),
            spread=("mean_spread_bps", "first"),
            forced=("forced_flow_notional", "first"),
        )
    )
    wallet_panel = wallet_panel.merge(coin_bucket_state, on=["coin", "bucket"], how="left")

    minute_controls = pd.read_csv(L2_CONTROLS_FILE)
    minute_controls["coin"] = minute_controls["coin"].astype(str)
    minute_controls["minute"] = (
        pd.to_datetime(minute_controls["minute_dt"], utc=True)
        .dt.strftime("%Y-%m-%d %H:%M")
    )
    wallet_panel["minute"] = wallet_panel["bucket_dt"].dt.strftime("%Y-%m-%d %H:%M")
    wallet_panel = wallet_panel.merge(
        minute_controls[["coin", "minute", "mean_depth_usd", "funding"]],
        on=["coin", "minute"], how="left",
    ).rename(columns={"mean_depth_usd": "depth"})

    for channel in CHANNELS:
        column = wallet_panel[channel].astype(float)
        wallet_panel[channel] = (column - column.mean()) / (column.std() + 1e-12)
        wallet_panel[f"{channel}_x_qz"] = wallet_panel[channel] * wallet_panel["q_z"]

    wallet_panel = wallet_panel.sort_values(["user", "coin", "bucket_dt"]).reset_index(drop=True)
    for channel in CHANNELS:
        wallet_panel[f"{channel}_x_qz_lag1"] = (
            wallet_panel.groupby(["user", "coin"])[f"{channel}_x_qz"].shift(1)
        )

    required = ["log1p_alo", "press_x_qz", "user", "log1p_alo_lag1",
                "t_x_qz", "t2_x_qz"]
    return (
        wallet_panel.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=required)
        .reset_index(drop=True)
    )


def main() -> None:
    """Run the 8-column horse-race and write the summary CSV."""
    print(f"BOOTSTRAP_REPLICATES={BOOTSTRAP_REPLICATES}")
    panel = build_panel()
    print(
        f"panel rows={len(panel):,} "
        f"wallets={panel['wallet_id'].nunique()} "
        f"masters={panel['user'].nunique()}"
    )

    fe = ["user_coin", "coin_minute"]
    lag_only = ["log1p_alo_lag1"]
    linear_trend = ["log1p_alo_lag1", "t_x_qz"]
    quadratic_trend = ["log1p_alo_lag1", "t_x_qz", "t2_x_qz"]
    channels_contemp = [f"{c}_x_qz" for c in CHANNELS]
    channels_lagged = [f"{c}_x_qz_lag1" for c in CHANNELS]

    specifications: list[tuple[str, list[str], list[str]]] = [
        ("(1) naive OLS",                  [],            []),
        ("(2) + wallet-coin FE",           ["user_coin"], []),
        ("(3) + coin-bucket FE",           fe,            []),
        ("(4) + lagged outcome",           fe,            lag_only),
        ("(5) + qz-linear trend",   fe,            linear_trend),
        ("(6) + qz-quadratic trend",       fe,            quadratic_trend),
        ("(7) + channels contemp.",        fe,            quadratic_trend + channels_contemp),
        ("(8) + channels lagged",          fe,            quadratic_trend + channels_lagged),
    ]

    rows = []
    for label, fixed_effects, controls in specifications:
        result = run_specification(panel, fixed_effects, controls)
        rows.append({"col": label, **result})
        print(
            f"  {label:<34} beta={result['beta']:+.4f} "
            f"se={result['se']:.4f} boot-p={result['p_boot']:.3g} "
            f"N={result['n']:,} R2={result['r2']:.3f}"
        )

    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"wrote {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
