"""Signed-inventory variant of the pressure signal on the ALO outcome."""

from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from bartik_iv import prepare_panel, demean_by

PROCESSED = ROOT / "data" / "processed"
STOCK = PROCESSED / "hlp_inventory_pressure_stock_second.csv"
FIVESEC_PANEL = PROCESSED / "fivesec_user_coin_panel_20251010.csv"
OUT_TERMS = PROCESSED / "signed_inventory_pressure_alo_terms.csv"
OUT_SUMMARY = PROCESSED / "signed_inventory_pressure_alo_summary.json"

def fit(df: pd.DataFrame, sample: str, term: str, extra: list[str],
        fe: list[str], spec: str) -> dict:
    cols = ["log1p_alo", term] + extra
    work = df.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["user"])
    dfd = demean_by(work, cols, fe)
    formula = f"log1p_alo ~ {term}"
    if extra:
        formula += " + " + " + ".join(extra)
    formula += " - 1"
    model = smf.ols(formula, data=dfd).fit(
        cov_type="cluster", cov_kwds={"groups": work["user"].values}
    )
    return {
        "sample": sample,
        "spec": spec,
        "term": term,
        "coef": float(model.params[term]),
        "se_cluster_user": float(model.bse[term]),
        "p_value": float(model.pvalues[term]),
        "n_obs": int(model.nobs),
        "n_users": int(work["user"].nunique()),
    }

def stock_series(freq: str, time_col: str) -> pd.DataFrame:
    s = pd.read_csv(STOCK)
    s["second"] = pd.to_datetime(s["second"], utc=True)
    max_abs_stock = float(s["net_inventory_pressure"].abs().max())
    max_abs_resolution = float(s["hlp_cum_abs"].abs().max())
    s["stock_norm"] = s["net_inventory_pressure"] / max_abs_stock if max_abs_stock > 0 else 0.0
    s["resolution_norm"] = s["hlp_cum_abs"] / max_abs_resolution if max_abs_resolution > 0 else 0.0
    s[time_col] = s["second"].dt.floor(freq)
    g = (
        s.groupby(time_col, as_index=False)
        .agg(
            stock_norm=("stock_norm", "last"),
            resolution_norm=("resolution_norm", "max"),
            net_inventory_pressure=("net_inventory_pressure", "last"),
            hlp_cum_abs=("hlp_cum_abs", "max"),
        )
        .sort_values(time_col)
    )
    g["stock_norm_lag1"] = g["stock_norm"].shift(1).fillna(0.0)
    g["resolution_norm_lag1"] = g["resolution_norm"].shift(1).fillna(0.0)
    return g

def minute_panel() -> pd.DataFrame:
    sig = stock_series("min", "minute_dt")
    panel = prepare_panel().copy()
    panel = panel.merge(
        sig[["minute_dt", "stock_norm_lag1", "resolution_norm_lag1"]],
        on="minute_dt",
        how="left",
    )
    panel["stock_pressure_x_qz"] = panel["stock_norm_lag1"] * panel["q_z"]
    panel["resolution_pressure_x_qz"] = panel["resolution_norm_lag1"] * panel["q_z"]
    return panel

def fivesec_panel() -> pd.DataFrame:
    sig = stock_series("5s", "bucket")
    panel = pd.read_csv(FIVESEC_PANEL)
    panel["bucket"] = pd.to_datetime(panel["bucket"], utc=True)
    panel = panel.merge(
        sig[["bucket", "stock_norm_lag1", "resolution_norm_lag1"]],
        on="bucket",
        how="left",
    )
    panel["stock_pressure_x_qz"] = panel["stock_norm_lag1"] * panel["q_z"]
    panel["resolution_pressure_x_qz"] = panel["resolution_norm_lag1"] * panel["q_z"]
    return panel

def main() -> None:
    rows: list[dict] = []
    m = minute_panel()
    for term in ["stock_pressure_x_qz", "resolution_pressure_x_qz"]:
        rows.append(fit(m, "minute", term, ["log1p_alo_lag1"], ["user_coin", "coin_minute"], "fe_lag"))
        rows.append(fit(
            m,
            "minute",
            term,
            ["log1p_alo_lag1", "forced_flow_notional_min", "max_abs_mid_return_1s", "mean_spread_bps"],
            ["user_coin", "coin_minute"],
            "fe_lag_controls",
        ))

    s = fivesec_panel()
    for term in ["stock_pressure_x_qz", "resolution_pressure_x_qz"]:
        rows.append(fit(s, "5sec", term, ["log1p_alo_lag1"], ["user_coin", "coin_bucket"], "fe_lag"))
        rows.append(fit(
            s,
            "5sec",
            term,
            ["log1p_alo_lag1", "forced_x_qz", "vol_x_qz", "spread_x_qz"],
            ["user_coin", "coin_bucket"],
            "fe_lag_controls",
        ))

    terms = pd.DataFrame(rows)
    terms.to_csv(OUT_TERMS, index=False)
    summary = {
        "terms": rows,
        "note": "stock_pressure is LIQ acquisition minus ADL unwind, normalized by its absolute peak. resolution_pressure is the old cumulative HLP forced-flow resolution volume.",
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(terms.to_string(index=False))
    print(f"\nSaved {OUT_TERMS}")
    print(f"Saved {OUT_SUMMARY}")

if __name__ == "__main__":
    main()
