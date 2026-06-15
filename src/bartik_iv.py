"""Formal 2SLS Bartik-style instrument estimator."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from linearmodels.iv import IV2SLS

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
BINANCE = ROOT / "data/raw/binance_oct10"
PANEL = PROCESSED / "user_minute_full_panel_20251010.csv"
HLP = PROCESSED / "iter14_hlp_inventory_per_second.csv"
CLS = PROCESSED / "mm_classifier_refined.csv"
OUT = PROCESSED / "model2_formal_2sls_summary.csv"

def load_binance_returns() -> pd.DataFrame:
    rows = []
    for coin in ["BTC", "ETH", "SOL"]:
        zip_path = BINANCE / f"{coin}USDT-klines-1m-2025-10-10.zip"
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                df = pd.read_csv(f)
            df.columns = ["ts", "o", "h", "l", "c", "v",
                           "ct", "qv", "nt", "tbv", "tqv", "ignore"][:len(df.columns)]
            df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
            df = df.dropna(subset=["ts"])
        df["minute_dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df["price"] = df["c"].astype(float)
        df = df.sort_values("minute_dt")
        df["return_1m"] = np.log(df["price"] / df["price"].shift(1))
        df["abs_return_1m"] = df["return_1m"].abs()
        df["coin"] = coin
        rows.append(df[["minute_dt", "coin", "price", "return_1m",
                         "abs_return_1m"]])
    return pd.concat(rows, ignore_index=True)

def build_hlp_pressure() -> pd.DataFrame:
    hlp = pd.read_csv(HLP)
    hlp["second"] = pd.to_datetime(hlp["second"], utc=True)
    hlp["minute_dt"] = hlp["second"].dt.floor("min")
    per_min = hlp.groupby("minute_dt", as_index=False).agg(
        hlp_abs_inventory=("hlp_cum_abs", "max"))
    max_p = float(per_min["hlp_abs_inventory"].max())
    per_min["hlp_pressure"] = (per_min["hlp_abs_inventory"] / max_p
                                 if max_p > 0 else 0)
    per_min = per_min.sort_values("minute_dt")
    per_min["hlp_pressure_lag1"] = per_min["hlp_pressure"].shift(1).fillna(0)
    return per_min

def prepare_panel() -> pd.DataFrame:
    print("Loading Binance + HLP pressure...")
    binance = load_binance_returns()
    T_LO = pd.Timestamp("2025-10-10 20:55:00+00:00")
    T_HI = pd.Timestamp("2025-10-10 21:45:00+00:00")
    binance = binance[(binance["minute_dt"] >= T_LO)
                       & (binance["minute_dt"] <= T_HI)]
    binance["abs_return_lag1"] = binance.groupby("coin")["abs_return_1m"].shift(1)

    panel = pd.read_csv(PANEL)
    panel["user"] = panel["user"].astype(str).str.lower()
    panel["minute_dt"] = pd.to_datetime(panel["minute"], utc=True)
    pressure = build_hlp_pressure()
    panel = panel.merge(pressure, on="minute_dt", how="left")
    panel["hlp_pressure_lag1"] = pd.to_numeric(
        panel["hlp_pressure_lag1"], errors="coerce").fillna(0)
    panel = panel[panel["coin"].isin(["BTC", "ETH", "SOL"])].copy()
    panel = panel.merge(binance[["minute_dt", "coin", "abs_return_lag1"]],
                         on=["minute_dt", "coin"], how="left")

    panel["log1p_actions"] = np.log1p(pd.to_numeric(
        panel["actions"], errors="coerce").fillna(0))
    panel["log1p_alo"] = np.log1p(pd.to_numeric(
        panel["alo_order_count"], errors="coerce").fillna(0))
    panel["active_any"] = panel["actions"].gt(0).astype(int)

    panel["log1p_q"] = pd.to_numeric(panel["log1p_q_pre_coin"],
                                       errors="coerce").fillna(0.0)
    sd = float(panel["log1p_q"].std(ddof=0))
    panel["q_z"] = ((panel["log1p_q"] - float(panel["log1p_q"].mean())) / sd
                     if sd > 0 else 0.0)
    panel["pressure_x_qz"] = panel["hlp_pressure_lag1"] * panel["q_z"]

    panel["binance_abs_ret_lag1"] = pd.to_numeric(
        panel["abs_return_lag1"], errors="coerce").fillna(0)
    panel["instrument_z_x_qz"] = (panel["binance_abs_ret_lag1"]
                                     * panel["q_z"])

    panel["user_coin"] = panel["user"] + "_" + panel["coin"].astype(str)
    panel["coin_minute"] = (panel["coin"].astype(str) + "_"
                              + panel["minute"].astype(str))
    panel = panel.sort_values(["user", "coin", "minute_dt"])
    for outcome in ["active_any", "log1p_actions", "log1p_alo"]:
        panel[f"{outcome}_lag1"] = panel.groupby("user_coin")[outcome].shift(1)

    panel = panel.replace([np.inf, -np.inf], np.nan).dropna()
    return panel

def demean_by(df: pd.DataFrame, cols: list[str],
              group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        for gc in group_cols:
            out[col] = out[col] - out.groupby(gc)[col].transform("mean")
    return out

def run_2sls_one_outcome(df: pd.DataFrame, outcome: str,
                          cohort_label: str) -> dict:
    cols_to_demean = [outcome, "pressure_x_qz", "instrument_z_x_qz",
                       f"{outcome}_lag1"]
    dfd = demean_by(df, cols_to_demean, ["user_coin", "coin_minute"])

    formula = (f"{outcome} ~ 1 + {outcome}_lag1 "
               f"+ [pressure_x_qz ~ instrument_z_x_qz]")
    try:
        model = IV2SLS.from_formula(formula, data=dfd)
        res = model.fit(cov_type="clustered",
                         clusters=df["user"].values)
        first_stage = res.first_stage
        fs_diag = first_stage.diagnostics
        partial_f = float(fs_diag.loc["pressure_x_qz", "f.stat"])
        partial_f_p = float(fs_diag.loc["pressure_x_qz", "f.pval"])
        partial_rsq = float(fs_diag.loc["pressure_x_qz", "partial.rsquared"])

        try:
            wh_test = res.wooldridge_regression
            wh_stat = float(wh_test.stat)
            wh_p = float(wh_test.pval)
        except Exception:
            wh_stat = float("nan")
            wh_p = float("nan")

        try:
            db_test = res.durbin()
            db_stat = float(db_test.stat)
            db_p = float(db_test.pval)
        except Exception:
            db_stat = float("nan")
            db_p = float("nan")

        beta_iv = float(res.params["pressure_x_qz"])
        se_iv = float(res.std_errors["pressure_x_qz"])
        p_iv = float(res.pvalues["pressure_x_qz"])

        ols_formula = (f"{outcome} ~ pressure_x_qz + {outcome}_lag1")
        m_ols = smf.ols(ols_formula, data=dfd).fit(
            cov_type="cluster", cov_kwds={"groups": df["user"].values})
        beta_ols = float(m_ols.params["pressure_x_qz"])
        se_ols = float(m_ols.bse["pressure_x_qz"])
        p_ols = float(m_ols.pvalues["pressure_x_qz"])

        return {
            "outcome": outcome,
            "cohort": cohort_label,
            "n_obs": int(res.nobs),
            "beta_2sls": beta_iv,
            "se_2sls": se_iv,
            "p_2sls": p_iv,
            "beta_ols_naive": beta_ols,
            "se_ols_naive": se_ols,
            "p_ols_naive": p_ols,
            "first_stage_partial_F": partial_f,
            "first_stage_partial_F_p": partial_f_p,
            "first_stage_partial_R2": partial_rsq,
            "wooldridge_endog_stat": wh_stat,
            "wooldridge_endog_p": wh_p,
            "durbin_endog_stat": db_stat,
            "durbin_endog_p": db_p,
            "weak_instrument": partial_f < 10.0,
        }
    except Exception as e:
        print(f"  ERR ({outcome}, {cohort_label}): {str(e)[:120]}")
        return {
            "outcome": outcome,
            "cohort": cohort_label,
            "error": str(e)[:200],
        }

def main() -> None:
    print("Formal 2SLS IV для pressure × q_z hypothesis")

    panel = prepare_panel()
    print(f"Panel: {len(panel):,} rows | {panel['user'].nunique()} users\n")

    cls = pd.read_csv(CLS, usecols=["user", "mm_like_refined"])
    cls["user"] = cls["user"].astype(str).str.lower()
    panel = panel.merge(cls, on="user", how="left")
    panel["mm_like_refined"] = panel["mm_like_refined"].fillna(0).astype(int)

    cohorts = {
        "all": panel,
        "mm_like": panel[panel["mm_like_refined"] == 1],
    }

    results = []
    for cohort_name, df_coh in cohorts.items():
        n_users = df_coh["user"].nunique()
        print(f"\n{'=' * 75}\nCOHORT: {cohort_name} (N_users={n_users}, "
              f"N_obs={len(df_coh):,})\n{'=' * 75}")
        for outcome in ["log1p_actions", "log1p_alo", "active_any"]:
            print(f"\n--- {outcome} ---")
            res = run_2sls_one_outcome(df_coh, outcome, cohort_name)
            results.append(res)
            if "error" in res:
                continue
            sig_iv = ("***" if res["p_2sls"] < 0.001
                      else "**" if res["p_2sls"] < 0.01
                      else "*" if res["p_2sls"] < 0.05 else "")
            sig_ols = ("***" if res["p_ols_naive"] < 0.001
                       else "**" if res["p_ols_naive"] < 0.01
                       else "*" if res["p_ols_naive"] < 0.05 else "")
            weak = " [WEAK]" if res["weak_instrument"] else ""
            print(f"  2SLS β  = {res['beta_2sls']:+.4f} "
                  f"(SE={res['se_2sls']:.4f}, p={res['p_2sls']:.3g}) {sig_iv}")
            print(f"  OLS β   = {res['beta_ols_naive']:+.4f} "
                  f"(SE={res['se_ols_naive']:.4f}, "
                  f"p={res['p_ols_naive']:.3g}) {sig_ols}")
            print(f"  First-stage partial F = {res['first_stage_partial_F']:.1f} "
                  f"(p={res['first_stage_partial_F_p']:.3g}){weak}")
            print(f"  Wooldridge endog test: stat={res['wooldridge_endog_stat']:.2f} "
                  f"p={res['wooldridge_endog_p']:.3g}")
            print(f"  Durbin endog test:    stat={res['durbin_endog_stat']:.2f} "
                  f"p={res['durbin_endog_p']:.3g}")
            print(f"  First-stage partial R² = {res['first_stage_partial_R2']:.4f}")
            ratio = (res["beta_2sls"] / res["beta_ols_naive"]
                     if res["beta_ols_naive"] != 0 else float("nan"))
            print(f"  Ratio 2SLS/OLS = {ratio:+.2f}×")

    pd.DataFrame(results).to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
