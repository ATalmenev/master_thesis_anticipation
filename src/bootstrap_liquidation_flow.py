"""Liquidation-flow variant of the block bootstrap-t regression."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from minute_battery import build_minute_extended
from bootstrap_utils import bootstrap_t

PROCESSED = ROOT / "data" / "processed"
STOCK = PROCESSED / "hlp_inventory_pressure_stock_second.csv"
MM_BASELINE = PROCESSED / "mm_classifier_refined_oct09_baseline.csv"
OUT = PROCESSED / "bootstrap_t_liq_summary.csv"

def add_liq_pressure(panel: pd.DataFrame) -> pd.DataFrame:
    s = pd.read_csv(STOCK)
    s["second"] = pd.to_datetime(s["second"], utc=True)
    s["cum_liq"] = s["liq_acquisition_notional"].cumsum()
    s["minute_dt"] = s["second"].dt.floor("min")
    g = s.groupby("minute_dt", as_index=False).agg(cum_liq=("cum_liq", "max"))
    mx = float(g["cum_liq"].max())
    g["liq_norm"] = g["cum_liq"] / mx if mx > 0 else 0.0
    g = g.sort_values("minute_dt")
    g["liq_norm_lag1"] = g["liq_norm"].shift(1)
    out = panel.merge(g[["minute_dt", "liq_norm_lag1"]], on="minute_dt", how="left")
    out["liq_norm_lag1"] = out["liq_norm_lag1"].fillna(0.0)
    out["liq_pressure_x_qz"] = out["liq_norm_lag1"] * out["q_z"]
    return out

def main() -> None:
    print("Bootstrap-t on pure-liquidation pressure cum(LIQ)")
    panel = add_liq_pressure(build_minute_extended())
    mm = set(pd.read_csv(MM_BASELINE).query("mm_like_oct09_baseline==1")["user"].str.lower())
    fe = ["user_coin", "coin_minute"]

    rows = []
    for cohort, sub in [("all_LP", panel),
                        ("mm_46", panel[panel["user"].isin(mm)])]:

        for xcol, lab in [("liq_pressure_x_qz", "LIQ only (monotone)"),
                         ("stock_pressure_x_qz", "signed LIQ-ADL (hump)"),
                         ("resolution_pressure_x_qz", "old LIQ+ADL (monotone)")]:
            if xcol not in sub.columns:
                continue
            r = bootstrap_t(sub, "log1p_alo", xcol, fe)
            rows.append({"cohort": cohort, "pressure": lab,
                         "beta_hat": r["beta_hat"], "t_obs": r["t_obs"],
                         "boot_t_p_two": r["boot_t_p_two"], "B": r["B"]})
            sig = ("0.001" if r["boot_t_p_two"] < 1e-3 else "0.01" if r["boot_t_p_two"] < 1e-2
                   else "0.05" if r["boot_t_p_two"] < 0.05 else "ns")
            print(f"[{cohort:<7}] {lab:<24} beta={r['beta_hat']:+.3f} "
                  f"t_obs={r['t_obs']:+.2f}  symmetric bootstrap-t p={r['boot_t_p_two']:.4g}  -> {sig}")
        print()

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
