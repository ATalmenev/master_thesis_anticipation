"""Pressure-versus-time decomposition helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from minute_battery import build_minute_extended
from bootstrap_liquidation_flow import add_liq_pressure
from bartik_iv import demean_by

PROCESSED = ROOT / "data" / "processed"
MM_BASELINE = PROCESSED / "mm_classifier_refined_oct09_baseline.csv"
OUT = PROCESSED / "pressure_vs_time_summary.csv"

def add_time(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    mins = np.sort(p["minute_dt"].unique())
    idx = {m: i for i, m in enumerate(mins)}
    t = p["minute_dt"].map(idx).to_numpy().astype(float)
    t = (t - t.mean()) / t.std()
    p["t_x_qz"] = t * p["q_z"]
    p["t2_x_qz"] = (t ** 2) * p["q_z"]
    return p

def fit_one(work: pd.DataFrame, press: str, extra: list[str], fe: list[str]) -> dict:
    cols = ["log1p_alo", press, "log1p_alo_lag1"] + extra
    w = work.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["user"])
    dfd = demean_by(w, cols, fe)
    rhs = " + ".join([press, "log1p_alo_lag1"] + extra)
    m = smf.ols(f"log1p_alo ~ {rhs} - 1", data=dfd).fit(
        cov_type="cluster", cov_kwds={"groups": w["user"].values})
    return {"coef": float(m.params[press]), "se": float(m.bse[press]),
            "p": float(m.pvalues[press])}

def main() -> None:
    print("Does press*q_z survive a time*q_z control?")
    panel = add_time(add_liq_pressure(build_minute_extended()))
    mm = set(pd.read_csv(MM_BASELINE).query("mm_like_oct09_baseline==1")["user"].str.lower())
    fe = ["user_coin", "coin_minute"]
    presses = [("liq_pressure_x_qz", "cum(LIQ) monotone"),
               ("resolution_pressure_x_qz", "old LIQ+ADL monotone"),
               ("stock_pressure_x_qz", "signed LIQ-ADL (=position, hump)")]
    specs = [("baseline", []), ("+ time*qz (lin)", ["t_x_qz"]),
             ("+ time*qz (lin+quad)", ["t_x_qz", "t2_x_qz"])]

    rows = []
    for cohort, sub in [("all_LP", panel),
                        ("mm_46", panel[panel["user"].isin(mm)])]:
        print(f"--- {cohort} (n={sub['user'].nunique()}) ---")
        for pcol, plab in presses:
            line = f"  {plab:<34}"
            for slab, extra in specs:
                r = fit_one(sub, pcol, extra, fe)
                star = ("***" if r["p"] < 1e-3 else "**" if r["p"] < 1e-2
                        else "*" if r["p"] < 0.05 else "")
                rows.append({"cohort": cohort, "press": plab, "spec": slab, **r})
                line += f"  {slab}: {r['coef']:+.3f}{star:<3}(p={r['p']:.2g})"
            print(line)
        print()

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)}")
    print("\nRead: if press*q_z collapses to insignificance once time*q_z is added,"
          " its identification is just a time trend in the q_z-slope.")

if __name__ == "__main__":
    main()
