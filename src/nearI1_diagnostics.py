"""Diagnostics for the non-standard limiting distribution under near-I(1)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from bartik_iv import prepare_panel
from null_tstat_montecarlo import (
    fast_cluster_t,
    make_demeaner,
    simulate_ar1,
)

PROCESSED = ROOT / "data" / "processed"
FIVESEC = PROCESSED / "fivesec_user_coin_panel_20251010.csv"
TERMS = PROCESSED / "signed_inventory_pressure_alo_terms.csv"
OUT = PROCESSED / "nonstandard_inference_summary.csv"
RHO_GRID = [0.95, 0.97, 0.99, 0.999, 1.0]
SEED = 42

def null_draws(panel: pd.DataFrame, fe: list[str], time_col: str,
               b_sims: int) -> tuple[float, dict[float, np.ndarray]]:
    code0 = pd.factorize(panel[fe[0]].to_numpy())[0]
    code1 = pd.factorize(panel[fe[1]].to_numpy())[0]
    demean = make_demeaner(code0, code1)
    ccode, uniq = pd.factorize(panel["user"].to_numpy())
    ncl = len(uniq)

    y = demean(panel["log1p_alo"].to_numpy().astype(float))
    x_obs = demean(panel["pressure_x_qz"].to_numpy().astype(float))
    t_obs = fast_cluster_t(x_obs, y, ccode, ncl)

    times = np.sort(panel[time_col].unique())
    T = len(times)
    tmap = {t: i for i, t in enumerate(times)}
    tidx = panel[time_col].map(tmap).to_numpy()
    qz = panel["q_z"].to_numpy().astype(float)
    rng = np.random.default_rng(SEED)

    draws: dict[float, np.ndarray] = {}
    for rho in RHO_GRID:
        ts = np.empty(b_sims)
        for b in range(b_sims):
            p = simulate_ar1(rho, T, rng)
            p = (p - p.min()) / (p.max() - p.min() + 1e-12)
            plag = np.concatenate(([0.0], p[:-1]))
            xres = demean(plag[tidx] * qz)
            ts[b] = fast_cluster_t(xres, y, ccode, ncl)
        draws[rho] = ts
    return t_obs, draws

def p_nonstandard(t_obs: float, draws: dict[float, np.ndarray]) -> dict:
    per_rho = {rho: float(np.mean(np.abs(ts) >= abs(t_obs)))
               for rho, ts in draws.items()}
    worst_p = max(per_rho.values())
    ur = np.abs(draws[1.0])
    return {
        "p_worst": worst_p,
        "crit95_ur": float(np.quantile(ur, 0.95)),
        "crit975_ur": float(np.quantile(ur, 0.975)),
        "crit99_ur": float(np.quantile(ur, 0.99)),
    }

def ips_W(panel: pd.DataFrame, time_col: str, unit: str = "user_coin") -> dict:
    from statsmodels.tsa.stattools import adfuller
    E_T, VAR_T = -1.533, 0.726
    ts = []
    for _, g in panel.sort_values(time_col).groupby(unit):
        s = g["log1p_alo"].to_numpy().astype(float)
        if s.size < 12 or np.var(s) < 1e-10:
            continue
        try:
            ts.append(adfuller(s, maxlag=0, regression="c", autolag=None)[0])
        except Exception:
            continue
    ts = np.asarray(ts)
    n = ts.size
    W = np.sqrt(n) * (ts.mean() - E_T) / np.sqrt(VAR_T)
    return {"n_series": int(n), "tbar": float(ts.mean()), "W": float(W),
            "p": float(stats.norm.cdf(W))}

def main() -> None:
    print("Non-standard (near-integration) inference for ALL regressions")
    print("Loading minute panel...")
    mp = prepare_panel()
    mp["tkey"] = mp["minute_dt"]
    print("Loading 5-sec panel...")
    fp = pd.read_csv(FIVESEC)
    fp["user"] = fp["user"].astype(str).str.lower()
    fp["tkey"] = pd.to_datetime(fp["bucket"], utc=True)

    terms = pd.read_csv(TERMS)
    terms["t_obs"] = terms["coef"] / terms["se_cluster_user"]

    panels = {
        "minute": (mp, ["user_coin", "coin_minute"], 999),
        "5sec": (fp, ["user_coin", "coin_bucket"], 299),
    }

    rows = []
    for gran, (panel, fe, b) in panels.items():
        print(f"\n[{gran}] simulating null t-distribution (B={b})...")
        t_old, draws = null_draws(panel, fe, "tkey", b)
        ndp = p_nonstandard(t_old, draws)
        print(f"  unit-root (rho=1) crit: 95%={ndp['crit95_ur']:.2f}  "
              f"97.5%={ndp['crit975_ur']:.2f}  99%={ndp['crit99_ur']:.2f}")
        ips = ips_W(panel, "tkey")
        print(f"  IPS per-LP ALO: N={ips['n_series']} series  tbar={ips['tbar']:.3f}  "
              f"W={ips['W']:.2f}  p={ips['p']:.2e}  "
              f"-> outcome {'STATIONARY' if ips['p'] < 0.01 else 'unit root'}")

        pn = p_nonstandard(t_old, draws)
        rows.append({"gran": gran, "spec": "old_press single-reg (no lag)",
                     "t_obs": t_old, "p_normal": 2 * stats.norm.sf(abs(t_old)),
                     "crit975_ur": pn["crit975_ur"], "p_nonstd_worst": pn["p_worst"],
                     "clears_5pct": abs(t_old) > pn["crit975_ur"]})

        sub = terms[terms["sample"] == gran]
        for _, r in sub.iterrows():
            t = float(r["t_obs"])
            pn = p_nonstandard(t, draws)
            rows.append({"gran": gran,
                         "spec": f"{r['term']} [{r['spec']}]",
                         "t_obs": t, "p_normal": 2 * stats.norm.sf(abs(t)),
                         "crit975_ur": pn["crit975_ur"],
                         "p_nonstd_worst": pn["p_worst"],
                         "clears_5pct": abs(t) > pn["crit975_ur"]})

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("Verdict table (vs non-standard 97.5% critical value)")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(out.to_string(index=False,
              formatters={"t_obs": "{:+.2f}".format,
                          "p_normal": "{:.1e}".format,
                          "crit975_ur": "{:.2f}".format,
                          "p_nonstd_worst": "{:.3f}".format}))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nRead: every |t| is compared to the simulated near-I(1) critical "
          "value, NOT 1.96.  'clears_5pct'=False means plain OLS is NOT "
          "significant under the free-random-walk null -- expected for a near-"
          "integrated regressor.  The effect is rescued by IVX (standard), the "
          "stationary per-LP outcome (IPS), and the block bootstrap.")

if __name__ == "__main__":
    main()
