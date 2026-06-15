"""Monte Carlo of the null t-statistic distribution under the cascade DGP."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from bartik_iv import prepare_panel, demean_by

PROCESSED = ROOT / "data" / "processed"
FIVESEC_PANEL = PROCESSED / "fivesec_user_coin_panel_20251010.csv"
OUT = PROCESSED / "simulated_null_tstat_summary.csv"
RHO_GRID = [0.95, 0.97, 0.99, 0.999, 1.0]
B = 999
SEED = 42

def fast_cluster_t(x: np.ndarray, y: np.ndarray, ccode: np.ndarray,
                   ncl: int) -> float:
    sxx = float(x @ x)
    if sxx <= 0:
        return 0.0
    beta = float(x @ y) / sxx
    s = x * (y - beta * x)
    g = np.bincount(ccode, weights=s, minlength=ncl)
    meat = float(g @ g)
    var = meat / (sxx * sxx)
    se = np.sqrt(var) if var > 0 else np.inf
    return beta / se if se > 0 else 0.0

def make_demeaner(code0: np.ndarray, code1: np.ndarray):
    n0, n1 = code0.max() + 1, code1.max() + 1
    cnt0 = np.bincount(code0, minlength=n0)
    cnt1 = np.bincount(code1, minlength=n1)

    def demean(v: np.ndarray) -> np.ndarray:
        v = v - (np.bincount(code0, weights=v, minlength=n0) / cnt0)[code0]
        v = v - (np.bincount(code1, weights=v, minlength=n1) / cnt1)[code1]
        return v
    return demean

def simulate_ar1(rho: float, T: int, rng: np.random.Generator) -> np.ndarray:
    e = rng.standard_normal(T)
    x = np.empty(T)
    x[0] = e[0]
    for t in range(1, T):
        x[t] = rho * x[t - 1] + e[t]
    return x

def run_panel(panel: pd.DataFrame, fe: list[str], time_col: str,
              gran: str, b_sims: int) -> list[dict]:
    df = panel.copy()
    code0 = pd.factorize(df[fe[0]].to_numpy())[0]
    code1 = pd.factorize(df[fe[1]].to_numpy())[0]
    demean = make_demeaner(code0, code1)
    ccode, ncl = pd.factorize(df["user"].to_numpy())
    ncl = len(ncl)

    y = demean(df["log1p_alo"].to_numpy().astype(float))
    x_obs = demean(df["pressure_x_qz"].to_numpy().astype(float))
    t_obs = fast_cluster_t(x_obs, y, ccode, ncl)

    times = np.sort(df[time_col].unique())
    T = len(times)
    tmap = {t: i for i, t in enumerate(times)}
    tidx = df[time_col].map(tmap).to_numpy()
    qz = df["q_z"].to_numpy().astype(float)
    rng = np.random.default_rng(SEED)

    rows = []
    for rho in RHO_GRID:
        t_star = np.empty(b_sims)
        for b in range(b_sims):
            p = simulate_ar1(rho, T, rng)
            p = (p - p.min()) / (p.max() - p.min() + 1e-12)
            plag = np.concatenate(([0.0], p[:-1]))
            xres = demean(plag[tidx] * qz)
            t_star[b] = fast_cluster_t(xres, y, ccode, ncl)
        crit = np.percentile(np.abs(t_star), [95, 97.5, 99])
        p_sim = float(np.mean(np.abs(t_star) >= abs(t_obs)))
        rows.append({"gran": gran, "rho": rho, "t_obs": t_obs,
                     "crit95": crit[0], "crit97.5": crit[1], "crit99": crit[2],
                     "p_sim": p_sim, "B": b_sims})
    return rows

def main() -> None:
    print("Simulated null distribution of t (Monte-Carlo crit values)")
    print("Loading minute panel...")
    mp = prepare_panel()
    mp["tkey"] = mp["minute_dt"]
    print("Loading 5-sec panel...")
    fp = pd.read_csv(FIVESEC_PANEL)
    fp["user"] = fp["user"].astype(str).str.lower()
    fp["tkey"] = pd.to_datetime(fp["bucket"], utc=True)

    results = []
    for gran, panel, fe, b_sims in [
        ("minute", mp, ["user_coin", "coin_minute"], 999),
        ("5sec", fp, ["user_coin", "coin_bucket"], 299),
    ]:
        print(f"\n{gran}: simulating null t over rho grid (B={b_sims})...")
        rows = run_panel(panel, fe, "tkey", gran, b_sims)
        results += rows
        t_obs = rows[0]["t_obs"]
        print(f"  observed |t| = {abs(t_obs):.2f}")
        print(f"  {'rho':>7}{'sim crit(97.5%)':>18}{'sim p':>12}")
        for r in rows:
            print(f"  {r['rho']:>7}{r['crit97.5']:>18.3f}{r['p_sim']:>12.3g}")
        worst = max(r["p_sim"] for r in rows)
        print(f"  worst-case p over rho grid (sim-Bonferroni): {worst:.3g}")

    pd.DataFrame(results).to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")
    print("\nRead: if observed |t| far exceeds the simulated 97.5% critical "
          "value at EVERY rho, the result is not a near-integration artifact.")

if __name__ == "__main__":
    main()
