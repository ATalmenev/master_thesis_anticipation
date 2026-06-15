"""HAC and cluster-robust standard errors for the ALO regression."""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bootstrap_utils import iter_demean, slope_se
from pressure_signal import (
    PANEL_FILE, build_per_coin_pressure, load_sub_to_entity)
from cohort import aggregate_to_wallet, load_tier_cohort

L = 8

def _resid(v, Z):
    return v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]

def main():
    s2e = load_sub_to_entity()
    ti = load_tier_cohort()
    tier_w = {w for w, i in ti.items() if i["tier"] >= 1}
    per_coin = build_per_coin_pressure()
    p = pd.read_csv(PANEL_FILE)
    w = aggregate_to_wallet(p, s2e, per_coin)
    w = w[w["wallet_id"].isin(tier_w)].copy()
    need = ["log1p_alo", "press_x_qz", "log1p_alo_lag1", "t_x_qz", "user"]
    w = w.replace([np.inf, -np.inf], np.nan).dropna(subset=need).reset_index(drop=True)

    y = w["log1p_alo"].to_numpy(float)
    x = w["press_x_qz"].to_numpy(float)
    c0 = pd.factorize(w["user_coin"].to_numpy())[0]
    c1 = pd.factorize(w["coin_minute"].to_numpy())[0]
    n0, n1 = c0.max() + 1, c1.max() + 1
    yd = iter_demean(y, c0, c1, n0, n1)
    xd = iter_demean(x, c0, c1, n0, n1)
    ctrl = np.column_stack([iter_demean(w[c].to_numpy(float), c0, c1, n0, n1)
                            for c in ["log1p_alo_lag1", "t_x_qz"]])
    xr, yr = _resid(xd, ctrl), _resid(yd, ctrl)

    beta = float(xr @ yr / (xr @ xr))
    e = yr - beta * xr
    sxx = float(xr @ xr)
    n = len(xr)

    ucl, uniq = pd.factorize(w["user"].to_numpy())
    ncl = len(uniq)
    _, se_cl = slope_se(xr, yr, ucl, ncl)

    tcode, tuniq = pd.factorize(w["coin_minute"].to_numpy()); T = len(tuniq)
    h = xr * e
    h_t = np.bincount(tcode, weights=h, minlength=T)

    def nw_lrv(series, bw):
        s = series - 0.0
        g0 = float(s @ s)
        acc = g0
        for j in range(1, bw + 1):
            wj = 1.0 - j / (bw + 1)
            gj = float(s[j:] @ s[:-j])
            acc += 2.0 * wj * gj
        return acc

    S_dk = nw_lrv(h_t, L)
    se_dk = np.sqrt(S_dk) / sxx

    order = np.argsort(tcode, kind="stable")
    S_nw = nw_lrv(h[order], L)
    se_nw = np.sqrt(S_nw) / sxx

    def p2(se): return float(2 * stats.norm.sf(abs(beta / se)))

    print(f"n={n:,}  T(coin-bucket)={T}  clusters(master)={ncl}")
    print(f"beta = {beta:+.4f}")
    print(f"(a) cluster-robust (master)   SE={se_cl:.4f}  t={beta/se_cl:+.2f}  p={p2(se_cl):.2e}")
    print(f"(b) Driscoll-Kraay HAC (L={L}) SE={se_dk:.4f}  t={beta/se_dk:+.2f}  p={p2(se_dk):.2e}")
    print(f"(c) pooled Newey-West (L={L})  SE={se_nw:.4f}  t={beta/se_nw:+.2f}  p={p2(se_nw):.2e}")

if __name__ == "__main__":
    main()
