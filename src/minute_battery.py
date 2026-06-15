"""Extended battery of minute-level robustness checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from signed_inventory_alo import stock_series, fit
from nearI1_diagnostics import ips_W

PROCESSED = ROOT / "data" / "processed"
EXT5 = PROCESSED / "fivesec_user_coin_panel_extended_20251010.csv"
MM_BASELINE = PROCESSED / "mm_classifier_refined_oct09_baseline.csv"
OUT = PROCESSED / "minute_extended_battery_summary.csv"
B = 20000
SEED = 42

def build_minute_extended() -> pd.DataFrame:
    p = pd.read_csv(EXT5)
    p["user"] = p["user"].astype(str).str.lower()
    p["minute_dt"] = pd.to_datetime(p["bucket"], utc=True).dt.floor("min")
    agg = (p.groupby(["user", "coin", "minute_dt"], as_index=False)
             .agg(alo_order_count=("alo_order_count", "sum"),
                  actions=("actions", "sum"),
                  log1p_q_pre_coin=("log1p_q_pre_coin", "first")))
    agg["log1p_alo"] = np.log1p(agg["alo_order_count"])
    sd = float(agg["log1p_q_pre_coin"].std(ddof=0))
    agg["q_z"] = (agg["log1p_q_pre_coin"] - agg["log1p_q_pre_coin"].mean()) / sd

    sig = stock_series("min", "minute_dt")
    agg = agg.merge(sig[["minute_dt", "stock_norm_lag1", "resolution_norm_lag1"]],
                    on="minute_dt", how="left")
    for c in ["stock_norm_lag1", "resolution_norm_lag1"]:
        agg[c] = agg[c].fillna(0.0)
    agg["stock_pressure_x_qz"] = agg["stock_norm_lag1"] * agg["q_z"]
    agg["resolution_pressure_x_qz"] = agg["resolution_norm_lag1"] * agg["q_z"]
    agg["user_coin"] = agg["user"] + "_" + agg["coin"].astype(str)
    agg["coin_minute"] = agg["coin"].astype(str) + "_" + agg["minute_dt"].astype(str)
    agg = agg.sort_values(["user_coin", "minute_dt"])
    agg["log1p_alo_lag1"] = agg.groupby("user_coin")["log1p_alo"].shift(1)
    agg["tkey"] = agg["minute_dt"]
    return agg.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["log1p_alo", "q_z", "stock_pressure_x_qz", "log1p_alo_lag1"])

def _dm(v, code, n):
    s = np.bincount(code, weights=v, minlength=n)
    c = np.bincount(code, minlength=n)
    return v - (s / np.where(c > 0, c, 1))[code]

def proper_bootstrap(df, xcol, fe, L, b=B, seed=SEED) -> dict:
    uc = pd.factorize(df[fe[0]].to_numpy())[0]
    ct = pd.factorize(df[fe[1]].to_numpy())[0]
    nuc, nct = uc.max() + 1, ct.max() + 1
    xr = df[xcol].to_numpy(float)
    yr = df["log1p_alo"].to_numpy(float)

    def beta(rows):
        x = _dm(_dm(xr[rows], uc[rows], nuc), ct[rows], nct)
        y = _dm(_dm(yr[rows], uc[rows], nuc), ct[rows], nct)
        sxx = float(x @ x)
        return float(x @ y) / sxx if sxx > 0 else np.nan

    bhat = beta(np.arange(len(df)))
    tcode = pd.factorize(df["tkey"].to_numpy())[0]
    T = tcode.max() + 1
    trows = [np.where(tcode == i)[0] for i in range(T)]
    nb = int(np.ceil(T / L))
    ms = max(T - L, 1)
    rng = np.random.default_rng(seed)
    bs = np.empty(b)
    for k in range(b):
        seq = np.concatenate([np.arange(s, s + L) for s in rng.integers(0, ms, nb)])[:T]
        seq = seq[seq < T]
        bs[k] = beta(np.concatenate([trows[t] for t in seq]))
    bs = bs[np.isfinite(bs)]
    n = bs.size
    n_ge0 = int((bs >= 0).sum())

    p_two = 2.0 * (min(n_ge0, n - n_ge0) + 1) / (n + 1)
    ci = {lvl: np.percentile(bs, [a, 100 - a])
          for lvl, a in [(95, 2.5), (99, 0.5), (99.9, 0.05)]}
    return {"beta_hat": bhat, "n_ge0": n_ge0, "p_two": p_two,
            "ci95": (float(ci[95][0]), float(ci[95][1])),
            "ci99": (float(ci[99][0]), float(ci[99][1])),
            "ci999": (float(ci[99.9][0]), float(ci[99.9][1])),
            "B": int(n)}

def main() -> None:
    print("MINUTE-EXTENDED battery (20:00-21:45)")
    panel = build_minute_extended()
    print(f"minute-extended panel: {panel['user'].nunique()} users, "
          f"{panel['minute_dt'].nunique()} minutes "
          f"({panel['minute_dt'].min()} .. {panel['minute_dt'].max()}), "
          f"{len(panel):,} obs\n")
    mmset = set(pd.read_csv(MM_BASELINE)
                .query("mm_like_oct09_baseline == 1")["user"].str.lower())
    fe = ["user_coin", "coin_minute"]

    rows = []
    for cohort, sub in [("all_LP", panel),
                        ("mm_oct09_baseline", panel[panel["user"].isin(mmset)])]:
        n = sub["user"].nunique()
        r = fit(sub.copy(), "minute_ext", "stock_pressure_x_qz",
                ["log1p_alo_lag1"], fe, "fe_lag")
        b, se = r["coef"], r["se_cluster_user"]
        ips = ips_W(sub, "tkey")
        bb = proper_bootstrap(sub.reset_index(drop=True), "stock_pressure_x_qz",
                              fe, L=8)
        lvl = ("0.001" if bb["ci999"][1] < 0 or bb["ci999"][0] > 0
               else "0.01" if bb["ci99"][1] < 0 or bb["ci99"][0] > 0
               else "0.05" if bb["ci95"][1] < 0 or bb["ci95"][0] > 0
               else "ns")
        print(f"[{cohort:<18}] n={n:>3}  minutes={sub['minute_dt'].nunique()}")
        print(f"  signed beta={b:+.4f}  se={se:.4f}  t={b/se:+.2f}  p={r['p_value']:.2e}")
        print(f"  IPS: N={ips['n_series']} tbar={ips['tbar']:+.3f} W={ips['W']:+.2f} "
              f"p={ips['p']:.2e} -> {'STATIONARY' if ips['p']<0.01 else 'unit root'}")
        print(f"  PROPER block boot (re-est FE, B={bb['B']}): beta_hat={bb['beta_hat']:+.3f}")
        print(f"    #(beta*>=0)={bb['n_ge0']}/{bb['B']}  boot p(2-sided)={bb['p_two']:.2g}")
        print(f"    95% CI=[{bb['ci95'][0]:+.3f},{bb['ci95'][1]:+.3f}]  "
              f"99% CI=[{bb['ci99'][0]:+.3f},{bb['ci99'][1]:+.3f}]  "
              f"99.9% CI=[{bb['ci999'][0]:+.3f},{bb['ci999'][1]:+.3f}]")
        print(f"    -> significant at level {lvl}")
        rows.append({"cohort": cohort, "n_users": n,
                     "minutes": int(sub["minute_dt"].nunique()),
                     "beta": b, "se": se, "t": b / se, "p_normal": r["p_value"],
                     "ips_W": ips["W"], "ips_p": ips["p"],
                     "boot_beta": bb["beta_hat"], "boot_n_ge0": bb["n_ge0"],
                     "boot_p_two": bb["p_two"],
                     "boot_ci95_hi": bb["ci95"][1], "boot_ci99_hi": bb["ci99"][1],
                     "boot_ci999_hi": bb["ci999"][1], "boot_sig_level": lvl})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nwrote {OUT.relative_to(ROOT)}")

if __name__ == "__main__":
    main()
