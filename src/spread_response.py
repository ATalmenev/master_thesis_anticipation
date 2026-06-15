"""Wallet-level bid-ask spread response to HLP pressure."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from inference import run
from pressure_signal import (
    PANEL_FILE, build_per_coin_pressure, load_sub_to_entity,
)
from cohort import load_tier_cohort

SPREAD = ROOT / "data" / "processed" / "wallet_spread_panel_2100_2200_20251010.csv"
OUT = ROOT / "data" / "processed" / "wallet_spread_on_pressure_summary.csv"

import os
CLUSTER_ON = os.environ.get("CLUSTER_ON", "master")

def build_qz_from_panel(s2e: dict[str, str]) -> pd.DataFrame:
    p = pd.read_csv(PANEL_FILE, usecols=["user", "coin", "q_pre_coin"])
    p["user"] = p["user"].astype(str).str.lower()
    p = p[p["user"].map(s2e).notna()].copy()
    pre = (p.groupby(["user", "coin"])["q_pre_coin"].first()
             .abs().reset_index(name="w_m_c_pre"))
    pre["q_z"] = (pre["w_m_c_pre"] - pre["w_m_c_pre"].mean()) / pre["w_m_c_pre"].std()
    return pre[["user", "coin", "q_z"]]

def build_spread_panel(s2e: dict[str, str]) -> pd.DataFrame:
    sp = pd.read_csv(SPREAD)
    sp["user"] = sp["user"].astype(str).str.lower()
    sp["master"] = sp["user"].map(s2e)
    sp = sp[sp["master"].notna()].copy()
    sp["bucket_dt"] = pd.to_datetime(sp["bucket"], utc=True).dt.floor("5s")
    sp["log1p_dist"] = np.log1p(sp["mean_distance_bps"].clip(lower=0))

    qz = build_qz_from_panel(s2e)
    sp = sp.merge(qz, on=["user", "coin"], how="left")
    sp = sp[sp["q_z"].notna()].copy()

    pcp = build_per_coin_pressure()
    pcp["bucket_dt"] = pd.to_datetime(pcp["bucket"], utc=True)
    sp = sp.merge(pcp[["coin", "bucket_dt", "press_c"]],
                  on=["coin", "bucket_dt"], how="left")
    sp["press_c"] = sp["press_c"].fillna(0.0)
    sp["press_x_qz"] = sp["press_c"] * sp["q_z"]

    sp = sp.sort_values(["user", "coin", "bucket_dt"]).reset_index(drop=True)
    sp["log1p_alo_lag1"] = (sp.groupby(["user", "coin"])["log1p_dist"]
                              .shift(1).fillna(0))
    t0 = sp["bucket_dt"].min()
    sp["t"] = (sp["bucket_dt"] - t0).dt.total_seconds() / 60.0
    sp["t2"] = sp["t"] ** 2
    sp["t_x_qz"] = sp["t"] * sp["q_z"]
    sp["t2_x_qz"] = sp["t2"] * sp["q_z"]

    sp["user_coin"] = sp["user"] + "_" + sp["coin"].astype(str)
    sp["coin_minute"] = (sp["coin"].astype(str) + "_"
                         + sp["bucket_dt"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    sp["tkey"] = sp["coin_minute"]

    sp["log1p_alo"] = sp["log1p_dist"].astype(float)

    sp["wallet_id"] = sp["user"]
    if CLUSTER_ON == "master":
        sp["user"] = sp["master"]
    return sp

def main() -> None:
    print(f"CLUSTER_ON = {CLUSTER_ON}")
    s2e = load_sub_to_entity()
    tier_info = load_tier_cohort()
    tier_w = {w for w, i in tier_info.items() if i["tier"] >= 1}
    mm_w = {w for w, i in tier_info.items() if i["mm_c4c5"] == 1}
    non_w = {w for w, i in tier_info.items()
             if i["tier"] >= 1 and i["mm_c4c5"] == 0}
    print(f"  tier-1+: {len(tier_w)}, mm(C4&C5): {len(mm_w)}, non_mm: {len(non_w)}")

    sp = build_spread_panel(s2e)
    print(f"  spread panel: {len(sp):,} rows, {sp['wallet_id'].nunique()} wallets, "
          f"{sp['user'].nunique()} clusters, "
          f"mean dist {sp['mean_distance_bps'].mean():.2f} bps")

    fe = ["user_coin", "coin_minute"]
    xcol = "press_x_qz"
    ctrls = ["log1p_alo_lag1", "t_x_qz", "t2_x_qz"]
    rows = []
    print(f"PRICE margin: log1p(mean_distance_bps) on press_x_qz, "
          f"FE=(wallet-coin, coin-bucket), cluster={CLUSTER_ON} ===")
    for label, ws in [("full_tier1plus", tier_w), ("mm_c4c5", mm_w),
                      ("non_mm", non_w)]:
        sub = sp[sp["wallet_id"].isin(ws)].copy()
        n_w = sub["wallet_id"].nunique()
        n_c = sub["user"].nunique()
        if n_w < 5:
            print(f"  {label}: {n_w} wallets too small"); continue
        r = run(sub, xcol, ctrls, fe, b=500)
        rows.append({"cohort": label, "n_wallets": n_w,
                     "n_clusters": n_c, **r})
        print(f"  {label:<18} wallets={n_w:>3} clusters={n_c:>3}  "
              f"beta={r['beta']:+.4f}  cluster-p={r['p_cluster_normal']:.4g}  "
              f"boot-t p={r['p_bootstrap_t']:.4g}  "
              f"CI=[{r['boot_t_ci_lo']:+.4f},{r['boot_t_ci_hi']:+.4f}]")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
