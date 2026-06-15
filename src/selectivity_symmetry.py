"""Order-type and bid-ask side selectivity tests."""

from __future__ import annotations

import os
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

OUT = ROOT / "data" / "processed" / "wallet_gtc_symmetry_summary.csv"
BOOT_B = int(os.environ.get("BOOT_B", "3000"))

def build_panel(s2e):
    p = pd.read_csv(PANEL_FILE)
    p["user"] = p["user"].astype(str).str.lower()
    p["master"] = p["user"].map(s2e)
    p = p[p["master"].notna()].copy()
    sum_cols = ["actions", "alo_order_count", "gtc_order_count",
                "buy_order_count", "sell_order_count"]
    agg = {c: "sum" for c in sum_cols}
    agg["q_pre_coin"] = "sum"
    w = p.groupby(["user", "coin", "bucket"], as_index=False).agg(agg)
    w["bucket_dt"] = pd.to_datetime(w["bucket"], utc=True)
    w = w.sort_values(["user", "coin", "bucket_dt"]).reset_index(drop=True)
    w["master"] = w["user"].map(s2e)

    pre = (w.groupby(["user", "coin"])["q_pre_coin"].first().abs()
             .reset_index(name="pre"))
    pre["q_z"] = (pre["pre"] - pre["pre"].mean()) / pre["pre"].std()
    w = w.merge(pre[["user", "coin", "q_z"]], on=["user", "coin"])

    pcp = build_per_coin_pressure()
    pcp["bucket_dt"] = pd.to_datetime(pcp["bucket"], utc=True)
    w = w.merge(pcp[["coin", "bucket_dt", "press_c"]],
                on=["coin", "bucket_dt"], how="left")
    w["press_c"] = w["press_c"].fillna(0.0)
    w["press_x_qz"] = w["press_c"] * w["q_z"]

    t0 = w["bucket_dt"].min()
    w["t"] = (w["bucket_dt"] - t0).dt.total_seconds() / 60.0
    w["t2"] = w["t"] ** 2
    w["t_x_qz"] = w["t"] * w["q_z"]
    w["t2_x_qz"] = w["t2"] * w["q_z"]
    w["user_coin"] = w["user"] + "_" + w["coin"].astype(str)
    w["coin_minute"] = (w["coin"].astype(str) + "_"
                        + w["bucket_dt"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    w["tkey"] = w["coin_minute"]
    w["wallet_id"] = w["user"]
    w["user"] = w["master"]
    return w

def run_outcome(w, src_col, label, rows):
    d = w.copy()
    d["log1p_alo"] = np.log1p(d[src_col].astype(float))
    d["log1p_alo_lag1"] = (d.groupby(["wallet_id", "coin"])["log1p_alo"]
                             .shift(1).fillna(0))
    fe = ["user_coin", "coin_minute"]
    ctrls = ["log1p_alo_lag1", "t_x_qz"]
    r = run(d, "press_x_qz", ctrls, fe, b=BOOT_B)
    rows.append({"outcome": label, "n_wallets": d["wallet_id"].nunique(),
                 "n_clusters": d["user"].nunique(), **r})
    print(f"  {label:<14} beta={r['beta']:+.4f}  cluster-p={r['p_cluster_normal']:.4g}  "
          f"boot-t p={r['p_bootstrap_t']:.4g}  "
          f"CI=[{r['boot_t_ci_lo']:+.4f},{r['boot_t_ci_hi']:+.4f}]")

def main():
    print(f"BOOT_B={BOOT_B}")
    s2e = load_sub_to_entity()
    tier_info = load_tier_cohort()
    tier_w = {w for w, i in tier_info.items() if i["tier"] >= 1}
    w = build_panel(s2e)
    w = w[w["wallet_id"].isin(tier_w)].copy()
    print(f"  full tier-1+ panel: {len(w):,} rows, {w['wallet_id'].nunique()} wallets, "
          f"{w['user'].nunique()} clusters")
    rows = []
    print("order-type and side selectivity, full tier-1+")
    print("  reference: ALO")
    run_outcome(w, "alo_order_count", "ALO (ref)", rows)
    run_outcome(w, "gtc_order_count", "GTC", rows)
    run_outcome(w, "buy_order_count", "BUY", rows)
    run_outcome(w, "sell_order_count", "SELL", rows)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
