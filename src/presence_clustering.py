"""Presence (extensive margin) regression with master-level clustering."""

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

OUT = ROOT / "data" / "processed" / "wallet_presence_cluster_master_summary.csv"

def aggregate_to_wallet_presence(p: pd.DataFrame, s2e: dict[str, str],
                                  per_coin_press: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["user"] = p["user"].astype(str).str.lower()
    p["master"] = p["user"].map(s2e)
    p = p[p["master"].notna()].copy()

    sum_cols = ["actions", "alo_order_count", "gtc_order_count",
                "ioc_order_count", "cancel_total",
                "buy_order_count", "sell_order_count"]
    agg = {c: "sum" for c in sum_cols}
    agg["q_pre_coin"] = "sum"
    w = (p.groupby(["user", "coin", "bucket"], as_index=False).agg(agg))

    w["presence"] = (w["buy_order_count"] + w["sell_order_count"] > 0).astype(int)
    w["bucket_dt"] = pd.to_datetime(w["bucket"], utc=True)
    w = w.sort_values(["user", "coin", "bucket_dt"]).reset_index(drop=True)
    w["master"] = w["user"].map(s2e)

    pre_per_wc = (w.groupby(["user", "coin"])["q_pre_coin"]
                    .first().abs().reset_index(name="w_m_c_pre"))
    pre_per_wc["q_z"] = ((pre_per_wc["w_m_c_pre"]
                         - pre_per_wc["w_m_c_pre"].mean())
                        / pre_per_wc["w_m_c_pre"].std())
    w = w.merge(pre_per_wc[["user", "coin", "q_z"]], on=["user", "coin"])

    pcp = per_coin_press.copy()
    pcp["bucket_dt"] = pd.to_datetime(pcp["bucket"], utc=True)
    w = w.merge(pcp[["coin", "bucket_dt", "press_c"]],
                on=["coin", "bucket_dt"], how="left")
    w["press_c"] = w["press_c"].fillna(0.0)
    w["press_x_qz"] = w["press_c"] * w["q_z"]
    w["presence_lag1"] = (w.groupby(["user", "coin"])
                              ["presence"].shift(1).fillna(0))

    t0 = w["bucket_dt"].min()
    w["t"] = (w["bucket_dt"] - t0).dt.total_seconds() / 60.0
    w["t2"] = w["t"] ** 2
    w["t_x_qz"] = w["t"] * w["q_z"]
    w["t2_x_qz"] = w["t2"] * w["q_z"]

    w["user_coin"] = w["user"] + "_" + w["coin"].astype(str)
    w["coin_minute"] = (w["coin"].astype(str) + "_"
                        + w["bucket_dt"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    w["tkey"] = w["coin_minute"]

    w["log1p_alo"] = w["presence"].astype(float)
    w["log1p_alo_lag1"] = w["presence_lag1"].astype(float)

    w["wallet_id"] = w["user"]
    w["user"] = w["master"]
    return w

def main() -> None:
    print("Loading sub -> entity mapping...")
    s2e = load_sub_to_entity()
    tier_info = load_tier_cohort()
    tier_wallets = {w for w, info in tier_info.items() if info["tier"] >= 1}
    print(f"  tier-1+: {len(tier_wallets)}")

    print("\nBuilding per-coin signed pressure...")
    per_coin_press = build_per_coin_pressure()

    print(f"\nLoading panel {PANEL_FILE.name}...")
    p = pd.read_csv(PANEL_FILE)
    print(f"  {len(p):,} rows")

    print("\nAggregating to wallet-level + presence outcome...")
    w = aggregate_to_wallet_presence(p, s2e, per_coin_press)
    print(f"  wallet panel: {len(w):,} rows, "
          f"{w['wallet_id'].nunique()} wallets, "
          f"{w['user'].nunique()} master clusters, presence rate "
          f"{w['log1p_alo'].mean():.3f}")

    fe = ["user_coin", "coin_minute"]
    xcol = "press_x_qz"
    ctrls = ["log1p_alo_lag1", "t_x_qz"]

    print(f"Presence: LP-prob on press_x_qz, FE=(wallet-coin, "
          f"coin-bucket), cluster=master ===")
    sub = w[w["wallet_id"].isin(tier_wallets)].copy()
    n_wallets = sub["wallet_id"].nunique()
    n_masters = sub["user"].nunique()
    r = run(sub, xcol, ctrls, fe, b=int(os.environ.get("BOOT_B", "3000")))
    rows = [{"cohort": "full_tier1plus", "n_wallets": n_wallets,
             "n_master_clusters": n_masters, **r}]
    print(f"  full_tier1plus wallets={n_wallets:>3} clusters={n_masters:>3}  "
          f"beta={r['beta']:+.4f}  "
          f"cluster-p={r['p_cluster_normal']:.4g}  "
          f"boot-t p={r['p_bootstrap_t']:.4g}  "
          f"sim CI=[{r['boot_t_ci_lo']:+.4f}, {r['boot_t_ci_hi']:+.4f}]")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
