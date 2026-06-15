"""Trading-margin regression: executed maker fills on HLP pressure."""

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
    build_per_coin_pressure, load_sub_to_entity,
)
from cohort import load_tier_cohort
from spread_response import build_qz_from_panel

SPREAD = ROOT / "data" / "processed" / "wallet_spread_panel_4h_20251010.csv"
OUT = ROOT / "data" / "processed" / "wallet_fills_on_pressure_summary.csv"
WIN_LO = pd.Timestamp("2025-10-10 19:00:00+00:00")
WIN_HI = pd.Timestamp("2025-10-10 22:45:00+00:00")
CLUSTER_ON = os.environ.get("CLUSTER_ON", "master")
OUTCOME = os.environ.get("OUTCOME", "fills")

def build_dense_fills(s2e: dict[str, str]) -> pd.DataFrame:
    sp = pd.read_csv(SPREAD)
    sp["user"] = sp["user"].astype(str).str.lower()
    sp["master"] = sp["user"].map(s2e)
    sp = sp[sp["master"].notna()].copy()
    sp["bucket_dt"] = pd.to_datetime(sp["bucket"], utc=True).dt.floor("5s")
    sp = (sp.groupby(["user", "coin", "bucket_dt"], as_index=False)
            .agg(n_maker_fills=("n_maker_fills", "sum"),
                 maker_notional=("maker_notional", "sum")))

    pairs = sp[["user", "coin"]].drop_duplicates()
    buckets = pd.date_range(WIN_LO, WIN_HI, freq="5s", inclusive="left")
    grid = (pairs.assign(key=1)
            .merge(pd.DataFrame({"bucket_dt": buckets, "key": 1}), on="key")
            .drop(columns="key"))
    d = grid.merge(sp, on=["user", "coin", "bucket_dt"], how="left")
    d["n_maker_fills"] = d["n_maker_fills"].fillna(0.0)
    d["maker_notional"] = d["maker_notional"].fillna(0.0)
    d["master"] = d["user"].map(s2e)

    if OUTCOME == "notional":
        d["outcome"] = np.log1p(d["maker_notional"])
    else:
        d["outcome"] = np.log1p(d["n_maker_fills"])

    qz = build_qz_from_panel(s2e)
    d = d.merge(qz, on=["user", "coin"], how="left")
    d = d[d["q_z"].notna()].copy()

    pcp = build_per_coin_pressure()
    pcp["bucket_dt"] = pd.to_datetime(pcp["bucket"], utc=True)
    d = d.merge(pcp[["coin", "bucket_dt", "press_c"]],
                on=["coin", "bucket_dt"], how="left")
    d["press_c"] = d["press_c"].fillna(0.0)
    d["press_x_qz"] = d["press_c"] * d["q_z"]

    d = d.sort_values(["user", "coin", "bucket_dt"]).reset_index(drop=True)

    d["log1p_alo"] = d["outcome"].astype(float)
    d["log1p_alo_lag1"] = (d.groupby(["user", "coin"])["outcome"]
                             .shift(1).fillna(0))
    t0 = d["bucket_dt"].min()
    d["t"] = (d["bucket_dt"] - t0).dt.total_seconds() / 60.0
    d["t2"] = d["t"] ** 2
    d["t_x_qz"] = d["t"] * d["q_z"]
    d["t2_x_qz"] = d["t2"] * d["q_z"]
    d["user_coin"] = d["user"] + "_" + d["coin"].astype(str)
    d["coin_minute"] = (d["coin"].astype(str) + "_"
                        + d["bucket_dt"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    d["tkey"] = d["coin_minute"]
    d["wallet_id"] = d["user"]
    if CLUSTER_ON == "master":
        d["user"] = d["master"]
    return d

def main() -> None:
    print(f"CLUSTER_ON={CLUSTER_ON} OUTCOME={OUTCOME}")
    s2e = load_sub_to_entity()
    tier_info = load_tier_cohort()
    tier_w = {w for w, i in tier_info.items() if i["tier"] >= 1}

    d = build_dense_fills(s2e)
    rate = (d["n_maker_fills"] > 0).mean()
    print(f"  dense panel: {len(d):,} rows, {d['wallet_id'].nunique()} wallets, "
          f"{d['user'].nunique()} clusters, fill-positive rate {rate:.3f}")

    fe = ["user_coin", "coin_minute"]
    xcol = "press_x_qz"
    ctrls = ["log1p_alo_lag1", "t_x_qz"]
    print(f"TRADING margin: log1p({OUTCOME}) on press_x_qz, "
          f"FE=(wallet-coin, coin-bucket), cluster={CLUSTER_ON} ===")
    sub = d[d["wallet_id"].isin(tier_w)].copy()
    n_wallets = sub["wallet_id"].nunique()
    n_clusters = sub["user"].nunique()
    r = run(sub, xcol, ctrls, fe, b=int(os.environ.get("BOOT_B", "3000")))
    rows = [{"cohort": "full_tier1plus", "n_wallets": n_wallets,
             "n_clusters": n_clusters, "outcome": OUTCOME, **r}]
    print(f"  full_tier1plus wallets={n_wallets:>3} clusters={n_clusters:>3}  "
          f"beta={r['beta']:+.4f}  cluster-p={r['p_cluster_normal']:.4g}  "
          f"boot-t p={r['p_bootstrap_t']:.4g}  "
          f"CI=[{r['boot_t_ci_lo']:+.4f},{r['boot_t_ci_hi']:+.4f}]")
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
