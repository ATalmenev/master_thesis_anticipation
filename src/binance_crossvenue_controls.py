"""Cross-venue Binance return and volatility robustness."""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from inference import run
from pressure_signal import (
    PANEL_FILE, build_per_coin_pressure, load_sub_to_entity)
from cohort import (
    load_tier_cohort, aggregate_to_wallet)

PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "binance_oct10"
CACHE = PROC / "binance_5sec_btc_eth_sol_20251010.csv"
B = int(os.environ.get("BOOT_B", "3000"))
LO = pd.Timestamp("2025-10-10 18:59:00+00:00")
HI = pd.Timestamp("2025-10-10 22:45:05+00:00")

def build_binance_5sec() -> pd.DataFrame:
    if CACHE.exists():
        d = pd.read_csv(CACHE, parse_dates=["bucket"])
        d["bucket"] = pd.to_datetime(d["bucket"], utc=True)
        return d
    frames = []
    for coin, sym in [("BTC", "BTCUSDT"), ("ETH", "ETHUSDT"), ("SOL", "SOLUSDT")]:
        z = RAW / f"{sym}-aggTrades-2025-10-10.zip"
        with zipfile.ZipFile(z) as zf:
            df = pd.read_csv(zf.open(zf.namelist()[0]),
                             usecols=["price", "transact_time"])
        df["t"] = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
        df = df[(df["t"] >= LO) & (df["t"] < HI)].sort_values("t")
        df["lr"] = np.log(df["price"]).diff()
        df["bucket"] = df["t"].dt.floor("5s")
        g = df.groupby("bucket").agg(
            close=("price", "last"),
            rv=("lr", lambda x: float(np.sqrt(np.nansum(np.square(x))))))
        g = g.reset_index()
        g["bret"] = np.log(g["close"]).diff().fillna(0.0)
        g["bvol"] = g["rv"].fillna(0.0)
        g["coin"] = coin
        g = g.sort_values("bucket")
        g["bret_lag1"] = g["bret"].shift(1).fillna(0.0)
        g["bvol_lag1"] = g["bvol"].shift(1).fillna(0.0)
        frames.append(g[["coin", "bucket", "bret", "bvol", "bret_lag1", "bvol_lag1"]])
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(CACHE, index=False)
    return out

def main():
    s2e = load_sub_to_entity()
    ti = load_tier_cohort()
    tier_w = {w for w, i in ti.items() if i["tier"] >= 1}
    pcp = build_per_coin_pressure()
    p = pd.read_csv(PANEL_FILE)
    w = aggregate_to_wallet(p, s2e, pcp)
    w["log1p_alo"] = w["log1p_alo"].astype(float)

    bm = build_binance_5sec()
    bm = bm.rename(columns={"bucket": "bucket_dt"})
    w = w.merge(bm, on=["coin", "bucket_dt"], how="left")
    nmiss = w[["bret", "bvol"]].isna().any(axis=1).sum()
    for c in ["bret", "bvol", "bret_lag1", "bvol_lag1"]:
        w[c] = w[c].fillna(0.0)
        w[c + "_x_qz"] = w[c] * w["q_z"]
    print(f"panel rows {len(w):,}; rows with no Binance 5-sec match "
          f"(pre-21:00 lag region) {nmiss:,}; "
          f"Binance vol range [{bm['bvol'].min():.2g},{bm['bvol'].max():.2g}]; B={B}")

    sub = w[w["wallet_id"].isin(tier_w)].copy()
    fe = ["user_coin", "coin_minute"]
    base = ["log1p_alo_lag1", "t_x_qz"]
    specs = [
        ("baseline (linear, no Binance)", base),
        ("+ Binance ret x qz",            base + ["bret_x_qz"]),
        ("+ Binance vol x qz",            base + ["bvol_x_qz"]),
        ("+ Binance ret & vol x qz",      base + ["bret_x_qz", "bvol_x_qz"]),
        ("+ Binance ret&vol, contemp+lag", base + ["bret_x_qz", "bvol_x_qz",
                                                   "bret_lag1_x_qz", "bvol_lag1_x_qz"]),
    ]
    print(f"\n{'spec':<34}{'beta':>9}{'cluster-p':>11}{'boot-t p':>10}  CI")
    for label, ctrls in specs:
        r = run(sub, "press_x_qz", ctrls, fe, b=B)
        print(f"{label:<34}{r['beta']:>+9.4f}{r['p_cluster_normal']:>11.3g}"
              f"{r['p_bootstrap_t']:>10.3g}  "
              f"[{r['boot_t_ci_lo']:+.4f},{r['boot_t_ci_hi']:+.4f}]")

if __name__ == "__main__":
    main()
