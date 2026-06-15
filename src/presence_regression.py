"""Wallet-level presence regression.

Outcome: 1{buy or sell order > 0} at the wallet-coin-5s level, full
tier-1+ cohort, errors clustered on master. Linear trend.
"""

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
from presence_clustering import aggregate_to_wallet_presence

OUT = ROOT / "data" / "processed" / "wallet_presence_summary.csv"

def main() -> None:
    """Run the presence regression on the full tier-1+ cohort."""
    print("Loading sub -> entity mapping...")
    s2e = load_sub_to_entity()
    tier_info = load_tier_cohort()
    tier_wallets = {w for w, info in tier_info.items() if info["tier"] >= 1}
    print(f"  tier-1+ wallets: {len(tier_wallets)}")

    print("\nBuilding per-coin signed pressure...")
    per_coin_press = build_per_coin_pressure()

    print(f"\nLoading panel {PANEL_FILE.name}...")
    p = pd.read_csv(PANEL_FILE)
    print(f"  {len(p):,} rows")

    print("\nAggregating to wallet-level + presence (buy or sell > 0)...")
    w = aggregate_to_wallet_presence(p, s2e, per_coin_press)
    w = w[w["wallet_id"].isin(tier_wallets)].copy()
    print(f"  wallet panel: {len(w):,} rows, "
          f"{w['wallet_id'].nunique()} wallets, "
          f"{w['user'].nunique()} master clusters, "
          f"presence rate = {w['log1p_alo'].mean():.3f}")

    fe = ["user_coin", "coin_minute"]
    xcol = "press_x_qz"
    ctrls = ["log1p_alo_lag1", "t_x_qz"]
    bootstrap_replicates = int(os.environ.get("BOOT_B", "3000"))

    print("Presence on press_x_qz, FE=(wallet-coin, coin-bucket), "
          "cluster=master, linear trend ===")
    r = run(w, xcol, ctrls, fe, b=bootstrap_replicates)
    n_wallets = w["wallet_id"].nunique()
    n_masters = w["user"].nunique()
    print(f"  full_tier1plus wallets={n_wallets:>3} masters={n_masters:>3}  "
          f"beta={r['beta']:+.4f}  "
          f"cluster-p={r['p_cluster_normal']:.4g}  "
          f"boot-t p={r['p_bootstrap_t']:.4g}  "
          f"sim CI=[{r['boot_t_ci_lo']:+.4f}, {r['boot_t_ci_hi']:+.4f}]")

    pd.DataFrame([{
        "cohort": "full_tier1plus",
        "n_wallets": n_wallets,
        "n_master_clusters": n_masters,
        **r,
    }]).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
