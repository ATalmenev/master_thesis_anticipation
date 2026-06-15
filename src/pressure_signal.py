"""Entity-level ALO regression on per-coin signed HLP pressure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from inference import run

PROCESSED = ROOT / "data" / "processed"
PANEL_FILE = PROCESSED / "fivesec_user_coin_panel_4h_20251010.csv"
ENTITIES_FILE = PROCESSED / "lp_master_full_subs.json"
PRESSURE_FILE = PROCESSED / "hlp_inventory_pressure_stock_by_coin_4h.csv"
OUT = PROCESSED / "entity_alo_signed_pressure_summary.csv"

def load_sub_to_entity() -> dict[str, str]:
    raw = json.loads(ENTITIES_FILE.read_text())
    s2e = {}
    for entity, subs in raw.items():
        e = entity.lower()
        for s in subs:
            s2e[s.lower()] = e
    return s2e

def build_per_coin_pressure() -> pd.DataFrame:
    p = pd.read_csv(PRESSURE_FILE)
    p["second"] = pd.to_datetime(p["second"], utc=True)
    p["bucket"] = p["second"].dt.floor("5s")

    per = (p.sort_values("second")
             .groupby(["coin", "bucket"], as_index=False)
             .agg(I_ct=("net_inventory_pressure", "last")))

    out = []
    for coin, g in per.groupby("coin"):
        g = g.sort_values("bucket").copy()
        m = float(max(g["I_ct"].max(), 1.0))
        g["press_c"] = g["I_ct"] / m
        g["dpress_c"] = g["press_c"].diff().fillna(0.0)
        out.append(g[["coin", "bucket", "press_c", "dpress_c"]])
    return pd.concat(out, ignore_index=True)

def aggregate_to_entity(p: pd.DataFrame, s2e: dict[str, str],
                        per_coin_press: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    p["entity"] = p["user"].astype(str).str.lower().map(s2e)
    p = p[p["entity"].notna()].copy()

    sum_cols = ["actions", "alo_order_count", "gtc_order_count",
                "ioc_order_count", "cancel_total", "txs"]
    pos_cols = ["q_pre_coin", "q_pre_focus", "q_pre_btc", "q_pre_eth",
                "q_pre_sol"]
    state_cols = ["forced_flow_notional", "max_abs_mid_return_1s",
                  "mean_spread_bps"]

    agg = {c: "sum" for c in sum_cols}
    for c in pos_cols:
        agg[c] = "sum"
    for c in state_cols:
        agg[c] = "first"
    e = (p.groupby(["entity", "coin", "bucket"], as_index=False)
            .agg(agg))
    e["log1p_alo"] = np.log1p(e["alo_order_count"])
    e["log1p_q_pre_coin"] = np.log1p(e["q_pre_coin"].abs())

    e["bucket_dt"] = pd.to_datetime(e["bucket"], utc=True)
    e = e.sort_values(["entity", "coin", "bucket_dt"]).reset_index(drop=True)

    pre_per_ec = (e.groupby(["entity", "coin"])["q_pre_coin"]
                    .first().abs().reset_index(name="e_m_c_pre"))
    pre_per_ec["q_z"] = ((pre_per_ec["e_m_c_pre"]
                          - pre_per_ec["e_m_c_pre"].mean())
                         / pre_per_ec["e_m_c_pre"].std())
    e = e.merge(pre_per_ec[["entity", "coin", "q_z"]],
                on=["entity", "coin"])

    per_coin_press = per_coin_press.copy()
    per_coin_press["bucket_dt"] = pd.to_datetime(per_coin_press["bucket"],
                                                 utc=True)
    e = e.merge(per_coin_press[["coin", "bucket_dt", "press_c",
                                 "dpress_c"]],
                on=["coin", "bucket_dt"], how="left")
    e["press_c"] = e["press_c"].fillna(0.0)
    e["dpress_c"] = e["dpress_c"].fillna(0.0)

    e["press_x_qz"] = e["press_c"] * e["q_z"]
    e["dpress_x_qz"] = e["dpress_c"] * e["q_z"]

    e["log1p_alo_lag1"] = (e.groupby(["entity", "coin"])
                              ["log1p_alo"].shift(1).fillna(0))

    t0 = e["bucket_dt"].min()
    e["t"] = (e["bucket_dt"] - t0).dt.total_seconds() / 60.0
    e["t2"] = e["t"] ** 2
    e["t_x_qz"] = e["t"] * e["q_z"]
    e["t2_x_qz"] = e["t2"] * e["q_z"]

    e["user"] = e["entity"]
    e["user_coin"] = e["entity"] + "_" + e["coin"].astype(str)
    e["coin_minute"] = (e["coin"].astype(str) + "_"
                        + e["bucket_dt"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    e["tkey"] = e["coin_minute"]

    return e

def main() -> None:
    print("Loading sub → entity mapping...")
    s2e = load_sub_to_entity()
    print(f"  {len(s2e)} subs → {len(set(s2e.values()))} entities\n")

    print(f"Building per-coin signed pressure...")
    per_coin_press = build_per_coin_pressure()
    print(f"  {len(per_coin_press):,} (coin, bucket) rows, "
          f"coins = {sorted(per_coin_press['coin'].unique())}")
    for c, g in per_coin_press.groupby("coin"):
        print(f"    {c}: press range [{g['press_c'].min():.3f}, "
              f"{g['press_c'].max():.3f}]")

    print(f"\nLoading panel {PANEL_FILE.name}...")
    p = pd.read_csv(PANEL_FILE)
    print(f"  {len(p):,} rows, {p['user'].nunique()} unique users")

    print("\nAggregating to entity-level + merging signed pressure...")
    e = aggregate_to_entity(p, s2e, per_coin_press)

    print(f"  entity panel: {len(e):,} rows, "
          f"{e['entity'].nunique()} entities, "
          f"{e['coin'].nunique()} coins, "
          f"{e['bucket_dt'].nunique()} buckets")

    print(f"log1p_alo ~ β·(press_c × q_z) + ctrls + 2way FE "
          f"===")
    print(f"  Pressure: per-coin SIGNED (Version A)")
    print(f"  cluster_unit = entity (N = {e['entity'].nunique()})")
    print(f"  time blocks (tkey) = {e['tkey'].nunique()}")

    fe = ["user_coin", "coin_minute"]
    xcol = "press_x_qz"
    specs = [
        ("FE+lag",                ["log1p_alo_lag1"]),
        ("FE+lag + time",         ["log1p_alo_lag1", "t_x_qz"]),
        ("FE+lag + time+time²",   ["log1p_alo_lag1", "t_x_qz", "t2_x_qz"]),
    ]

    rows = []
    print(f"\n  {'spec':<26}{'beta':>10}{'cluster-SE p':>16}"
          f"{'bootstrap-t p':>16}{'bootstrap-t 95% CI':>28}")
    for slab, ctrls in specs:
        r = run(e, xcol, ctrls, fe)
        rows.append({"spec": slab, **r})
        print(f"  {slab:<26}{r['beta']:>+10.4f}{r['p_cluster_normal']:>16.3g}"
              f"{r['p_bootstrap_t']:>16.3g}"
              f"   [{r['boot_t_ci_lo']:+.4f}, {r['boot_t_ci_hi']:+.4f}]")

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
