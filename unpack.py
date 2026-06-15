"""Распаковать все .parquet в data/processed в .csv (один раз перед запуском)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent / "data" / "processed"


def main() -> None:
    parquet_files = sorted(DATA.glob("*.parquet"))
    if not parquet_files:
        print(f"no .parquet files in {DATA}")
        return
    for path in parquet_files:
        csv_path = path.with_suffix(".csv")
        if csv_path.exists():
            continue
        df = pd.read_parquet(path)
        df.to_csv(csv_path, index=False)
        print(f"  {path.name} -> {csv_path.name}")
    print(f"done ({len(parquet_files)} files)")


if __name__ == "__main__":
    main()
