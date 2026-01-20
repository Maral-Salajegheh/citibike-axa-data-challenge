#!/usr/bin/env python3
"""
filter_nypd_crashes.py - WITH INPUT VALIDATION

Filters NYPD crash data by year and month.
Now includes validation to catch invalid inputs before processing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize(col: str) -> str:
    """Normalize column names to lowercase with underscores."""
    return (
        str(col).strip().lower()
        .replace(" ", "_").replace("-", "_").replace("/", "_")
    )


def validate_years(years: list[int]) -> None:
    """
    Validate year inputs are reasonable (2013-2030).
    
    Raises:
        ValueError: If any year is outside reasonable bounds.
    """
    MIN_YEAR = 2013  # Citi Bike launched June 2013
    MAX_YEAR = 2030  # Future-proof but catch typos
    
    for y in years:
        if not (MIN_YEAR <= y <= MAX_YEAR):
            raise ValueError(
                f"Year out of reasonable range ({MIN_YEAR}-{MAX_YEAR}): {y}\n"
                f"Citi Bike launched in 2013. NYPD crash data starts around 2012.\n"
                f"Check --years argument. Received: {years}"
            )


def validate_months(months: list[int]) -> None:
    """
    Validate month inputs are in range 1-12.
    
    Raises:
        ValueError: If any month is outside 1-12.
    """
    for m in months:
        if not (1 <= m <= 12):
            raise ValueError(
                f"Month out of range (1-12): {m}\n"
                f"Check --months argument for typos.\n"
                f"Received: {months}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter NYPD crash data by year and month",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/filter_nypd_crashes.py --years 2023 2024 --months 1 2 3
  python scripts/filter_nypd_crashes.py --years 2023 --months 1 2 3 4 5 6 7 8 9 10 11 12
        """
    )
    ap.add_argument("--in-path", default="data/raw/nypd/h9gi-nx95_full.csv",
                    help="Path to full NYPD crash CSV")
    ap.add_argument("--out-path", default=None,
                    help="Output path. If omitted, auto-generated under data/processed/")
    ap.add_argument("--years", nargs="+", type=int, required=True,
                    help="Years to filter (e.g., 2023 2024)")
    ap.add_argument("--months", nargs="+", type=int, required=True,
                    help="Months to filter (1-12, e.g., 1 2 3)")
    args = ap.parse_args()

    # ========== VALIDATE INPUTS ==========
    print("Validating inputs...")
    try:
        validate_years(args.years)
        validate_months(args.months)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")
    print(f"✓ Valid years: {sorted(args.years)}")
    print(f"✓ Valid months: {sorted(args.months)}")
    # =====================================

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"ERROR: Input file not found: {in_path}")
    
    years = set(args.years)
    months = set(args.months)

    if args.out_path:
        out_path = Path(args.out_path)
    else:
        years_tag = "_".join(str(y) for y in sorted(args.years))
        months_tag = "_".join(f"{m:02d}" for m in sorted(args.months))
        out_path = Path(f"data/processed/nypd_crashes_y{years_tag}_m{months_tag}.csv")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read header to map column names
    raw_cols = pd.read_csv(in_path, nrows=0).columns.tolist()
    norm_map = {c: normalize(c) for c in raw_cols}
    inv_map = {v: k for k, v in norm_map.items()}

    needed_norm = [
        "collision_id", "crash_date", "crash_time", "borough",
        "latitude", "longitude",
        "number_of_persons_injured",
        "number_of_cyclist_injured",
        "number_of_cyclist_killed",
    ]

    missing = [c for c in needed_norm if c not in inv_map]
    if missing:
        raise SystemExit(
            "Missing expected columns after normalization:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + f"\n\nAvailable columns (first 20): {list(norm_map.values())[:20]}"
        )

    usecols_raw = [inv_map[c] for c in needed_norm]

    rows_in = 0
    rows_out = 0
    first_write = True

    print(f"\nProcessing: {in_path}")
    print(f"Output: {out_path}")
    print(f"Filtering: years={sorted(years)}, months={sorted(months)}\n")

    for chunk in pd.read_csv(in_path, usecols=usecols_raw, chunksize=200_000):
        rows_in += len(chunk)
        chunk = chunk.rename(columns=norm_map)

        chunk["crash_date"] = pd.to_datetime(chunk["crash_date"], errors="coerce")
        chunk["latitude"] = pd.to_numeric(chunk["latitude"], errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk["longitude"], errors="coerce")
        chunk = chunk.dropna(subset=["crash_date", "latitude", "longitude"])

        chunk = chunk[
            chunk["crash_date"].dt.year.isin(years)
            & chunk["crash_date"].dt.month.isin(months)
        ]

        rows_out += len(chunk)

        if len(chunk) > 0:
            chunk.to_csv(out_path, mode="w" if first_write else "a", index=False, header=first_write)
            first_write = False

        print(f"Processed {rows_in:,} rows → kept {rows_out:,} rows", end="\r")

    print(f"\nProcessed {rows_in:,} rows → kept {rows_out:,} rows")
    print(f"Saved: {out_path}")
    
    if rows_out == 0:
        print("\n⚠ WARNING: No rows matched the filter criteria.")
        print("   This might indicate:")
        print("   - No crashes recorded for these years/months")
        print("   - Date format issues in the source data")
        print("   - Check that your year/month filters are correct")


if __name__ == "__main__":
    main()