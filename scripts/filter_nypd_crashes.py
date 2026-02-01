#!/usr/bin/env python3
"""
filter_nypd_crashes.py - WITH INPUT VALIDATION + QA REPORT

Filters NYPD crash data by year and month and writes a cleaned subset CSV that is safe for
spatial crash-proximity work.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def normalize(col: str) -> str:
    """Normalize column names to lowercase with underscores."""
    return str(col).strip().lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def validate_years(years: list[int]) -> None:
    """Validate year inputs are reasonable (2013-2030)."""
    MIN_YEAR = 2013  # Citi Bike launched June 2013
    MAX_YEAR = 2030  # future-proof but catches typos
    bad = [y for y in years if not (MIN_YEAR <= y <= MAX_YEAR)]
    if bad:
        raise ValueError(
            f"Year(s) out of range ({MIN_YEAR}-{MAX_YEAR}): {bad}\n"
            f"Check --years argument. Received: {years}"
        )


def validate_months(months: list[int]) -> None:
    """Validate month inputs are in range 1-12."""
    bad = [m for m in months if not (1 <= m <= 12)]
    if bad:
        raise ValueError(
            f"Month(s) out of range (1-12): {bad}\nCheck --months argument. Received: {months}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Filter NYPD crash data by year and month (with QA + coord cleaning)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/filter_nypd_crashes.py --years 2023 2024 --months 1 2 3
  python scripts/filter_nypd_crashes.py --years 2023 --months 1 2 3 --apply-nyc-bbox
        """,
    )
    ap.add_argument(
        "--in-path", default="data/raw/nypd/h9gi-nx95_full.csv", help="Path to full NYPD crash CSV"
    )
    ap.add_argument(
        "--out-path",
        default=None,
        help="Output path. If omitted, auto-generated under data/processed/",
    )
    ap.add_argument("--years", nargs="+", type=int, required=True, help="Years to filter")
    ap.add_argument("--months", nargs="+", type=int, required=True, help="Months to filter (1-12)")
    ap.add_argument(
        "--apply-nyc-bbox",
        action="store_true",
        help="If set, keep only crashes inside an NYC-ish bounding box (recommended for station proximity).",
    )
    ap.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="CSV chunk size for streaming processing",
    )

    args = ap.parse_args()

    # ========== VALIDATE INPUTS ==========
    print("Validating inputs...")
    try:
        validate_years(args.years)
        validate_months(args.months)
    except ValueError as e:
        raise SystemExit(f"ERROR: {e}")
    years = set(args.years)
    months = set(args.months)
    print(f"✓ Valid years: {sorted(years)}")
    print(f"✓ Valid months: {sorted(months)}")
    # =====================================

    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"ERROR: Input file not found: {in_path}")

    # Output path
    if args.out_path:
        out_path = Path(args.out_path)
    else:
        years_tag = "_".join(str(y) for y in sorted(years))
        months_tag = "_".join(f"{m:02d}" for m in sorted(months))
        out_path = Path(f"data/processed/nypd_crashes_y{years_tag}_m{months_tag}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read header to map column names
    raw_cols = pd.read_csv(in_path, nrows=0).columns.tolist()
    norm_map = {c: normalize(c) for c in raw_cols}

    # Guard: detect normalized collisions (fail fast, strict)
    norm_vals = list(norm_map.values())
    dupes = sorted({v for v in norm_vals if norm_vals.count(v) > 1})
    if dupes:
        raise SystemExit(
            "ERROR: Column normalization produced duplicate names:\n"
            + "\n".join(f"  - {d}" for d in dupes)
            + "\nFix normalize() or handle duplicates explicitly."
        )

    inv_map = {v: k for k, v in norm_map.items()}

    needed_norm = [
        "collision_id",
        "crash_date",
        "crash_time",
        "borough",
        "latitude",
        "longitude",
        "number_of_persons_injured",
        "number_of_cyclist_injured",
        "number_of_cyclist_killed",
    ]
    missing = [c for c in needed_norm if c not in inv_map]
    if missing:
        raise SystemExit(
            "Missing expected columns after normalization:\n"
            + "\n".join(f"  - {c}" for c in missing)
            + f"\n\nAvailable columns (first 30 normalized): {norm_vals[:30]}"
        )

    usecols_raw = [inv_map[c] for c in needed_norm]

    # NYC-ish bbox (broad & conservative)
    NYC_LAT_MIN, NYC_LAT_MAX = 40.3, 41.1
    NYC_LNG_MIN, NYC_LNG_MAX = -74.5, -73.2

    # QA counters (totals across all chunks)
    qa = {
        "rows_in": 0,
        "bad_crash_date": 0,           # crash_date parse failed (NaT)
        "bad_lat_or_lng_nan": 0,       # lat/lng NaN after numeric conversion
        "bad_latlng_out_of_range": 0,  # outside valid earth ranges
        "bad_latlng_zero_zero": 0,     # (0,0)
        "filtered_out_by_year_month": 0,
        "filtered_out_by_nyc_bbox": 0,
        "rows_out": 0,
        "rows_written_chunks": 0,      # chunks with >0 rows written
    }

    first_write = True

    print(f"\nProcessing: {in_path}")
    print(f"Output:     {out_path}")
    print(f"Filter:     years={sorted(years)}, months={sorted(months)}")
    print(f"NYC bbox:   {'ON' if args.apply_nyc_bbox else 'OFF'}")
    print(f"Chunk size: {int(args.chunk_size):,}\n")

    for chunk in pd.read_csv(in_path, usecols=usecols_raw, chunksize=int(args.chunk_size)):
        qa["rows_in"] += len(chunk)
        chunk = chunk.rename(columns=norm_map)

        # Parse/cast into local series first so we can compute QA masks pre-drop
        crash_date = pd.to_datetime(chunk["crash_date"], format="%m/%d/%Y", errors="coerce")
        lat = pd.to_numeric(chunk["latitude"], errors="coerce")
        lng = pd.to_numeric(chunk["longitude"], errors="coerce")

        # QA masks on raw-casted data (note: categories can overlap)
        m_bad_date = crash_date.isna()
        m_bad_nan = lat.isna() | lng.isna()
        m_bad_range = ~lat.between(-90, 90) | ~lng.between(-180, 180)
        m_zero_zero = (lat == 0) & (lng == 0)

        qa["bad_crash_date"] += int(m_bad_date.sum())
        qa["bad_lat_or_lng_nan"] += int(m_bad_nan.sum())
        qa["bad_latlng_out_of_range"] += int(m_bad_range.sum())
        qa["bad_latlng_zero_zero"] += int(m_zero_zero.sum())

        # Assign cleaned columns back
        chunk["crash_date"] = crash_date
        chunk["latitude"] = lat
        chunk["longitude"] = lng

        # Drop invalid essentials
        chunk = chunk.dropna(subset=["crash_date", "latitude", "longitude"])

        # Keep only valid earth-range coords
        chunk = chunk[
            chunk["latitude"].between(-90, 90) & chunk["longitude"].between(-180, 180)
        ]

        # Drop bogus default coord
        chunk = chunk[~((chunk["latitude"] == 0) & (chunk["longitude"] == 0))]

        # Filter by year/month (QA)
        before_ym = len(chunk)
        keep_ym = chunk["crash_date"].dt.year.isin(years) & chunk["crash_date"].dt.month.isin(months)
        chunk = chunk[keep_ym]
        qa["filtered_out_by_year_month"] += int(before_ym - len(chunk))

        # Optional bbox filter (QA)
        if args.apply_nyc_bbox:
            before_bbox = len(chunk)
            keep_bbox = (
                chunk["latitude"].between(NYC_LAT_MIN, NYC_LAT_MAX)
                & chunk["longitude"].between(NYC_LNG_MIN, NYC_LNG_MAX)
            )
            chunk = chunk[keep_bbox]
            qa["filtered_out_by_nyc_bbox"] += int(before_bbox - len(chunk))

        qa["rows_out"] += len(chunk)

        # Write chunk
        if len(chunk) > 0:
            chunk.to_csv(
                out_path,
                mode="w" if first_write else "a",
                index=False,
                header=first_write,
            )
            qa["rows_written_chunks"] += 1
            first_write = False

        print(f"Processed {qa['rows_in']:,} rows → kept {qa['rows_out']:,} rows", end="\r")

    print(f"\nProcessed {qa['rows_in']:,} rows → kept {qa['rows_out']:,} rows")
    print(f"Saved: {out_path}")

    # Final QA report
    print("\n=== QA: Filtering breakdown (totals) ===")
    print(f"Input rows:                      {qa['rows_in']:,}")
    print(f"Bad crash_date (unparseable):    {qa['bad_crash_date']:,}")
    print(f"Bad lat/lng NaN:                 {qa['bad_lat_or_lng_nan']:,}")
    print(f"Bad lat/lng out of range:        {qa['bad_latlng_out_of_range']:,}")
    print(f"Bad lat/lng == (0,0):            {qa['bad_latlng_zero_zero']:,}")
    print(f"Filtered out by year/month:      {qa['filtered_out_by_year_month']:,}")
    if args.apply_nyc_bbox:
        print(f"Filtered out by NYC bbox:        {qa['filtered_out_by_nyc_bbox']:,}")
    print(f"Output rows:                     {qa['rows_out']:,}")
    if qa["rows_in"] > 0:
        print(f"Retention:                       {100.0 * qa['rows_out'] / qa['rows_in']:.2f}%")
    print(f"Chunks written (non-empty):      {qa['rows_written_chunks']:,}")

    if qa["rows_out"] == 0:
        print("\n⚠ WARNING: No rows matched the filter criteria.")
        print("   Possible causes:")
        print("   - No crashes recorded for these years/months in the source")
        print("   - Wrong year/month arguments")
        print("   - Unexpected date format in source (but we're using MM/DD/YYYY)")


if __name__ == "__main__":
    main()
