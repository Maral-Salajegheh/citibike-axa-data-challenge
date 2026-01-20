#!/usr/bin/env python3
"""
summarize_citibike_usage.py

Reproducible summary metrics from partitioned Citi Bike parquet.

Expected parquet layout:
  <PARQUET_DIR>/mode=<mode>/year=<YYYY>/month=<MM>/tripdata.parquet

Writes CSVs into --out-dir:
  - citibike_trips_by_month.csv
  - citibike_trips_by_year.csv
  - citibike_trips_by_dow.csv            (includes year)
  - citibike_trips_by_hour.csv           (includes year)
  - citibike_station_exposure.csv        (includes year, month)
  - station_risk_exposure_plus_crashproximity.csv                    (overall / lifetime aggregate)
  - station_risk_exposure_plus_crashproximity_by_year.csv            (NEW: station × year)
  - station_risk_exposure_plus_crashproximity_by_year_month.csv      (NEW: station × year × month)
  - summary_highlights.md (unless --no-highlights)

NYC/JC hygiene (station exposure):
- NYC mode: removes JC/HB-prefixed stations AND stations inside a Jersey City bounding box
- JC mode: keeps only JC/HB-prefixed stations OR stations inside that JC bounding box
This helps prevent cross-system contamination (NYC vs JC).

Crash proximity (NYPD proxy):
- NYC-only proxy. If --mode=jc, writes zeros with risk_proxy_available=0.
- Overall file remains lifetime-aggregated for stability.
- NEW per-year and per-year-month files are computed by filtering crashes by crash_date.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Path parsing
# -----------------------------
_PART_RE = re.compile(r"mode=([^/]+)/year=(\d{4})/month=(\d{1,2})/tripdata\.parquet$")


def parse_partition(p: Path) -> Optional[Tuple[str, int, int]]:
    m = _PART_RE.search(str(p).replace("\\", "/"))
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


# -----------------------------
# PyArrow helpers (minimal load)
# -----------------------------
def _require_pyarrow() -> None:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "ERROR: pyarrow is required.\n"
            "Install in your venv:\n"
            "  .venv/bin/pip install pyarrow\n"
        ) from e


def parquet_num_rows(path: Path) -> int:
    _require_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return int(pf.metadata.num_rows)


def parquet_available_columns(path: Path) -> List[str]:
    _require_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    return list(pf.schema.names)


def iter_parquet_batches(parquet_path: Path, columns: List[str], batch_size: int = 250_000):
    _require_pyarrow()
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    for batch in pf.iter_batches(columns=columns, batch_size=batch_size):
        yield batch


# -----------------------------
# Generic helpers
# -----------------------------
def first_existing_column(available: Iterable[str], candidates: List[str]) -> Optional[str]:
    avail_map = {str(c).lower(): str(c) for c in available}
    for cand in candidates:
        key = str(cand).lower()
        if key in avail_map:
            return avail_map[key]
    return None


def parse_radii_m(raw: str) -> List[float]:
    out: List[float] = []
    s = (raw or "").strip()
    if not s:
        return out
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            val = float(part)
            if val > 0:
                out.append(val)
        except Exception:
            continue
    return sorted(set(out))


def write_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)


# -----------------------------
# Station exposure (year/month)
# -----------------------------
def _pick_time_col(cols: List[str]) -> Optional[str]:
    # common Citi Bike schemas
    candidates = [
        "started_at",
        "starttime",
        "start_time",
        "start_time_local",
        "Start Time",
        "start_datetime",
    ]
    return first_existing_column(cols, candidates)


def _pick_station_cols(cols: List[str]):
    start_id = first_existing_column(cols, ["start_station_id", "start station id", "Start Station ID"])
    start_name = first_existing_column(cols, ["start_station_name", "start station name", "Start Station Name"])
    end_id = first_existing_column(cols, ["end_station_id", "end station id", "End Station ID"])
    end_name = first_existing_column(cols, ["end_station_name", "end station name", "End Station Name"])

    # coords vary a lot across years
    start_lat = first_existing_column(cols, ["start_lat", "start station latitude", "start_station_latitude", "Start Station Latitude"])
    start_lng = first_existing_column(cols, ["start_lng", "start_lon", "start station longitude", "start_station_longitude", "Start Station Longitude"])
    end_lat = first_existing_column(cols, ["end_lat", "end station latitude", "end_station_latitude", "End Station Latitude"])
    end_lng = first_existing_column(cols, ["end_lng", "end_lon", "end station longitude", "end_station_longitude", "End Station Longitude"])

    return start_id, start_name, start_lat, start_lng, end_id, end_name, end_lat, end_lng


def apply_nyc_jc_station_hygiene(df_st: pd.DataFrame, mode_filter: Optional[str]) -> pd.DataFrame:
    """
    df_st must have:
      - mode, start_station_id, station_lat, station_lng
    """
    if df_st.empty:
        return df_st

    if not mode_filter:
        return df_st

    mode_filter = mode_filter.lower().strip()
    if mode_filter not in ("nyc", "jc"):
        return df_st

    # JC bounding box (approx)
    JC_LAT_MIN, JC_LAT_MAX = 40.6990, 40.7490
    JC_LNG_MIN, JC_LNG_MAX = -74.0778, -74.0278

    # prefix markers
    prefixes = ("jc", "hb")

    out = df_st.copy()

    # unify ids to str for prefix test; coords numeric for bbox
    out["_id_str"] = out["start_station_id"].astype(str).str.lower()
    out["_lat_temp"] = pd.to_numeric(out["station_lat"], errors="coerce")
    out["_lng_temp"] = pd.to_numeric(out["station_lng"], errors="coerce")

    in_jc_bbox = (
        (out["_lat_temp"] >= JC_LAT_MIN) &
        (out["_lat_temp"] <= JC_LAT_MAX) &
        (out["_lng_temp"] >= JC_LNG_MIN) &
        (out["_lng_temp"] <= JC_LNG_MAX)
    )
    has_prefix = out["_id_str"].str.startswith(prefixes)

    if mode_filter == "nyc":
        # drop JC-prefixed OR in JC bbox (conservative)
        out = out[~(has_prefix | in_jc_bbox)].copy()
    else:
        # JC: keep if prefix OR in JC bbox
        out = out[(has_prefix | in_jc_bbox)].copy()

    out = out.drop(columns=["_id_str", "_lat_temp", "_lng_temp"])
    return out


def compute_station_exposure(
    parquet_files: List[Path],
    mode_filter: Optional[str],
    exposure_mode: str,  # "touchpoints" or "starts"
    batch_size: int,
) -> pd.DataFrame:
    """
    Returns station exposure by (mode, year, month, start_station_id, start_station_name)
    with:
      - start_trips, end_trips, trips (= touchpoints or starts), touchpoints
      - station_lat, station_lng (mean of available coords from starts)
    """
    rows = []

    for fp in parquet_files:
        part = parse_partition(fp)
        if not part:
            continue
        mode, year, month = part

        if mode_filter and mode.lower() != mode_filter.lower():
            continue

        cols = parquet_available_columns(fp)
        tcol = _pick_time_col(cols)
        start_id, start_name, start_lat, start_lng, end_id, end_name, end_lat, end_lng = _pick_station_cols(cols)

        needed = [c for c in [start_id, start_name, end_id, end_name] if c]
        # coords optional
        if start_lat and start_lng:
            needed += [start_lat, start_lng]
        if end_lat and end_lng:
            needed += [end_lat, end_lng]

        if not start_id or not start_name or not end_id or not end_name:
            # If missing essential station cols, skip this parquet
            continue

        for batch in iter_parquet_batches(fp, columns=needed, batch_size=batch_size):
            df = batch.to_pandas()

            # starts
            st = df[[start_id, start_name]].copy()
            st.columns = ["station_id", "station_name"]
            st["start_trips"] = 1
            if start_lat and start_lng:
                st["station_lat"] = pd.to_numeric(df[start_lat], errors="coerce")
                st["station_lng"] = pd.to_numeric(df[start_lng], errors="coerce")
            else:
                st["station_lat"] = pd.NA
                st["station_lng"] = pd.NA

            # ends
            en = df[[end_id, end_name]].copy()
            en.columns = ["station_id", "station_name"]
            en["end_trips"] = 1
            if end_lat and end_lng:
                en["station_lat"] = pd.to_numeric(df[end_lat], errors="coerce")
                en["station_lng"] = pd.to_numeric(df[end_lng], errors="coerce")
            else:
                en["station_lat"] = pd.NA
                en["station_lng"] = pd.NA

            st["end_trips"] = 0
            en["start_trips"] = 0

            both = pd.concat([st, en], ignore_index=True)
            both["mode"] = mode
            both["year"] = year
            both["month"] = month

            rows.append(both)

    if not rows:
        return pd.DataFrame()

    out = pd.concat(rows, ignore_index=True)

    # aggregate
    g = out.groupby(["mode", "year", "month", "station_id", "station_name"], as_index=False).agg(
        start_trips=("start_trips", "sum"),
        end_trips=("end_trips", "sum"),
        station_lat=("station_lat", "mean"),
        station_lng=("station_lng", "mean"),
    )
    g["touchpoints"] = g["start_trips"] + g["end_trips"]

    if exposure_mode == "starts":
        g["trips"] = g["start_trips"]
    else:
        g["trips"] = g["touchpoints"]

    # rename to match your notebooks/scripts
    g = g.rename(columns={"station_id": "start_station_id", "station_name": "start_station_name"})

    # hygiene filter (NYC/JC)
    g = g.rename(columns={"start_station_id": "start_station_id", "start_station_name": "start_station_name"})
    g = g.rename(columns={"start_station_id": "start_station_id"})  # no-op (clarity)
    # apply hygiene expects 'start_station_id' and 'station_lat/lng'
    tmp = g.rename(columns={"start_station_id": "start_station_id"})
    tmp = tmp.rename(columns={"start_station_id": "start_station_id"})  # no-op
    # map for hygiene function
    hygienic = tmp.rename(columns={"start_station_id": "start_station_id"})
    hygienic = hygienic.rename(columns={"start_station_id": "start_station_id"})  # no-op
    # just use a consistent input
    hygienic = g.rename(columns={"start_station_id": "start_station_id"})
    hygienic = hygienic.rename(columns={"start_station_id": "start_station_id"})  # no-op

    hygienic = hygienic.rename(columns={"start_station_id": "start_station_id"})
    # Create expected column name for hygiene: start_station_id
    hygienic = hygienic.rename(columns={"start_station_id": "start_station_id"})
    hygienic = hygienic.rename(columns={"start_station_id": "start_station_id"})  # no-op
    # Hygiene expects columns: mode, start_station_id, station_lat, station_lng
    # We already have those columns (start_station_id, station_lat, station_lng).
    hygienic = apply_nyc_jc_station_hygiene(
        hygienic.rename(columns={"start_station_id": "start_station_id"}),  # no-op
        mode_filter=mode_filter,
    )

    return hygienic.reset_index(drop=True)


# -----------------------------
# DOW / Hour summaries (include YEAR)
# -----------------------------
def compute_dow_and_hour(
    parquet_files: List[Path],
    mode_filter: Optional[str],
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      df_dow: columns [mode, year, dow, trips]
      df_hr:  columns [mode, year, hour, trips]
    """
    dow_rows = []
    hr_rows = []

    for fp in parquet_files:
        part = parse_partition(fp)
        if not part:
            continue
        mode, year, month = part

        if mode_filter and mode.lower() != mode_filter.lower():
            continue

        cols = parquet_available_columns(fp)
        tcol = _pick_time_col(cols)
        if not tcol:
            continue

        for batch in iter_parquet_batches(fp, columns=[tcol], batch_size=batch_size):
            df = batch.to_pandas()
            ts = pd.to_datetime(df[tcol], errors="coerce")
            ts = ts.dropna()
            if ts.empty:
                continue

            dow = ts.dt.dayofweek  # Monday=0
            hr = ts.dt.hour

            dow_rows.append(
                pd.DataFrame({"mode": mode, "year": year, "dow": dow}).groupby(["mode", "year", "dow"], as_index=False).size()
            )
            hr_rows.append(
                pd.DataFrame({"mode": mode, "year": year, "hour": hr}).groupby(["mode", "year", "hour"], as_index=False).size()
            )

    if dow_rows:
        df_dow = pd.concat(dow_rows, ignore_index=True).groupby(["mode", "year", "dow"], as_index=False)["size"].sum()
        df_dow = df_dow.rename(columns={"size": "trips"}).sort_values(["mode", "year", "dow"]).reset_index(drop=True)
    else:
        df_dow = pd.DataFrame(columns=["mode", "year", "dow", "trips"])

    if hr_rows:
        df_hr = pd.concat(hr_rows, ignore_index=True).groupby(["mode", "year", "hour"], as_index=False)["size"].sum()
        df_hr = df_hr.rename(columns={"size": "trips"}).sort_values(["mode", "year", "hour"]).reset_index(drop=True)
    else:
        df_hr = pd.DataFrame(columns=["mode", "year", "hour", "trips"])

    return df_dow, df_hr


# -----------------------------
# Crash proximity (overall + by-year + by-year-month)
# -----------------------------
EARTH_RADIUS_M = 6371000.0


def _to_radians(lat_series: pd.Series, lng_series: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lat_rad": pd.to_numeric(lat_series, errors="coerce") * math.pi / 180.0,
            "lng_rad": pd.to_numeric(lng_series, errors="coerce") * math.pi / 180.0,
        }
    )


def _load_crashes_with_time(nypd_csv: Path) -> pd.DataFrame:
    crashes = pd.read_csv(nypd_csv, low_memory=False)

    # crash date column (filter_nypd_crashes.py typically writes crash_date)
    date_col = first_existing_column(crashes.columns, ["crash_date", "CRASH DATE", "CRASH_DATE"])
    if not date_col:
        raise RuntimeError(
            f"Could not find crash date column in NYPD file: {nypd_csv}\n"
            "Expected 'crash_date' or 'CRASH DATE'."
        )

    lat_col = first_existing_column(crashes.columns, ["LATITUDE", "latitude", "lat"])
    lng_col = first_existing_column(crashes.columns, ["LONGITUDE", "longitude", "lng", "lon"])
    if not lat_col or not lng_col:
        raise RuntimeError(
            f"Could not find crash lat/lng columns in NYPD file: {nypd_csv}\n"
            f"Columns seen (first 60): {list(crashes.columns)[:60]}"
        )

    out = crashes[[date_col, lat_col, lng_col]].copy()
    out = out.rename(columns={date_col: "crash_date", lat_col: "latitude", lng_col: "longitude"})
    out["crash_date"] = pd.to_datetime(out["crash_date"], errors="coerce")
    out["latitude"] = pd.to_numeric(out["latitude"], errors="coerce")
    out["longitude"] = pd.to_numeric(out["longitude"], errors="coerce")
    out = out.dropna(subset=["crash_date", "latitude", "longitude"]).reset_index(drop=True)
    out["year"] = out["crash_date"].dt.year.astype(int)
    out["month"] = out["crash_date"].dt.month.astype(int)
    return out


def _compute_crash_counts_for_stations(
    stations: pd.DataFrame,
    crashes: pd.DataFrame,
    radii_m: List[float],
) -> pd.DataFrame:
    """
    stations must include station_lat/station_lng
    crashes must include latitude/longitude
    """
    try:
        from sklearn.neighbors import BallTree
    except Exception as e:
        raise RuntimeError(
            "scikit-learn is required for crash proximity (BallTree).\n"
            "Install:\n"
            "  .venv/bin/pip install scikit-learn\n"
        ) from e

    st = stations.copy()
    st["station_lat"] = pd.to_numeric(st["station_lat"], errors="coerce")
    st["station_lng"] = pd.to_numeric(st["station_lng"], errors="coerce")
    st = st.dropna(subset=["station_lat", "station_lng"]).reset_index(drop=True)

    for r_m in radii_m:
        st[f"crashes_within_{int(r_m)}m"] = 0

    if st.empty:
        return st

    if crashes.empty:
        return st

    crash_rad = _to_radians(crashes["latitude"], crashes["longitude"])[["lat_rad", "lng_rad"]].values
    tree = BallTree(crash_rad, metric="haversine")

    st_rad = _to_radians(st["station_lat"], st["station_lng"])[["lat_rad", "lng_rad"]].values

    for r_m in radii_m:
        r_rad = float(r_m) / EARTH_RADIUS_M
        counts = tree.query_radius(st_rad, r=r_rad, count_only=True)
        st[f"crashes_within_{int(r_m)}m"] = counts.astype(int)

    return st


def add_rate_cols(df: pd.DataFrame, radii_m: List[float]) -> pd.DataFrame:
    out = df.copy()
    out["trips"] = pd.to_numeric(out["trips"], errors="coerce")
    out = out.dropna(subset=["trips"])
    out = out[out["trips"] > 0].copy()
    for r_m in radii_m:
        col = f"crashes_within_{int(r_m)}m"
        if col not in out.columns:
            out[col] = 0
        out[f"{col}_per_100k_trips"] = (pd.to_numeric(out[col], errors="coerce").fillna(0) / out["trips"]) * 100000.0
    return out


# -----------------------------
# Highlights
# -----------------------------
def build_highlights_markdown(
    df_year: pd.DataFrame,
    df_month: pd.DataFrame,
    df_dow: pd.DataFrame,
    df_hr: pd.DataFrame,
    mode: Optional[str],
    top_n: int,
) -> List[str]:
    lines: List[str] = []
    lines.append("# Summary highlights")
    lines.append("")
    if mode:
        lines.append(f"- Mode: **{mode}**")
    lines.append("")

    # Year
    if not df_year.empty:
        best = df_year.sort_values("trips", ascending=False).head(top_n)
        lines.append(f"## Top {top_n} years by trips")
        for _, r in best.iterrows():
            lines.append(f"- {int(r['year'])}: {int(r['trips'])}")
        lines.append("")

    # Month (top 10 across all)
    if not df_month.empty:
        bestm = df_month.sort_values("trips", ascending=False).head(10)
        lines.append("## Top 10 year-months by trips")
        for _, r in bestm.iterrows():
            lines.append(f"- {int(r['year'])}-{int(r['month']):02d}: {int(r['trips'])}")
        lines.append("")

    # DOW
    if not df_dow.empty:
        bestd = df_dow.groupby("dow", as_index=False)["trips"].sum().sort_values("trips", ascending=False).head(top_n)
        lines.append(f"## Top {top_n} days-of-week by trips (all years combined)")
        for _, r in bestd.iterrows():
            lines.append(f"- dow={int(r['dow'])}: {int(r['trips'])}")
        lines.append("")

    # Hour
    if not df_hr.empty:
        besth = df_hr.groupby("hour", as_index=False)["trips"].sum().sort_values("trips", ascending=False).head(top_n)
        lines.append(f"## Top {top_n} hours by trips (all years combined)")
        for _, r in besth.iterrows():
            lines.append(f"- hour={int(r['hour'])}: {int(r['trips'])}")
        lines.append("")

    return lines


def write_markdown(lines: List[str], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarize Citi Bike usage from partitioned parquet files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parquet-dir", required=True, help="Root of run parquet, e.g. data/processed/citibike_parquet/<RUN_TAG>")
    ap.add_argument("--out-dir", required=True, help="Where to write CSV outputs (e.g. summaries/<RUN_TAG>)")
    ap.add_argument("--mode", default=None, help="Optional mode filter, e.g. jc or nyc")

    ap.add_argument("--nypd-crash-csv", default=None, help="Optional NYPD crash CSV (NYC-only) to compute station proximity risk")
    ap.add_argument("--radii-m", default="250,500", help="Crash radii in meters (comma-separated), default 250,500")

    ap.add_argument(
        "--station-exposure",
        choices=["touchpoints", "starts"],
        default="touchpoints",
        help="Exposure definition for citibike_station_exposure.csv: touchpoints=start+end (default), starts=start-only.",
    )

    ap.add_argument("--top-n", type=int, default=5, help="Top N for highlights (months always prints top 10)")
    ap.add_argument("--no-print", action="store_true", help="Disable printing highlights to terminal")
    ap.add_argument("--no-highlights", action="store_true", help="Do not write summary_highlights.md")
    ap.add_argument("--highlights-md", default=None, help="Path for highlights markdown. Default: <out-dir>/summary_highlights.md")

    ap.add_argument("--batch-size", type=int, default=250_000, help="Parquet scan batch size (default: 250k rows)")

    args = ap.parse_args()

    parquet_dir = Path(args.parquet_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(parquet_dir.glob("mode=*/year=*/month=*/tripdata.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found under: {parquet_dir}")

    mode_filter = str(args.mode).strip().lower() if args.mode else None
    radii_f = parse_radii_m(args.radii_m)

    # -----------------------------
    # Phase 1: trips by month/year (metadata)
    # -----------------------------
    month_rows = []
    for fp in files:
        part = parse_partition(fp)
        if not part:
            continue
        mode, year, month = part
        if mode_filter and mode.lower() != mode_filter:
            continue
        n = parquet_num_rows(fp)
        month_rows.append({"mode": mode, "year": year, "month": month, "trips": n})

    df_month = pd.DataFrame(month_rows)
    if df_month.empty:
        df_month = pd.DataFrame(columns=["mode", "year", "month", "trips"])
    else:
        df_month = df_month.groupby(["mode", "year", "month"], as_index=False)["trips"].sum()
        df_month = df_month.sort_values(["mode", "year", "month"]).reset_index(drop=True)

    df_year = (
        df_month.groupby(["mode", "year"], as_index=False)["trips"].sum()
        if not df_month.empty
        else pd.DataFrame(columns=["mode", "year", "trips"])
    )

    write_csv(df_month, out_dir / "citibike_trips_by_month.csv")
    write_csv(df_year, out_dir / "citibike_trips_by_year.csv")

    # -----------------------------
    # Phase 2: DOW / Hour
    # -----------------------------
    df_dow, df_hr = compute_dow_and_hour(files, mode_filter=mode_filter, batch_size=int(args.batch_size))
    write_csv(df_dow, out_dir / "citibike_trips_by_dow.csv")
    write_csv(df_hr, out_dir / "citibike_trips_by_hour.csv")

    # -----------------------------
    # Phase 3: station exposure (year/month)
    # -----------------------------
    df_st = compute_station_exposure(
        parquet_files=files,
        mode_filter=mode_filter,
        exposure_mode=str(args.station_exposure),
        batch_size=int(args.batch_size),
    )

    if df_st.empty:
        print("WARNING: Station exposure is empty (missing station columns in parquet?).")
        df_st = pd.DataFrame(columns=[
            "mode", "year", "month",
            "start_station_id", "start_station_name",
            "start_trips", "end_trips", "touchpoints", "trips",
            "station_lat", "station_lng",
        ])

    write_csv(df_st, out_dir / "citibike_station_exposure.csv")

    # -----------------------------
    # Phase 4: crash proximity risk CSVs (optional; NYC-only)
    # -----------------------------
    if args.nypd_crash_csv and radii_f:
        nypd_path = Path(args.nypd_crash_csv)

        # If JC mode: proxy unavailable -> zeros for all 3 outputs
        if mode_filter == "jc":
            # overall
            df_st_agg = df_st.groupby(["mode", "start_station_id", "start_station_name"], as_index=False).agg(
                trips=("trips", "sum"),
                start_trips=("start_trips", "sum"),
                end_trips=("end_trips", "sum"),
                touchpoints=("touchpoints", "sum"),
                station_lat=("station_lat", "mean"),
                station_lng=("station_lng", "mean"),
                year_min=("year", "min"),
                year_max=("year", "max"),
                years_count=("year", "nunique"),
                months_count=("month", "nunique"),
            )
            df_st_agg["data_quality"] = "lifetime_aggregate"
            df_st_agg["temporal_coverage"] = (
                df_st_agg["months_count"].astype(str) + " months spanning " +
                df_st_agg["year_min"].astype(str) + "-" + df_st_agg["year_max"].astype(str)
            )
            df_risk = df_st_agg.copy()
            for r_m in radii_f:
                df_risk[f"crashes_within_{int(r_m)}m"] = 0
            df_risk = add_rate_cols(df_risk, radii_f)
            df_risk["risk_proxy_available"] = 0
            write_csv(df_risk, out_dir / "station_risk_exposure_plus_crashproximity.csv")

            # by year
            df_yr = df_st.groupby(["mode", "start_station_id", "start_station_name", "year"], as_index=False).agg(
                trips=("trips", "sum"),
                start_trips=("start_trips", "sum"),
                end_trips=("end_trips", "sum"),
                touchpoints=("touchpoints", "sum"),
                station_lat=("station_lat", "mean"),
                station_lng=("station_lng", "mean"),
            )
            for r_m in radii_f:
                df_yr[f"crashes_within_{int(r_m)}m"] = 0
            df_yr = add_rate_cols(df_yr, radii_f)
            df_yr["risk_proxy_available"] = 0
            df_yr["data_quality"] = "yearly"
            write_csv(df_yr, out_dir / "station_risk_exposure_plus_crashproximity_by_year.csv")

            # by year-month
            df_ym = df_st.groupby(["mode", "start_station_id", "start_station_name", "year", "month"], as_index=False).agg(
                trips=("trips", "sum"),
                start_trips=("start_trips", "sum"),
                end_trips=("end_trips", "sum"),
                touchpoints=("touchpoints", "sum"),
                station_lat=("station_lat", "mean"),
                station_lng=("station_lng", "mean"),
            )
            for r_m in radii_f:
                df_ym[f"crashes_within_{int(r_m)}m"] = 0
            df_ym = add_rate_cols(df_ym, radii_f)
            df_ym["risk_proxy_available"] = 0
            df_ym["data_quality"] = "year_month"
            write_csv(df_ym, out_dir / "station_risk_exposure_plus_crashproximity_by_year_month.csv")

            print("Saved crash-proxy CSVs (JC mode: proxy unavailable; wrote zeros).")

        else:
            # NYC mode: compute if file exists
            if not nypd_path.exists():
                print(f"WARNING: --nypd-crash-csv not found: {nypd_path} (skipping crash proximity outputs)")
            else:
                try:
                    crashes_all = _load_crashes_with_time(nypd_path)

                    # ---------- overall (keep existing idea: lifetime aggregate) ----------
                    df_st_sorted = df_st.sort_values(
                        ["mode", "start_station_id", "start_station_name", "year", "month"]
                    ).reset_index(drop=True)

                    df_st_agg = df_st_sorted.groupby(["mode", "start_station_id", "start_station_name"], as_index=False).agg(
                        trips=("trips", "sum"),
                        start_trips=("start_trips", "sum"),
                        end_trips=("end_trips", "sum"),
                        touchpoints=("touchpoints", "sum"),
                        station_lat=("station_lat", "mean"),
                        station_lng=("station_lng", "mean"),
                        year_min=("year", "min"),
                        year_max=("year", "max"),
                        years_count=("year", "nunique"),
                        months_count=("month", "nunique"),
                        first_month=("month", lambda x: x.iloc[0]),
                        last_month=("month", lambda x: x.iloc[-1]),
                    )
                    df_st_agg["observation_months"] = (
                        (df_st_agg["year_max"] - df_st_agg["year_min"]) * 12 +
                        (df_st_agg["last_month"] - df_st_agg["first_month"]) + 1
                    )
                    df_st_agg["data_quality"] = "lifetime_aggregate"
                    df_st_agg["temporal_coverage"] = (
                        df_st_agg["months_count"].astype(str) + " months spanning " +
                        df_st_agg["year_min"].astype(str) + "-" + df_st_agg["year_max"].astype(str)
                    )

                    df_risk_overall = _compute_crash_counts_for_stations(
                        stations=df_st_agg,
                        crashes=crashes_all,          # all crashes
                        radii_m=radii_f,
                    )
                    df_risk_overall = add_rate_cols(df_risk_overall, radii_f)
                    df_risk_overall["risk_proxy_available"] = 1

                    # nice sort
                    sort_col = f"crashes_within_{int(max(radii_f))}m_per_100k_trips"
                    if sort_col in df_risk_overall.columns:
                        df_risk_overall = df_risk_overall.sort_values([sort_col, "trips"], ascending=[False, False]).reset_index(drop=True)

                    write_csv(df_risk_overall, out_dir / "station_risk_exposure_plus_crashproximity.csv")
                    print(f"Saved -> {out_dir / 'station_risk_exposure_plus_crashproximity.csv'} (overall / lifetime aggregate)")

                    # ---------- by year ----------
                    df_st_year = df_st.groupby(["mode", "start_station_id", "start_station_name", "year"], as_index=False).agg(
                        trips=("trips", "sum"),
                        start_trips=("start_trips", "sum"),
                        end_trips=("end_trips", "sum"),
                        touchpoints=("touchpoints", "sum"),
                        station_lat=("station_lat", "mean"),
                        station_lng=("station_lng", "mean"),
                        months_covered=("month", "nunique"),
                    )
                    yearly_parts = []
                    for yy in sorted(df_st_year["year"].dropna().unique()):
                        st_y = df_st_year[df_st_year["year"] == yy].copy()
                        cr_y = crashes_all[crashes_all["year"] == int(yy)].copy()
                        st_y = _compute_crash_counts_for_stations(st_y, cr_y, radii_f)
                        yearly_parts.append(st_y)

                    df_risk_year = pd.concat(yearly_parts, ignore_index=True) if yearly_parts else df_st_year.copy()
                    df_risk_year = add_rate_cols(df_risk_year, radii_f)
                    df_risk_year["risk_proxy_available"] = 1
                    df_risk_year["data_quality"] = "yearly"
                    write_csv(df_risk_year, out_dir / "station_risk_exposure_plus_crashproximity_by_year.csv")
                    print(f"Saved -> {out_dir / 'station_risk_exposure_plus_crashproximity_by_year.csv'}")

                    # ---------- by year-month ----------
                    df_st_ym = df_st.groupby(["mode", "start_station_id", "start_station_name", "year", "month"], as_index=False).agg(
                        trips=("trips", "sum"),
                        start_trips=("start_trips", "sum"),
                        end_trips=("end_trips", "sum"),
                        touchpoints=("touchpoints", "sum"),
                        station_lat=("station_lat", "mean"),
                        station_lng=("station_lng", "mean"),
                    )
                    ym_parts = []
                    keys = df_st_ym[["year", "month"]].dropna().drop_duplicates().sort_values(["year", "month"]).itertuples(index=False, name=None)
                    for (yy, mm) in keys:
                        st_m = df_st_ym[(df_st_ym["year"] == int(yy)) & (df_st_ym["month"] == int(mm))].copy()
                        cr_m = crashes_all[(crashes_all["year"] == int(yy)) & (crashes_all["month"] == int(mm))].copy()
                        st_m = _compute_crash_counts_for_stations(st_m, cr_m, radii_f)
                        ym_parts.append(st_m)

                    df_risk_ym = pd.concat(ym_parts, ignore_index=True) if ym_parts else df_st_ym.copy()
                    df_risk_ym = add_rate_cols(df_risk_ym, radii_f)
                    df_risk_ym["risk_proxy_available"] = 1
                    df_risk_ym["data_quality"] = "year_month"
                    write_csv(df_risk_ym, out_dir / "station_risk_exposure_plus_crashproximity_by_year_month.csv")
                    print(f"Saved -> {out_dir / 'station_risk_exposure_plus_crashproximity_by_year_month.csv'}")

                except Exception as e:
                    print(f"WARNING: Crash proximity skipped due to error: {e}")
    else:
        # still helpful to write empty by-year files? we keep it simple: do nothing if not requested
        pass

    # -----------------------------
    # Highlights
    # -----------------------------
    highlights_lines = build_highlights_markdown(
        df_year=df_year,
        df_month=df_month,
        df_dow=df_dow,
        df_hr=df_hr,
        mode=mode_filter,
        top_n=int(args.top_n),
    )

    if not args.no_print:
        for ln in highlights_lines:
            # nicer terminal output
            print(ln.replace("# ", "").replace("## ", "").replace("**", ""))

    if not args.no_highlights:
        out_md = Path(args.highlights_md) if args.highlights_md else (out_dir / "summary_highlights.md")
        write_markdown(highlights_lines, out_md)


if __name__ == "__main__":
    main()
