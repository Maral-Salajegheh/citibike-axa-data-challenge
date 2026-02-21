#!/usr/bin/env python3
"""
ingest_tripdata.py

Robust Citi Bike tripdata ingestion (ZIP -> Parquet) supporting:
- NYC monthly ZIPs: YYYYMM-citibike-tripdata(.csv)?.zip
- NYC yearly  ZIPs: YYYY-citibike-tripdata(.csv)?.zip (may contain CSV(s) or nested ZIP(s))
- JC  monthly ZIPs: JC-YYYYMM-citibike-tripdata(.csv)?.zip
- JC  yearly  ZIPs: JC-YYYY-citibike-tripdata(.csv)?.zip (rare, but supported)

FIXES INCLUDED (copy/paste safe):
1) CSV parsing bug fix:
   - The prior approach used pyarrow to "peek" the header, which advanced the ZipExtFile stream.
     The second open_csv then started mid-file, causing schema/column-count errors like:
       "Expected 15 columns, got 9".
   - This version reads the header line safely, rewinds (or buffers), and uses pyarrow with skip_rows=1.

2) Legacy header mapping fix:
   - Legacy monthly files often have headers like "Trip Duration", "Start Time", "User Type",
     "Start Station Latitude", etc.
   - After standardization, those become "trip_duration", "start_time", "user_type",
     "start_station_latitude", etc.
   - This version maps BOTH legacy variants and modern variants into one canonical schema.

3) Station filtering bug fix:
   - pc.starts_with must receive a Python str, not pa.scalar(prefix) (pyarrow compatibility).

4) Station filtering strategy improvement for JC:
   - Early JC data uses numeric station IDs (e.g., 3185) and does NOT use "JC"/"HB" prefixes.
     A strict prefix-based filter would drop ~100% of JC rows.
   - This version uses:
       - Prefix filtering when IDs are prefixed (JC/HB), AND
       - A Jersey City / Hoboken bounding-box fallback when IDs are numeric.
     This keeps JC legacy months and still blocks obvious cross-system contamination.

Canonical output columns:
  ride_id (string, generated for legacy if missing)
  rideable_type (string)
  started_at (timestamp[us])
  ended_at   (timestamp[us])
  start_station_id (string)
  start_station_name (string)
  end_station_id (string)
  end_station_name (string)
  start_lat (string)
  start_lng (string)
  end_lat   (string)
  end_lng   (string)
  member_casual (string; mapped from legacy usertype if needed)
  usertype (string; legacy raw)
  bike_id  (string)
  tripduration_seconds (string; legacy raw kept as string to avoid cast failures)
  gender (string)
  birth_year (string)

Notes:
- All CSV fields are forced to string at read-time for robustness.
- Timestamp parsing is done after read.
- Output is written as one Parquet file per (mode, year, month) partition.
- Default station-id policy is `name_coord` to reduce cross-year id drift for the same station.
"""

from __future__ import annotations

import argparse
import csv
import faulthandler
import gc
import hashlib
import io
import math
import os
import re
import shutil
import sys
import tempfile
import zipfile
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# ============================
# STATION ID FILTERING
# ============================

# Jersey City and Hoboken station prefixes (modern-ish IDs)
JC_STATION_PREFIXES = ("JC", "HB")

# Bounding boxes (approximate) used when IDs are numeric / non-prefixed.
# These are intentionally conservative and focus on JC/Hoboken.
JC_BBOX_LAT_MIN = 40.66
JC_BBOX_LAT_MAX = 40.78
JC_BBOX_LNG_MIN = -74.12
JC_BBOX_LNG_MAX = -74.00

# NYC bbox (used only if you later want bbox filtering for NYC; kept here as optional)
NYC_BBOX_LAT_MIN = 40.55
NYC_BBOX_LAT_MAX = 40.93
NYC_BBOX_LNG_MIN = -74.10
NYC_BBOX_LNG_MAX = -73.70

# Optional station-id remapping loaded from --station-id-crosswalk
# Shape: {"any": {old: new}, "nyc": {...}, "jc": {...}}
STATION_ID_CROSSWALK: dict[str, dict[str, str]] = {"any": {}, "nyc": {}, "jc": {}}
STATION_ID_POLICY_ALIASES = {
    "raw": "raw",
    "name_coord": "name_coord",
    "canonical_name": "name_coord",
}
STATION_ID_POLICIES = tuple(STATION_ID_POLICY_ALIASES.keys())
NORMALIZED_STATION_ID_POLICIES = tuple(sorted(set(STATION_ID_POLICY_ALIASES.values())))
STATION_COORD_ROUND_DECIMALS = 4
STATION_COORD_MERGE_THRESHOLD_M = 50.0
STATION_OUTLIER_REPORT_TOP_N = 30

# Stateful cluster cache used by name_coord policy during a single ingestion run.
# shape: mode -> normalized_station_name -> [cluster_dict, ...]
STATION_NAME_COORD_CLUSTERS: dict[str, dict[str, list[dict[str, object]]]] = {
    "any": {},
    "nyc": {},
    "jc": {},
}

_STATION_ID_PREFIX_RE = re.compile(r"^(JC|HB)[\s\-_]+", re.IGNORECASE)
_STATION_ID_NUMERIC_RE = re.compile(r"^(?P<prefix>[A-Z]+)?(?P<intpart>\d+)(?:\.(?P<frac>\d+))?$")
_STATION_NAME_MULTI_SPACE_RE = re.compile(r"\s+")
_STATION_NAME_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")


def _normalize_station_id_scalar(station_id: str | None) -> str | None:
    if station_id is None:
        return None

    s = str(station_id).strip().upper()
    if s in ("", "NONE", "NULL", "NAN"):
        return None

    # Normalize prefixed styles: JC-123 / JC_123 / JC 123 -> JC123
    s = _STATION_ID_PREFIX_RE.sub(r"\1", s)

    m = _STATION_ID_NUMERIC_RE.fullmatch(s)
    if not m:
        return s

    prefix = m.group("prefix") or ""
    intpart = str(int(m.group("intpart")))
    frac = m.group("frac")
    if not frac:
        return f"{prefix}{intpart}"

    frac = frac.rstrip("0")
    if frac == "":
        return f"{prefix}{intpart}"
    return f"{prefix}{intpart}.{frac}"


def _normalize_station_id_policy(station_id_policy: str) -> str:
    key = station_id_policy.strip().lower()
    normalized = STATION_ID_POLICY_ALIASES.get(key)
    if normalized is None:
        raise ValueError(
            f"Invalid station id policy: '{station_id_policy}'. "
            f"Must be one of: {', '.join(STATION_ID_POLICIES)}"
        )
    return normalized


def _crosswalk_for_mode(mode: str) -> dict[str, str]:
    merged = dict(STATION_ID_CROSSWALK.get("any", {}))
    merged.update(STATION_ID_CROSSWALK.get(mode, {}))
    return merged


def _normalize_station_name_for_key(name: str | None) -> str | None:
    if name is None:
        return None

    s = str(name).strip().lower()
    if not s:
        return None

    s = _STATION_NAME_PUNCT_RE.sub(" ", s)
    s = _STATION_NAME_MULTI_SPACE_RE.sub(" ", s).strip()
    return s or None


def _normalize_coord_for_key(coord: str | None) -> str | None:
    if coord is None:
        return None

    s = str(coord).strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return None

    s = s.replace(",", "")
    try:
        value = float(s)
    except Exception:
        return None

    if not math.isfinite(value):
        return None

    rounded = round(value, STATION_COORD_ROUND_DECIMALS)
    return f"{rounded:.{STATION_COORD_ROUND_DECIMALS}f}"


def _parse_coord_float(coord: str | None) -> float | None:
    if coord is None:
        return None

    s = str(coord).strip()
    if s == "" or s.lower() in {"none", "null", "nan"}:
        return None

    s = s.replace(",", "")
    try:
        value = float(s)
    except Exception:
        return None
    if not math.isfinite(value):
        return None
    return value


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    # Great-circle distance in meters.
    r = 6_371_000.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dlng / 2.0) ** 2
    )
    c = 2.0 * math.asin(min(1.0, math.sqrt(a)))
    return r * c


def _canonical_station_id_prefix(mode: str) -> str:
    if mode == "nyc":
        return "NYCSTN"
    if mode == "jc":
        return "JCSTN"
    return "STN"


def _clusters_for_mode(mode: str) -> dict[str, list[dict[str, object]]]:
    if mode in ("nyc", "jc", "any"):
        return STATION_NAME_COORD_CLUSTERS[mode]
    return STATION_NAME_COORD_CLUSTERS["any"]


def _make_station_cluster_id(mode: str, name_key: str, lat: float, lng: float) -> str:
    prefix = _canonical_station_id_prefix(mode)
    digest = hashlib.sha1(f"{mode}|{name_key}|{lat:.6f}|{lng:.6f}".encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"


def _assign_clustered_station_id(
    mode: str,
    raw_name: str | None,
    name_key: str,
    lat: float,
    lng: float,
    base_id: str | None,
) -> str:
    clusters_by_name = _clusters_for_mode(mode)
    clusters = clusters_by_name.setdefault(name_key, [])

    if not clusters:
        new_id = _make_station_cluster_id(mode, name_key, lat, lng)
        raw_ids = set()
        if base_id:
            raw_ids.add(base_id)
        clusters.append(
            {
                "lat": lat,
                "lng": lng,
                "count": 1,
                "canonical_id": new_id,
                "raw_ids": raw_ids,
                "display_name": (str(raw_name).strip() if raw_name is not None else name_key),
            }
        )
        return new_id

    best_idx = 0
    best_dist = float("inf")
    for idx, c in enumerate(clusters):
        d = _haversine_m(
            float(c["lat"]),
            float(c["lng"]),
            lat,
            lng,
        )
        if d < best_dist:
            best_dist = d
            best_idx = idx

    if best_dist <= STATION_COORD_MERGE_THRESHOLD_M:
        c = clusters[best_idx]
        cnt = int(c["count"])
        c["lat"] = (float(c["lat"]) * cnt + lat) / (cnt + 1)
        c["lng"] = (float(c["lng"]) * cnt + lng) / (cnt + 1)
        c["count"] = cnt + 1
        if base_id:
            c_raw_ids = c.get("raw_ids")
            if isinstance(c_raw_ids, set):
                c_raw_ids.add(base_id)
        return str(c["canonical_id"])

    # Outlier branch: same normalized name, but farther than threshold from every existing cluster.
    new_id = _make_station_cluster_id(mode, name_key, lat, lng)
    raw_ids = set()
    if base_id:
        raw_ids.add(base_id)
    clusters.append(
        {
            "lat": lat,
            "lng": lng,
            "count": 1,
            "canonical_id": new_id,
            "raw_ids": raw_ids,
            "display_name": (str(raw_name).strip() if raw_name is not None else name_key),
        }
    )
    return new_id


def _station_coord_outliers_for_mode(mode: str) -> list[dict[str, object]]:
    mode_clusters = _clusters_for_mode(mode)
    out: list[dict[str, object]] = []

    for name_key, clusters in mode_clusters.items():
        if len(clusters) <= 1:
            continue

        max_sep = 0.0
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                ci = clusters[i]
                cj = clusters[j]
                d = _haversine_m(
                    float(ci["lat"]),
                    float(ci["lng"]),
                    float(cj["lat"]),
                    float(cj["lng"]),
                )
                if d > max_sep:
                    max_sep = d

        if max_sep <= STATION_COORD_MERGE_THRESHOLD_M:
            continue

        raw_ids: set[str] = set()
        display_name = name_key
        for c in clusters:
            c_raw_ids = c.get("raw_ids")
            if isinstance(c_raw_ids, set):
                for rid in c_raw_ids:
                    if rid:
                        raw_ids.add(str(rid))
            disp = c.get("display_name")
            if isinstance(disp, str) and disp.strip():
                display_name = disp.strip()

        # User explicitly asked to report outliers with different station ids.
        if len(raw_ids) <= 1:
            continue

        out.append(
            {
                "name": display_name,
                "name_key": name_key,
                "max_sep_m": max_sep,
                "cluster_count": len(clusters),
                "raw_id_count": len(raw_ids),
                "sample_ids": sorted(raw_ids)[:5],
            }
        )

    out.sort(key=lambda r: float(r["max_sep_m"]), reverse=True)
    return out


def _print_station_coord_outlier_report(mode: str) -> None:
    rows = _station_coord_outliers_for_mode(mode)
    print()
    print("=" * 70)
    print(f"STATION COORD OUTLIERS > {STATION_COORD_MERGE_THRESHOLD_M:.0f}m (mode={mode})")
    print("=" * 70)

    if not rows:
        print("✓ No >50m outlier station names with different station IDs.")
        return

    print(
        f"Found {len(rows):,} station name(s) with >{STATION_COORD_MERGE_THRESHOLD_M:.0f}m "
        "coordinate spread and multiple station IDs."
    )
    limit = min(STATION_OUTLIER_REPORT_TOP_N, len(rows))
    for r in rows[:limit]:
        sample_ids = ",".join(str(x) for x in r["sample_ids"])
        print(
            " - "
            f"{r['name']} | max_sep={float(r['max_sep_m']):.1f}m | "
            f"clusters={int(r['cluster_count'])} | raw_ids={int(r['raw_id_count'])} | "
            f"sample_ids=[{sample_ids}]"
        )
    if len(rows) > limit:
        print(f" ... and {len(rows) - limit:,} more.")


def _reset_station_name_coord_clusters() -> None:
    for key in STATION_NAME_COORD_CLUSTERS:
        STATION_NAME_COORD_CLUSTERS[key].clear()


def _normalize_station_id_array(
    arr: pa.ChunkedArray, mode: str, apply_crosswalk: bool = True
) -> pa.ChunkedArray:
    if not (pa.types.is_string(arr.type) or pa.types.is_large_string(arr.type)):
        arr = pc.cast(arr, pa.string(), safe=False)

    mode_map = _crosswalk_for_mode(mode) if apply_crosswalk else {}
    out_chunks: list[pa.Array] = []

    for chunk in arr.chunks:
        normalized_vals: list[str | None] = []
        for i in range(len(chunk)):
            scalar = chunk[i]
            raw = scalar.as_py() if scalar.is_valid else None
            norm = _normalize_station_id_scalar(raw)
            if norm is not None and mode_map:
                norm = mode_map.get(norm, norm)
            normalized_vals.append(norm)
        out_chunks.append(pa.array(normalized_vals, type=pa.string()))

    return pa.chunked_array(out_chunks, type=pa.string())


def _canonicalize_station_ids_from_identity(
    station_ids: pa.ChunkedArray,
    station_names: pa.ChunkedArray,
    station_lats: pa.ChunkedArray,
    station_lngs: pa.ChunkedArray,
    mode: str,
) -> pa.ChunkedArray:
    id_values = pc.cast(station_ids, pa.string(), safe=False).combine_chunks().to_pylist()
    name_values = pc.cast(station_names, pa.string(), safe=False).combine_chunks().to_pylist()
    lat_values = pc.cast(station_lats, pa.string(), safe=False).combine_chunks().to_pylist()
    lng_values = pc.cast(station_lngs, pa.string(), safe=False).combine_chunks().to_pylist()

    mode_map = _crosswalk_for_mode(mode)
    out: list[str | None] = []

    for raw_id, raw_name, raw_lat, raw_lng in zip(id_values, name_values, lat_values, lng_values):
        base_id = _normalize_station_id_scalar(raw_id)

        # Explicit crosswalk mappings win over generated canonical ids.
        if base_id is not None and mode_map:
            mapped = mode_map.get(base_id)
            if mapped is not None:
                out.append(mapped)
                continue

        name_key = _normalize_station_name_for_key(raw_name)
        lat_f = _parse_coord_float(raw_lat)
        lng_f = _parse_coord_float(raw_lng)
        if name_key is None or lat_f is None or lng_f is None:
            out.append(base_id)
            continue

        canonical_id = _assign_clustered_station_id(
            mode=mode,
            raw_name=(str(raw_name) if raw_name is not None else None),
            name_key=name_key,
            lat=lat_f,
            lng=lng_f,
            base_id=base_id,
        )
        out.append(canonical_id)

    return pa.chunked_array([pa.array(out, type=pa.string())], type=pa.string())


def _load_station_id_crosswalk(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Station-id crosswalk not found: {path}")

    def norm_header(h: str) -> str:
        h = (h or "").strip().lower()
        h = re.sub(r"[^a-z0-9]+", "_", h).strip("_")
        return h

    def first_key(headers: dict[str, str], options: list[str]) -> str | None:
        for opt in options:
            if opt in headers:
                return headers[opt]
        return None

    out: dict[str, dict[str, str]] = {"any": {}, "nyc": {}, "jc": {}}

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Crosswalk has no header row: {path}")

        header_map = {norm_header(h): h for h in reader.fieldnames}
        old_key = first_key(header_map, ["old_id", "from_id", "legacy_id", "source_id"])
        new_key = first_key(header_map, ["new_id", "to_id", "canonical_id", "target_id"])
        mode_key = first_key(header_map, ["mode"])

        if old_key is None or new_key is None:
            raise ValueError(
                "Crosswalk must contain old/new id columns (e.g. old_id,new_id or from_id,to_id)"
            )

        for row in reader:
            old_raw = row.get(old_key)
            new_raw = row.get(new_key)
            old_id = _normalize_station_id_scalar(old_raw)
            new_id = _normalize_station_id_scalar(new_raw)
            if old_id is None or new_id is None:
                continue

            mode_val = "any"
            if mode_key is not None:
                mode_val = str(row.get(mode_key, "any") or "any").strip().lower()
                if mode_val not in ("any", "nyc", "jc"):
                    mode_val = "any"

            out[mode_val][old_id] = new_id

    return out


def _should_keep_station_id(station_id: str, mode: str) -> bool:
    if not station_id or str(station_id).strip() == "":
        return True

    station_id_upper = str(station_id).strip().upper()
    is_jc_station = any(station_id_upper.startswith(prefix) for prefix in JC_STATION_PREFIXES)

    if mode == "nyc":
        return not is_jc_station
    if mode == "jc":
        return is_jc_station
    return True


def _bbox_mask_for_mode(tbl: pa.Table, mode: str) -> pa.ChunkedArray | None:
    """
    Bounding-box-based mask for cases where station_id prefixes are not usable (e.g., JC legacy numeric IDs).
    Uses canonical columns start_lat/start_lng/end_lat/end_lng if present.

    Returns:
      A boolean ChunkedArray mask, or None if bbox cannot be computed.
    """
    if tbl.num_rows == 0:
        return None

    needed = {"start_lat", "start_lng", "end_lat", "end_lng"}
    if not needed.issubset(set(tbl.column_names)):
        return None

    # Cast string coords -> float64 (invalid -> null)
    def to_f64(colname: str) -> pa.ChunkedArray:
        arr = tbl[colname]
        if not (
            pa.types.is_string(arr.type)
            or pa.types.is_large_string(arr.type)
            or pa.types.is_floating(arr.type)
        ):
            arr = pc.cast(arr, pa.string(), safe=False)
        # Try cast to float; failures become null if safe=False? In practice, some versions may raise.
        # So: replace commas, trim, then cast with safe=False.
        s = pc.utf8_trim_whitespace(pc.cast(arr, pa.string(), safe=False))
        # Some datasets might have empty strings
        s = pc.if_else(pc.equal(s, pa.scalar("", pa.string())), pa.scalar(None, pa.string()), s)
        return pc.cast(s, pa.float64(), safe=False)

    start_lat = to_f64("start_lat")
    start_lng = to_f64("start_lng")
    end_lat = to_f64("end_lat")
    end_lng = to_f64("end_lng")

    if mode == "jc":
        lat_min, lat_max, lng_min, lng_max = (
            JC_BBOX_LAT_MIN,
            JC_BBOX_LAT_MAX,
            JC_BBOX_LNG_MIN,
            JC_BBOX_LNG_MAX,
        )
    elif mode == "nyc":
        lat_min, lat_max, lng_min, lng_max = (
            NYC_BBOX_LAT_MIN,
            NYC_BBOX_LAT_MAX,
            NYC_BBOX_LNG_MIN,
            NYC_BBOX_LNG_MAX,
        )
    else:
        return None

    def in_bbox(lat: pa.ChunkedArray, lng: pa.ChunkedArray) -> pa.ChunkedArray:
        lat_ok = pc.and_(
            pc.greater_equal(lat, pa.scalar(lat_min)), pc.less_equal(lat, pa.scalar(lat_max))
        )
        lng_ok = pc.and_(
            pc.greater_equal(lng, pa.scalar(lng_min)), pc.less_equal(lng, pa.scalar(lng_max))
        )
        ok = pc.and_(lat_ok, lng_ok)
        # If lat/lng is null, ok becomes null; treat null as False for filtering
        ok = pc.fill_null(ok, False)
        return ok

    start_ok = in_bbox(start_lat, start_lng)
    end_ok = in_bbox(end_lat, end_lng)

    # If strict, require BOTH start and end in bbox; else either.
    # We will compute strict in caller.
    # Return both masks packed? Simpler: return tuple? but signature returns one.
    # We'll store as struct? Not needed; compute here for strict=True and strict=False later.
    # We'll return a dict-like? Not allowed. We'll return a struct array? Too heavy.
    # Instead caller recomputes by calling this, so we return start_ok AND end_ok by default?
    # We'll return start_ok and let caller combine with end_ok by re-calling? Not good.
    # So we return a struct-like by packing to a table? We'll not.
    # We'll just return None here; caller will compute its own by calling helper again—wasteful.
    # Better: return start_ok and attach end_ok in closure in caller. We'll do bbox logic in caller instead.
    return None  # handled directly in _filter_table_by_station_ids


def _filter_table_by_station_ids(tbl: pa.Table, mode: str, strict: bool = True) -> pa.Table:
    """
    Filter rows by station IDs to reduce NYC/JC contamination.

    Strategy:
    - If station IDs look prefixed (JC/HB), use prefix filtering.
    - Otherwise (legacy numeric IDs), use a JC/Hoboken bounding box fallback for mode=jc.
      (And optional NYC bbox currently NYC uses prefix-only rejection.)
    """
    if mode not in ("nyc", "jc"):
        return tbl

    if tbl.num_rows == 0:
        return tbl

    start_col = tbl.column("start_station_id") if "start_station_id" in tbl.column_names else None
    end_col = tbl.column("end_station_id") if "end_station_id" in tbl.column_names else None

    if start_col is None and end_col is None:
        return tbl

    # ---- Prefix-based mask ----
    def prefix_mask(arr: pa.ChunkedArray) -> pa.ChunkedArray:
        str_arr = pc.cast(arr, pa.string(), safe=False)
        upper_arr = pc.utf8_upper(str_arr)

        is_jc = None
        for prefix in JC_STATION_PREFIXES:
            # IMPORTANT: pass prefix as Python str (NOT pa.scalar) for pyarrow compatibility
            m = pc.starts_with(upper_arr, prefix)
            is_jc = m if is_jc is None else pc.or_(is_jc, m)

        if is_jc is None:
            is_jc = pc.cast(
                pa.chunked_array([pa.array([False] * tbl.num_rows)]), pa.bool_(), safe=False
            )

        # Fill nulls to False (a null station_id shouldn't cause keep)
        is_jc = pc.fill_null(is_jc, False)

        if mode == "nyc":
            return pc.invert(is_jc)
        return is_jc

    # Determine whether prefix filtering is meaningful (i.e., any IDs actually have JC/HB prefixes)
    def has_any_prefixed(arr: pa.ChunkedArray) -> bool:
        try:
            s = pc.utf8_upper(pc.cast(arr, pa.string(), safe=False))
            found = False
            for prefix in JC_STATION_PREFIXES:
                m = pc.starts_with(s, prefix)
                m = pc.fill_null(m, False)
                if int(pc.sum(pc.cast(m, pa.int64())).as_py() or 0) > 0:
                    found = True
                    break
            return found
        except Exception:
            return False

    prefix_applicable = False
    if start_col is not None and has_any_prefixed(start_col):
        prefix_applicable = True
    elif end_col is not None and has_any_prefixed(end_col):
        prefix_applicable = True

    # ---- BBox fallback (mainly for JC legacy numeric IDs) ----
    def bbox_masks() -> tuple[pa.ChunkedArray, pa.ChunkedArray] | None:
        needed = {"start_lat", "start_lng", "end_lat", "end_lng"}
        if not needed.issubset(set(tbl.column_names)):
            return None

        def to_f64(colname: str) -> pa.ChunkedArray:
            arr = tbl[colname]
            s = pc.utf8_trim_whitespace(pc.cast(arr, pa.string(), safe=False))
            s = pc.if_else(pc.equal(s, pa.scalar("", pa.string())), pa.scalar(None, pa.string()), s)
            # cast to float (invalid -> error in some versions). We'll guard with try.
            try:
                return pc.cast(s, pa.float64(), safe=False)
            except Exception:
                # If cast fails hard, return all nulls => bbox not usable
                return pa.chunked_array(
                    [pa.nulls(tbl.num_rows, type=pa.float64())], type=pa.float64()
                )

        start_lat = to_f64("start_lat")
        start_lng = to_f64("start_lng")
        end_lat = to_f64("end_lat")
        end_lng = to_f64("end_lng")

        if mode == "jc":
            lat_min, lat_max, lng_min, lng_max = (
                JC_BBOX_LAT_MIN,
                JC_BBOX_LAT_MAX,
                JC_BBOX_LNG_MIN,
                JC_BBOX_LNG_MAX,
            )
        elif mode == "nyc":
            lat_min, lat_max, lng_min, lng_max = (
                NYC_BBOX_LAT_MIN,
                NYC_BBOX_LAT_MAX,
                NYC_BBOX_LNG_MIN,
                NYC_BBOX_LNG_MAX,
            )
        else:
            return None

        def in_bbox(lat: pa.ChunkedArray, lng: pa.ChunkedArray) -> pa.ChunkedArray:
            lat_ok = pc.and_(
                pc.greater_equal(lat, pa.scalar(lat_min)), pc.less_equal(lat, pa.scalar(lat_max))
            )
            lng_ok = pc.and_(
                pc.greater_equal(lng, pa.scalar(lng_min)), pc.less_equal(lng, pa.scalar(lng_max))
            )
            ok = pc.and_(lat_ok, lng_ok)
            ok = pc.fill_null(ok, False)
            return ok

        return in_bbox(start_lat, start_lng), in_bbox(end_lat, end_lng)

    # Choose filtering mode:
    # - If prefix_applicable: use prefix logic.
    # - Else if mode == jc: use bbox fallback (legacy numeric).
    # - Else (nyc without prefixed): do nothing (keep all) to avoid dropping valid NYC numeric IDs.
    use_bbox = (not prefix_applicable) and (mode == "jc")

    if use_bbox:
        bm = bbox_masks()
        if bm is None:
            # Can't bbox filter; return as-is
            return tbl
        start_ok, end_ok = bm
    else:
        start_ok = prefix_mask(start_col) if start_col is not None else None
        end_ok = prefix_mask(end_col) if end_col is not None else None

    # Combine masks
    if start_ok is not None and end_ok is not None:
        keep_mask = pc.and_(start_ok, end_ok) if strict else pc.or_(start_ok, end_ok)
    elif start_ok is not None:
        keep_mask = start_ok
    elif end_ok is not None:
        keep_mask = end_ok
    else:
        return tbl

    filtered = tbl.filter(keep_mask)

    rows_before = tbl.num_rows
    rows_after = filtered.num_rows
    rows_dropped = rows_before - rows_after
    if rows_dropped > 0:
        pct_dropped = (rows_dropped / rows_before) * 100
        strategy = "bbox" if use_bbox else "prefix"
        print(
            f"      Station filter ({mode}, {strategy}): dropped {rows_dropped:,} rows ({pct_dropped:.1f}%)"
        )

    return filtered


# -----------------------------
# INPUT VALIDATION
# -----------------------------
def validate_years(years: list[int]) -> None:
    MIN_YEAR = 2013
    MAX_YEAR = 2030
    for y in years:
        if not (MIN_YEAR <= y <= MAX_YEAR):
            raise ValueError(
                f"Year out of reasonable range ({MIN_YEAR}-{MAX_YEAR}): {y}\n"
                f"Citi Bike launched in 2013. Check --years argument.\n"
                f"Received: {years}"
            )


def validate_months(months: list[int]) -> None:
    for m in months:
        if not (1 <= m <= 12):
            raise ValueError(
                f"Month out of range (1-12): {m}\n"
                f"Check --months argument for typos.\n"
                f"Received: {months}"
            )


def validate_mode(mode: str) -> None:
    valid_modes = ["nyc", "jc", "auto"]
    mode_lower = mode.strip().lower()
    if mode_lower not in valid_modes:
        raise ValueError(
            f"Invalid mode: '{mode}'. Must be one of: {', '.join(valid_modes)}\n"
            f"For ingest, mode must be explicit (nyc/jc) or 'auto' to infer from filenames."
        )


def validate_station_id_policy(station_id_policy: str) -> None:
    _normalize_station_id_policy(station_id_policy)


def validate_pipeline_inputs(
    years: list[int], months: list[int], mode: str, station_id_policy: str = "name_coord"
) -> None:
    validate_years(years)
    validate_months(months)
    validate_mode(mode)
    validate_station_id_policy(station_id_policy)


# -----------------------------
# ZIP member cleanup
# -----------------------------
MACOS_JUNK_PREFIXES = ("__MACOSX/",)
MACOS_JUNK_PATTERNS = (
    re.compile(r"(^|/)\._"),
    re.compile(r"(^|/)\.DS_Store$"),
)


def _norm_zip_name(n: str) -> str:
    return n.replace("\\", "/").strip()


def is_macos_junk(name: str) -> bool:
    name = _norm_zip_name(name)
    if any(name.startswith(p) for p in MACOS_JUNK_PREFIXES):
        return True
    for pat in MACOS_JUNK_PATTERNS:
        if pat.search(name):
            return True
    return False


def _dedupe_preserve_order(names: list[str]) -> list[str]:
    seen = set()
    out = []
    for n in names:
        nn = _norm_zip_name(n)
        key = nn.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(nn)
    return out


_PART_SUFFIX_RE = re.compile(
    r"^(?P<stem>.*?-citibike-tripdata)(?:_(?P<part>\d+))?\.csv$", re.IGNORECASE
)


def _select_month_csv_members(csv_names: list[str]) -> list[str]:
    csv_names = _dedupe_preserve_order(csv_names)

    parts: list[tuple[int, str]] = []
    base: list[str] = []

    for n in csv_names:
        bn = Path(n).name
        m = _PART_SUFFIX_RE.match(bn)
        if not m:
            base.append(n)
            continue
        part = m.group("part")
        if part is None:
            base.append(n)
        else:
            parts.append((int(part), n))

    if parts:
        parts.sort(key=lambda x: x[0])
        return [n for _, n in parts]
    return csv_names


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_replace(tmp_path: Path, final_path: Path) -> None:
    os.replace(str(tmp_path), str(final_path))


def parse_compression(s: str) -> str | None:
    s = s.strip().lower()
    if s in ("none", "off", "no", "false", ""):
        return None
    if s in ("snappy", "gzip", "zstd", "brotli", "lz4"):
        return s
    raise ValueError(f"Unsupported compression: {s}")


# -----------------------------
# ZIP filename patterns
# -----------------------------
NYC_YEARLY_RE = re.compile(r"^(?P<year>\d{4})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE)
NYC_MONTHLY_RE = re.compile(
    r"^(?P<year>\d{4})(?P<month>\d{2})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE
)

JC_YEARLY_RE = re.compile(r"^JC-(?P<year>\d{4})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE)
JC_MONTHLY_RE = re.compile(
    r"^JC-(?P<year>\d{4})(?P<month>\d{2})-citibike-tripdata(?:\.csv)?\.zip$", re.IGNORECASE
)

INTERNAL_MONTH_ZIP_RE = re.compile(
    r"^(?:JC-)?(?P<year>\d{4})(?P<month>\d{2})-citibike-tripdata(?:\.csv)?\.zip$",
    re.IGNORECASE,
)

ANY_YYYYMM_RE = re.compile(r"(?P<year>20\d{2})(?P<month>0[1-9]|1[0-2])")


# -----------------------------
# Header sanitization + naming
# -----------------------------
_NAME_CLEAN_RE = re.compile(r"[^0-9a-zA-Z_]+")


def _standardize_name(name: str) -> str:
    n = name.strip().lower()
    n = n.replace("\ufeff", "")
    n = n.replace(" ", "_")
    n = _NAME_CLEAN_RE.sub("_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n


def _sanitize_header(header: list[str]) -> list[str]:
    out: list[str] = []
    counts: dict[str, int] = {}
    for i, raw in enumerate(header):
        raw = "" if raw is None else str(raw)
        h = raw.strip()
        if h.startswith("\ufeff"):
            h = h.lstrip("\ufeff")
        if h == "":
            h = f"col_{i + 1}"
        base = h
        key = base.lower()
        if key in counts:
            counts[key] += 1
            base = f"{base}__{counts[key]}"
        else:
            counts[key] = 1
        out.append(base)
    return out


# -----------------------------
# Timestamp parsing
# -----------------------------
TIME_COL_PRIORITY = ["started_at", "starttime", "start_time"]


def _month_bounds_us(year: int, month: int) -> tuple[pa.Scalar, pa.Scalar]:
    start_dt = datetime(year, month, 1)
    end_dt = datetime(year + 1, 1, 1) if month == 12 else datetime(year, month + 1, 1)
    return (
        pa.scalar(start_dt, type=pa.timestamp("us")),
        pa.scalar(end_dt, type=pa.timestamp("us")),
    )


def _parse_timestamp_chunked(arr: pa.ChunkedArray) -> pa.ChunkedArray:
    if pa.types.is_timestamp(arr.type):
        return arr

    s_arr = arr
    if not (pa.types.is_string(s_arr.type) or pa.types.is_large_string(s_arr.type)):
        s_arr = pc.cast(s_arr, pa.string(), safe=False)

    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%z",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]

    out_chunks: list[pa.Array] = []
    total_valid = 0

    for chunk in s_arr.chunks:
        s = chunk
        if not (pa.types.is_string(s.type) or pa.types.is_large_string(s.type)):
            s = pc.cast(s, pa.string(), safe=False)

        s = pc.utf8_trim_whitespace(s)
        s = pc.replace_substring_regex(s, r"\s+UTC$", "")
        s = pc.replace_substring_regex(s, r"([+-]\d{2}):(\d{2})$", r"\1\2")
        s = pc.replace_substring_regex(s, r"^(.*)\.\d+([Z]|[+-]\d{4})$", r"\1\2")
        s = pc.replace_substring_regex(s, r"^(.*)\.\d+$", r"\1")

        parsed_any = None
        for fmt in fmts:
            parsed = pc.strptime(s, format=fmt, unit="us", error_is_null=True)
            parsed_any = parsed if parsed_any is None else pc.coalesce(parsed_any, parsed)

        total_valid += int(pc.sum(pc.is_valid(parsed_any)).as_py() or 0)
        out_chunks.append(parsed_any)

    if total_valid == 0:
        return s_arr
    return pa.chunked_array(out_chunks, type=pa.timestamp("us"))


# -----------------------------
# Canonicalization
# -----------------------------
CANONICAL_COLS = [
    ("ride_id", pa.string()),
    ("rideable_type", pa.string()),
    ("started_at", pa.timestamp("us")),
    ("ended_at", pa.timestamp("us")),
    ("start_station_id", pa.string()),
    ("start_station_name", pa.string()),
    ("end_station_id", pa.string()),
    ("end_station_name", pa.string()),
    ("start_lat", pa.string()),
    ("start_lng", pa.string()),
    ("end_lat", pa.string()),
    ("end_lng", pa.string()),
    ("member_casual", pa.string()),
    ("usertype", pa.string()),
    ("bike_id", pa.string()),
    ("tripduration_seconds", pa.string()),
    ("gender", pa.string()),
    ("birth_year", pa.string()),
]
CANONICAL_SCHEMA = pa.schema([pa.field(n, t) for n, t in CANONICAL_COLS])


def _col_or_null(tbl: pa.Table, name: str, typ: pa.DataType) -> pa.ChunkedArray:
    if name in tbl.column_names:
        col = tbl[name]
        if not col.type.equals(typ):
            col = pc.cast(col, typ, safe=False)
        return col
    return pa.chunked_array([pa.nulls(tbl.num_rows, type=typ)], type=typ)


def _map_usertype_to_member_casual(usertype: pa.ChunkedArray) -> pa.ChunkedArray:
    s = pc.utf8_lower(pc.utf8_trim_whitespace(pc.cast(usertype, pa.string(), safe=False)))
    is_sub = pc.equal(s, pa.scalar("subscriber"))
    is_cust = pc.equal(s, pa.scalar("customer"))
    out = pc.if_else(
        is_sub,
        pa.scalar("member"),
        pc.if_else(is_cust, pa.scalar("casual"), pa.scalar(None, pa.string())),
    )
    return pc.cast(out, pa.string(), safe=False)


def _make_legacy_ride_id(tbl: pa.Table) -> pa.ChunkedArray:
    def first_existing(options: list[str]) -> str | None:
        for n in options:
            if n in tbl.column_names:
                return n
        return None

    candidates = [
        first_existing(["starttime", "start_time", "started_at"]),
        first_existing(["stoptime", "stop_time", "ended_at", "end_time"]),
        first_existing(["bikeid", "bike_id"]),
        first_existing(["start_station_id"]),
        first_existing(["end_station_id"]),
        first_existing(["tripduration", "trip_duration", "tripduration_seconds"]),
    ]

    parts = []
    for name in candidates:
        if name and name in tbl.column_names:
            col = tbl[name]
            if isinstance(col, pa.ChunkedArray):
                col = col.combine_chunks()
            parts.append(pc.cast(col, pa.string(), safe=False))
        else:
            parts.append(pa.array([None] * tbl.num_rows, type=pa.string()))

    ride_ids = []
    for i in range(tbl.num_rows):
        row_parts = []
        for part in parts:
            v = part[i].as_py() if part[i].is_valid else ""
            row_parts.append(str(v) if v is not None else "")
        ride_ids.append(hashlib.md5("|".join(row_parts).encode("utf-8")).hexdigest())

    return pa.array(ride_ids, type=pa.string())


def _standardize_table_column_names(tbl: pa.Table) -> pa.Table:
    return tbl.rename_columns([_standardize_name(n) for n in tbl.column_names])


def _normalize_to_canonical(tbl: pa.Table, mode: str, station_id_policy: str) -> pa.Table:
    if tbl.num_rows == 0:
        return pa.Table.from_arrays(
            [pa.nulls(0, type=t) for _, t in CANONICAL_COLS], schema=CANONICAL_SCHEMA
        )

    # --- timestamps (modern + legacy variants) ---
    started = None
    ended = None

    if "started_at" in tbl.column_names:
        started = _parse_timestamp_chunked(tbl["started_at"])
    elif "starttime" in tbl.column_names:
        started = _parse_timestamp_chunked(tbl["starttime"])
    elif "start_time" in tbl.column_names:
        started = _parse_timestamp_chunked(tbl["start_time"])

    if "ended_at" in tbl.column_names:
        ended = _parse_timestamp_chunked(tbl["ended_at"])
    elif "stoptime" in tbl.column_names:
        ended = _parse_timestamp_chunked(tbl["stoptime"])
    elif "stop_time" in tbl.column_names:
        ended = _parse_timestamp_chunked(tbl["stop_time"])
    elif "end_time" in tbl.column_names:
        ended = _parse_timestamp_chunked(tbl["end_time"])

    if started is None:
        started = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.timestamp("us"))])
    if ended is None:
        ended = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.timestamp("us"))])

    # --- ride_id ---
    if "ride_id" in tbl.column_names:
        ride_id = pc.cast(tbl["ride_id"], pa.string(), safe=False)
    else:
        ride_id = _make_legacy_ride_id(tbl)

    # --- rideable_type ---
    rideable_type = _col_or_null(tbl, "rideable_type", pa.string())

    # --- station ids/names ---
    start_station_id_raw = _normalize_station_id_array(
        _col_or_null(tbl, "start_station_id", pa.string()), mode=mode, apply_crosswalk=False
    )
    start_station_name = _col_or_null(tbl, "start_station_name", pa.string())
    end_station_id_raw = _normalize_station_id_array(
        _col_or_null(tbl, "end_station_id", pa.string()), mode=mode, apply_crosswalk=False
    )
    end_station_name = _col_or_null(tbl, "end_station_name", pa.string())

    # --- coordinates (modern vs legacy) ---
    start_lat = (
        _col_or_null(tbl, "start_lat", pa.string())
        if "start_lat" in tbl.column_names
        else _col_or_null(tbl, "start_station_latitude", pa.string())
    )
    start_lng = (
        _col_or_null(tbl, "start_lng", pa.string())
        if "start_lng" in tbl.column_names
        else _col_or_null(tbl, "start_station_longitude", pa.string())
    )
    end_lat = (
        _col_or_null(tbl, "end_lat", pa.string())
        if "end_lat" in tbl.column_names
        else _col_or_null(tbl, "end_station_latitude", pa.string())
    )
    end_lng = (
        _col_or_null(tbl, "end_lng", pa.string())
        if "end_lng" in tbl.column_names
        else _col_or_null(tbl, "end_station_longitude", pa.string())
    )

    if station_id_policy == "name_coord":
        start_station_id = _canonicalize_station_ids_from_identity(
            start_station_id_raw, start_station_name, start_lat, start_lng, mode
        )
        end_station_id = _canonicalize_station_ids_from_identity(
            end_station_id_raw, end_station_name, end_lat, end_lng, mode
        )
    elif station_id_policy == "raw":
        start_station_id = _normalize_station_id_array(
            start_station_id_raw, mode=mode, apply_crosswalk=True
        )
        end_station_id = _normalize_station_id_array(
            end_station_id_raw, mode=mode, apply_crosswalk=True
        )
    else:
        raise ValueError(
            f"Unsupported station id policy: {station_id_policy}. "
            f"Expected one of: {', '.join(NORMALIZED_STATION_ID_POLICIES)}"
        )

    # --- member/casual + usertype (legacy variants) ---
    if "member_casual" in tbl.column_names:
        member_casual = pc.cast(tbl["member_casual"], pa.string(), safe=False)
    else:
        if "usertype" in tbl.column_names:
            member_casual = _map_usertype_to_member_casual(tbl["usertype"])
        elif "user_type" in tbl.column_names:
            member_casual = _map_usertype_to_member_casual(tbl["user_type"])
        else:
            member_casual = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.string())])

    usertype = (
        _col_or_null(tbl, "usertype", pa.string())
        if "usertype" in tbl.column_names
        else _col_or_null(tbl, "user_type", pa.string())
    )

    # --- bike id ---
    if "bike_id" in tbl.column_names:
        bike_id = pc.cast(tbl["bike_id"], pa.string(), safe=False)
    elif "bikeid" in tbl.column_names:
        bike_id = pc.cast(tbl["bikeid"], pa.string(), safe=False)
    else:
        bike_id = pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.string())])

    # --- trip duration (legacy variants) ---
    if "tripduration_seconds" in tbl.column_names:
        tripduration_seconds = _col_or_null(tbl, "tripduration_seconds", pa.string())
    elif "tripduration" in tbl.column_names:
        tripduration_seconds = _col_or_null(tbl, "tripduration", pa.string())
    else:
        tripduration_seconds = _col_or_null(tbl, "trip_duration", pa.string())

    gender = _col_or_null(tbl, "gender", pa.string())
    birth_year = (
        pc.cast(tbl["birth_year"], pa.string(), safe=False)
        if "birth_year" in tbl.column_names
        else pa.chunked_array([pa.nulls(tbl.num_rows, type=pa.string())])
    )

    canonical = pa.Table.from_arrays(
        [
            ride_id,
            rideable_type,
            started,
            ended,
            start_station_id,
            start_station_name,
            end_station_id,
            end_station_name,
            start_lat,
            start_lng,
            end_lat,
            end_lng,
            member_casual,
            usertype,
            bike_id,
            tripduration_seconds,
            gender,
            birth_year,
        ],
        schema=CANONICAL_SCHEMA,
    )

    # Station filtering (prefix or bbox fallback)
    canonical = _filter_table_by_station_ids(canonical, mode, strict=True)

    return canonical


# -----------------------------
# CSV reading helpers
# -----------------------------
def _extract_yyyymm(filename: str) -> tuple[int, int] | None:
    m = ANY_YYYYMM_RE.search(filename)
    if not m:
        return None
    y = int(m.group("year"))
    mo = int(m.group("month"))
    return (y, mo)


def _read_header_line_csv(bio) -> list[str]:
    """
    Read the first line as a CSV header safely. Handles UTF-8 BOM and quoted headers.
    """
    first = bio.readline()
    if not first:
        return []
    try:
        line = first.decode("utf-8-sig")
    except Exception:
        line = first.decode("latin-1", errors="replace")
    row = next(csv.reader([line], delimiter=","))
    return [c.strip() for c in row]


def read_csv_to_arrow(path_or_bio, chunk_size: int = 500_000) -> Iterator[pa.Table]:
    """
    Read CSV yielding Arrow Tables.

    Key: safely handle ZIP streams.
    - Read header line ourselves
    - Rewind (or buffer) and then let pyarrow parse with skip_rows=1
    """
    bio = path_or_bio
    buffered: io.BytesIO | None = None

    try:
        header = _read_header_line_csv(bio)
        if hasattr(bio, "seek"):
            bio.seek(0)
        else:
            raise OSError("stream not seekable")
    except Exception:
        raw = bio.read()
        buffered = io.BytesIO(raw)
        header = _read_header_line_csv(buffered)
        buffered.seek(0)
        bio = buffered

    if not header:
        raise ValueError("CSV appears empty or header could not be read.")

    clean_header = _sanitize_header(header)
    col_types = {n: pa.string() for n in clean_header}

    read_opts = pacsv.ReadOptions(
        block_size=1 << 20,  # bytes
        skip_rows=1,  # we already read the header line
        column_names=clean_header,
    )
    convert_opts = pacsv.ConvertOptions(column_types=col_types, strings_can_be_null=True)
    parse_opts = pacsv.ParseOptions(delimiter=",")

    reader = pacsv.open_csv(
        bio, read_options=read_opts, parse_options=parse_opts, convert_options=convert_opts
    )
    for batch in reader:
        yield pa.Table.from_batches([batch])


def iter_tables_from_csv_member(
    zf: zipfile.ZipFile,
    csv_name: str,
    enforce_year: int,
    enforce_month: int,
    keep_null_times: bool,
    mode: str,
    station_id_policy: str,
) -> Iterator[pa.Table]:
    with zf.open(csv_name) as bio:
        for raw_tbl in read_csv_to_arrow(bio, chunk_size=500_000):
            if raw_tbl.num_rows == 0:
                continue

            tbl = _standardize_table_column_names(raw_tbl)
            tbl = _normalize_to_canonical(tbl, mode=mode, station_id_policy=station_id_policy)

            if tbl.num_rows == 0:
                continue

            if enforce_month:
                start_col = tbl["started_at"]
                if pa.types.is_timestamp(start_col.type):
                    lower, upper = _month_bounds_us(enforce_year, enforce_month)
                    in_month = pc.and_(
                        pc.greater_equal(start_col, lower), pc.less(start_col, upper)
                    )

                    if not keep_null_times:
                        keep = pc.and_(in_month, pc.is_valid(start_col))
                    else:
                        keep = pc.or_(in_month, pc.is_null(start_col))

                    tbl = tbl.filter(keep)

            if tbl.num_rows > 0:
                yield tbl


def _iter_tables_from_zip_member_for_month(
    zf: zipfile.ZipFile,
    zip_name: str,
    year: int,
    month: int,
    keep_null_times: bool,
    mode: str,
    station_id_policy: str,
    max_depth: int = 2,
) -> Iterator[pa.Table]:
    if max_depth <= 0:
        return

    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd) / "nested.zip"
        with zf.open(zip_name) as src, open(tmp_path, "wb") as dst:
            shutil.copyfileobj(src, dst)

        with zipfile.ZipFile(tmp_path) as inner_zf:
            names = [n for n in inner_zf.namelist() if not is_macos_junk(n)]
            names = _dedupe_preserve_order(names)

            csvs = [n for n in names if n.lower().endswith(".csv")]
            zips = [n for n in names if n.lower().endswith(".zip")]

            for csv_name in csvs:
                yield from iter_tables_from_csv_member(
                    inner_zf, csv_name, year, month, keep_null_times, mode, station_id_policy
                )

            for nested_zip_name in zips:
                yield from _iter_tables_from_zip_member_for_month(
                    inner_zf,
                    nested_zip_name,
                    year,
                    month,
                    keep_null_times,
                    mode,
                    station_id_policy,
                    max_depth - 1,
                )


# -----------------------------
# Parquet writing (atomic)
# -----------------------------
def write_parquet_atomic_tables(
    out_path: Path,
    table_iter: Iterator[pa.Table],
    force: bool,
    compression: str | None,
) -> int:
    if out_path.exists() and not force:
        return 0

    ensure_dir(out_path.parent)
    tmp_path = out_path.parent / f"{out_path.name}.tmp.{os.getpid()}"

    try:
        writer = None
        total_rows = 0

        for i, tbl in enumerate(table_iter):
            if tbl.num_rows == 0:
                continue

            if not tbl.schema.equals(CANONICAL_SCHEMA):
                cols = []
                for field in CANONICAL_SCHEMA:
                    if field.name in tbl.column_names:
                        col = tbl[field.name]
                        if not col.type.equals(field.type):
                            col = pc.cast(col, field.type, safe=False)
                    else:
                        col = pa.nulls(tbl.num_rows, type=field.type)
                    cols.append(col)
                tbl = pa.Table.from_arrays(cols, schema=CANONICAL_SCHEMA)

            if writer is None:
                writer = pq.ParquetWriter(
                    tmp_path,
                    schema=CANONICAL_SCHEMA,
                    compression=compression,
                    version="2.6",
                    use_dictionary=True,
                    write_statistics=True,
                )

            writer.write_table(tbl)
            total_rows += tbl.num_rows

            if (i + 1) % 10 == 0:
                gc.collect()

        if writer is None:
            return 0

        writer.close()
        writer = None
        atomic_replace(tmp_path, out_path)
        return total_rows

    except Exception:
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass
        raise


# -----------------------------
# Meta + ingestion
# -----------------------------
@dataclass(frozen=True)
class ZipMeta:
    mode: str  # nyc | jc
    kind: str  # yearly | monthly
    year: int
    month: int | None
    path: Path


def detect_zip_meta(p: Path, mode: str) -> ZipMeta | None:
    name = p.name

    if mode not in ("nyc", "jc", "auto"):
        raise ValueError("mode must be one of: nyc, jc, auto")

    inferred_mode = "jc" if name.lower().startswith("jc-") else "nyc"
    use_mode = inferred_mode if mode == "auto" else mode

    if use_mode == "nyc":
        m = NYC_YEARLY_RE.match(name)
        if m:
            return ZipMeta(mode="nyc", kind="yearly", year=int(m.group("year")), month=None, path=p)
        m = NYC_MONTHLY_RE.match(name)
        if m:
            return ZipMeta(
                mode="nyc",
                kind="monthly",
                year=int(m.group("year")),
                month=int(m.group("month")),
                path=p,
            )
        return None

    m = JC_YEARLY_RE.match(name)
    if m:
        return ZipMeta(mode="jc", kind="yearly", year=int(m.group("year")), month=None, path=p)
    m = JC_MONTHLY_RE.match(name)
    if m:
        return ZipMeta(
            mode="jc",
            kind="monthly",
            year=int(m.group("year")),
            month=int(m.group("month")),
            path=p,
        )
    return None


def month_out_path(out_dir: Path, mode: str, year: int, month: int) -> Path:
    return out_dir / f"mode={mode}" / f"year={year}" / f"month={month:02d}" / "tripdata.parquet"


def ingest_monthly_zip(
    meta: ZipMeta,
    out_dir: Path,
    months_filter: set | None,
    force: bool,
    compression: str | None,
    keep_null_times: bool,
    station_id_policy: str,
) -> None:
    assert meta.kind == "monthly" and meta.month is not None

    if months_filter is not None and meta.month not in months_filter:
        return

    out_path = month_out_path(out_dir, meta.mode, meta.year, meta.month)
    if out_path.exists() and not force:
        print(f" - {meta.path.name} -> exists, skipping: {out_path}")
        return

    print(f" - {meta.path.name} (mode={meta.mode}, {meta.year}-{meta.month:02d})")

    with zipfile.ZipFile(meta.path) as zf:
        names = [n for n in zf.namelist() if not is_macos_junk(n)]
        names = _dedupe_preserve_order(names)

        csvs = [n for n in names if n.lower().endswith(".csv")]
        inner_zips = [n for n in names if n.lower().endswith(".zip")]

        if not csvs and not inner_zips:
            print("   -> no CSVs or nested ZIPs found, skipping.")
            return

        if csvs:
            csvs = _select_month_csv_members(csvs)
            print(f"   -> month={meta.month:02d}: {len(csvs)} CSV file(s) selected")
            for n in csvs:
                print(f"      reading: {Path(n).name}")

        if (not csvs) and inner_zips:
            print(f"   -> month={meta.month:02d}: no CSVs; found {len(inner_zips)} nested ZIP(s)")
            for n in inner_zips:
                print(f"      opening nested zip: {Path(n).name}")

        def table_iter() -> Iterator[pa.Table]:
            for csv_name in csvs:
                yield from iter_tables_from_csv_member(
                    zf,
                    csv_name,
                    enforce_year=meta.year,
                    enforce_month=meta.month,
                    keep_null_times=keep_null_times,
                    mode=meta.mode,
                    station_id_policy=station_id_policy,
                )
            for zn in inner_zips:
                yield from _iter_tables_from_zip_member_for_month(
                    zf,
                    zn,
                    meta.year,
                    meta.month,
                    keep_null_times=keep_null_times,
                    mode=meta.mode,
                    station_id_policy=station_id_policy,
                    max_depth=2,
                )

        rows = write_parquet_atomic_tables(
            out_path, table_iter(), force=force, compression=compression
        )

    if rows > 0:
        print(f"      -> wrote {out_path} (rows={rows:,})")


def ingest_yearly_zip(
    meta: ZipMeta,
    out_dir: Path,
    months_filter: set | None,
    force: bool,
    compression: str | None,
    keep_null_times: bool,
    station_id_policy: str,
) -> None:
    assert meta.kind == "yearly"
    print(f" - {meta.path.name} (mode={meta.mode}, {meta.year}, YEARLY ZIP)")

    with zipfile.ZipFile(meta.path) as zf:
        names = [n for n in zf.namelist() if not is_macos_junk(n)]
        names = _dedupe_preserve_order(names)

        csv_members = [n for n in names if n.lower().endswith(".csv")]
        zip_members = [n for n in names if n.lower().endswith(".zip")]

        if not csv_members and not zip_members:
            print("   -> no CSVs or nested ZIPs found, skipping.")
            return

        by_month_csv: dict[int, list[str]] = {}
        for n in csv_members:
            got = _extract_yyyymm(Path(n).name)
            if not got:
                continue
            y, mo = got
            if y == meta.year:
                by_month_csv.setdefault(mo, []).append(n)

        by_month_zip: dict[int, list[str]] = {}
        for n in zip_members:
            m = INTERNAL_MONTH_ZIP_RE.match(Path(n).name)
            if not m:
                continue
            y = int(m.group("year"))
            mo = int(m.group("month"))
            if y == meta.year:
                by_month_zip.setdefault(mo, []).append(n)

        months_present = sorted(set(by_month_csv.keys()) | set(by_month_zip.keys()))
        if not months_present:
            print("   -> no month-like CSVs / nested zips found inside yearly ZIP; skipping.")
            return

        for mo in months_present:
            if months_filter is not None and mo not in months_filter:
                continue

            out_path = month_out_path(out_dir, meta.mode, meta.year, mo)
            if out_path.exists() and not force:
                print(f"   -> {meta.year}-{mo:02d}: exists, skipping: {out_path}")
                continue

            csvs = _select_month_csv_members(by_month_csv.get(mo, []))
            zips = _dedupe_preserve_order(by_month_zip.get(mo, []))

            print(
                f"   -> month={mo:02d}: {len(csvs)} CSV file(s) selected, {len(zips)} nested ZIP(s)"
            )
            for n in csvs:
                print(f"      reading: {Path(n).name}")
            for n in zips:
                print(f"      opening nested zip: {Path(n).name}")

            def table_iter() -> Iterator[pa.Table]:
                for csv_name in csvs:
                    yield from iter_tables_from_csv_member(
                        zf,
                        csv_name,
                        enforce_year=meta.year,
                        enforce_month=mo,
                        keep_null_times=keep_null_times,
                        mode=meta.mode,
                        station_id_policy=station_id_policy,
                    )
                for zip_name in zips:
                    yield from _iter_tables_from_zip_member_for_month(
                        zf,
                        zip_name,
                        meta.year,
                        mo,
                        keep_null_times=keep_null_times,
                        mode=meta.mode,
                        station_id_policy=station_id_policy,
                        max_depth=2,
                    )

            rows = write_parquet_atomic_tables(
                out_path, table_iter(), force=force, compression=compression
            )
            if rows > 0:
                print(f"      -> wrote {out_path} (rows={rows:,})")


def main(argv: Sequence[str] | None = None) -> int:
    faulthandler.enable(all_threads=True)

    ap = argparse.ArgumentParser(
        description="Ingest Citi Bike trip data from ZIP files to partitioned Parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--raw-dir", required=True, type=Path, help="Directory containing downloaded ZIP files"
    )
    ap.add_argument(
        "--out-dir", required=True, type=Path, help="Output directory for partitioned parquet"
    )
    ap.add_argument("--mode", required=True, type=str, help="Mode: nyc|jc|auto")
    ap.add_argument(
        "--years", nargs="+", required=True, type=int, help="Years to process (e.g., 2023 2024)"
    )
    ap.add_argument("--months", nargs="+", required=True, type=int, help="Months to process (1-12)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing parquet files")
    ap.add_argument(
        "--keep-null-times",
        action="store_true",
        help="Keep rows with null started_at in month partitions",
    )
    ap.add_argument(
        "--compression",
        type=str,
        default="none",
        help="Parquet compression: none|snappy|gzip|zstd|brotli|lz4 (default: none)",
    )
    ap.add_argument(
        "--station-id-policy",
        type=str,
        default="name_coord",
        choices=list(STATION_ID_POLICIES),
        help=(
            "Station ID handling policy: "
            "raw (keep normalized/raw IDs) or name_coord (canonical id from station name+coords)."
        ),
    )
    ap.add_argument(
        "--station-id-crosswalk",
        type=Path,
        default=None,
        help=(
            "Optional CSV mapping old station IDs to canonical IDs. "
            "Accepted columns: old_id/new_id (or from_id/to_id), optional mode=any|nyc|jc."
        ),
    )

    # kept for compatibility (ignored)
    ap.add_argument(
        "--chunksize", type=int, default=250_000, help="(ignored) kept for compatibility"
    )
    ap.add_argument("--csv-engine", choices=["python", "c"], default="python", help="(ignored)")
    ap.add_argument("--extract-csv", action="store_true", help="(ignored)")

    args = ap.parse_args(argv)

    print("=" * 70)
    print("INPUT VALIDATION")
    print("=" * 70)
    normalized_policy_preview = ""
    try:
        validate_pipeline_inputs(args.years, args.months, args.mode, args.station_id_policy)
        normalized_policy_preview = _normalize_station_id_policy(args.station_id_policy)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"✓ Valid years: {sorted(args.years)}")
    print(f"✓ Valid months: {sorted(args.months)}")
    print(f"✓ Valid mode: {args.mode}")
    print(f"✓ Station ID policy: {normalized_policy_preview} (input: {args.station_id_policy})")
    print()

    # conservative env defaults
    os.environ.setdefault("ARROW_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.out_dir
    mode: str = args.mode.strip().lower()
    years = set(args.years)
    months_filter = set(args.months) if args.months else None
    force: bool = bool(args.force)
    keep_null_times: bool = bool(args.keep_null_times)
    compression: str | None = parse_compression(args.compression)
    station_id_policy: str = normalized_policy_preview

    global STATION_ID_CROSSWALK
    if args.station_id_crosswalk is not None:
        try:
            STATION_ID_CROSSWALK = _load_station_id_crosswalk(args.station_id_crosswalk)
        except Exception as e:
            print(f"ERROR: failed loading station-id crosswalk: {e}", file=sys.stderr)
            return 2
    else:
        STATION_ID_CROSSWALK = {"any": {}, "nyc": {}, "jc": {}}

    if not raw_dir.exists():
        print(f"ERROR: raw-dir does not exist: {raw_dir}", file=sys.stderr)
        return 2

    zips = sorted([p for p in raw_dir.iterdir() if p.is_file() and p.suffix.lower() == ".zip"])
    metas: list[ZipMeta] = []
    for p in zips:
        meta = detect_zip_meta(p, mode=mode)
        if meta is None:
            continue
        if meta.year not in years:
            continue
        metas.append(meta)

    if not metas:
        print("No ZIPs selected for ingestion (check raw-dir / filters / mode).")
        return 0

    if station_id_policy == "name_coord":
        _reset_station_name_coord_clusters()

    print("=" * 70)
    print("INGESTION PLAN")
    print("=" * 70)
    print(f"Raw dir:       {raw_dir}")
    print(f"Output dir:    {out_dir}")
    print(f"Mode:          {mode}")
    print(f"Years:         {sorted(years)}")
    print(f"Months filter: {sorted(months_filter) if months_filter else 'ALL'}")
    print(f"Compression:   {compression or 'NONE'}")
    print(f"Station IDs:   {station_id_policy}")
    print(f"Force:         {force}")
    print(
        "Station ID remap: "
        + (
            f"enabled ({sum(len(v) for v in STATION_ID_CROSSWALK.values())} mapping(s))"
            if args.station_id_crosswalk is not None
            else "disabled"
        )
    )
    print(f"ZIPs found:    {len(metas)}")
    print("Station filtering: ENABLED (jc uses bbox fallback for legacy numeric IDs)")
    print()

    print("=" * 70)
    print(f"PROCESSING {len(metas)} ZIP(S)")
    print("=" * 70)

    metas.sort(key=lambda m: (m.mode, m.year, 0 if m.kind == "monthly" else 1, m.month or 0))

    for meta in metas:
        if meta.kind == "monthly":
            ingest_monthly_zip(
                meta,
                out_dir,
                months_filter,
                force,
                compression,
                keep_null_times,
                station_id_policy,
            )
        else:
            ingest_yearly_zip(
                meta,
                out_dir,
                months_filter,
                force,
                compression,
                keep_null_times,
                station_id_policy,
            )

    if station_id_policy == "name_coord":
        processed_modes = sorted(set(m.mode for m in metas))
        for report_mode in processed_modes:
            _print_station_coord_outlier_report(report_mode)

    print()
    print("=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)
    print(f"✓ Station filtering ensured {mode.upper()} separation (best-effort for legacy IDs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
