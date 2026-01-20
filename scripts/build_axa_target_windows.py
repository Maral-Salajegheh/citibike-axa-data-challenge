#!/usr/bin/env python3
"""
build_axa_target_windows.py

Build an AXA-ready 'when to act' table from per-run time summaries.

Inputs (in summaries/<RUN_TAG>/):
  - citibike_trips_by_hour.csv
  - citibike_trips_by_dow.csv
  - citibike_trips_by_month.csv

Output:
  - axa_target_windows.csv

Key behavior:
- If hour/dow tables include 'year', aggregate across years for targeting selection
  (avoid selecting year-specific rows while labeling only hour/dow).
- pct_within_week_part and pct_of_mode_year_trips are computed as percentages where meaningful.
- Month windows are made percentage-consistent (no reuse of percent columns for trips/day).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def top_n(df: pd.DataFrame, n: int, sort_col: str) -> pd.DataFrame:
    return df.sort_values(sort_col, ascending=False).head(n).copy()


def _coerce_int(df: pd.DataFrame, col: str) -> None:
    df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)


def _ensure_week_part(df: pd.DataFrame) -> None:
    if "week_part" not in df.columns:
        df["week_part"] = "overall"
    df["week_part"] = df["week_part"].astype(str)


def _normalize_mode(df: pd.DataFrame, mode_arg: Optional[str]) -> tuple[pd.DataFrame, str]:
    if "mode" not in df.columns:
        df = df.copy()
        df["mode"] = "unknown"

    df["mode"] = df["mode"].astype(str).str.lower()

    if mode_arg:
        m = mode_arg.strip().lower()
        df2 = df[df["mode"] == m].copy()
        if df2.empty:
            raise SystemExit(f"--mode={m!r} not found in input rows. Available modes: {sorted(df['mode'].unique())}")
        return df2, m

    modes = sorted(df["mode"].unique())
    if len(modes) != 1:
        raise SystemExit(
            "Input contains multiple modes but --mode was not provided.\n"
            f"Available modes: {modes}\n"
            "Run with e.g. --mode nyc or --mode jc."
        )
    return df, modes[0]


def _compute_pct_columns(df: pd.DataFrame, group_mode_col: str = "mode") -> pd.DataFrame:
    df = df.copy()
    _coerce_int(df, "trips")
    _ensure_week_part(df)

    scope_cols = [group_mode_col]
    if "year" in df.columns:
        scope_cols.append("year")

    totals_scope = df.groupby(scope_cols)["trips"].transform("sum").replace({0: pd.NA})
    df["pct_of_mode_year_trips"] = (df["trips"] / totals_scope) * 100.0

    totals_part = df.groupby(scope_cols + ["week_part"])["trips"].transform("sum").replace({0: pd.NA})
    df["pct_within_week_part"] = (df["trips"] / totals_part) * 100.0
    return df


def _aggregate_across_years_for_targeting(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    df = df.copy()
    _coerce_int(df, "trips")
    _ensure_week_part(df)

    agg = (
        df.groupby(keys, as_index=False)["trips"]
          .sum()
          .sort_values(keys)
          .reset_index(drop=True)
    )
    agg = _compute_pct_columns(agg, group_mode_col="mode")
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, help="Run summary directory, e.g. summaries/<RUN_TAG>")
    ap.add_argument("--out-dir", required=True, help="Output directory (usually same as --in-dir)")
    ap.add_argument("--mode", default=None, help="Optional mode filter if in-dir accidentally contains multiple modes")
    ap.add_argument("--top-hours", type=int, default=5, help="Top hours per week_part")
    ap.add_argument("--top-dows", type=int, default=3, help="Top days-of-week per week_part")
    ap.add_argument("--top-months", type=int, default=6, help="Top months overall (by trips_per_day)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_hour = in_dir / "citibike_trips_by_hour.csv"
    p_dow = in_dir / "citibike_trips_by_dow.csv"
    p_month = in_dir / "citibike_trips_by_month.csv"

    for p in (p_hour, p_dow, p_month):
        if not p.exists():
            raise FileNotFoundError(f"Missing required input: {p}")

    hour = pd.read_csv(p_hour)
    dow = pd.read_csv(p_dow)
    month = pd.read_csv(p_month)

    # Normalize/filter modes consistently (use hour as primary)
    hour, mode = _normalize_mode(hour, args.mode)

    if "mode" in dow.columns:
        dow["mode"] = dow["mode"].astype(str).str.lower()
        dow = dow[dow["mode"] == mode].copy()
    else:
        dow = dow.copy()
        dow["mode"] = mode

    if "mode" in month.columns:
        month["mode"] = month["mode"].astype(str).str.lower()
        month = month[month["mode"] == mode].copy()
    else:
        month = month.copy()
        month["mode"] = mode

    # -----------------------------
    # Hour targeting
    # -----------------------------
    hour = hour.copy()
    hour["hour"] = pd.to_numeric(hour.get("hour", pd.NA), errors="coerce")
    hour = hour.dropna(subset=["hour"]).copy()
    hour["hour"] = hour["hour"].astype(int)
    _ensure_week_part(hour)
    _coerce_int(hour, "trips")

    if "year" in hour.columns:
        hour_tgt = _aggregate_across_years_for_targeting(hour, keys=["mode", "hour", "week_part"])
    else:
        if "pct_within_week_part" not in hour.columns or "pct_of_mode_year_trips" not in hour.columns:
            hour = _compute_pct_columns(hour, group_mode_col="mode")
        hour_tgt = hour

    hour_out_parts = []
    for wp in ["weekday", "weekend"]:
        sub = hour_tgt[hour_tgt["week_part"] == wp].copy()
        if sub.empty:
            continue
        sub = top_n(sub, args.top_hours, "trips")
        sub["window_type"] = "hour_peak"
        sub["segment"] = wp
        sub["window_label"] = sub["hour"].astype(int).astype(str).str.zfill(2) + ":00"
        sub["priority_metric"] = sub["pct_within_week_part"]
        hour_out_parts.append(sub)

    hour_out = pd.concat(hour_out_parts, ignore_index=True) if hour_out_parts else pd.DataFrame(
        columns=["window_type", "segment", "window_label", "trips", "pct_of_mode_year_trips", "pct_within_week_part", "priority_metric"]
    )
    hour_out = hour_out[["window_type","segment","window_label","trips","pct_of_mode_year_trips","pct_within_week_part","priority_metric"]]

    # Optional commuter window
    commuter = hour_tgt[hour_tgt["week_part"] == "weekday"].copy()
    if not commuter.empty:
        commuter = commuter[commuter["hour"].isin([7, 8, 9, 16, 17, 18, 19])].copy()
        commuter = commuter.sort_values("trips", ascending=False)
        commuter["window_type"] = "commuter_hours"
        commuter["segment"] = "weekday"
        commuter["window_label"] = commuter["hour"].astype(int).astype(str).str.zfill(2) + ":00"
        commuter["priority_metric"] = commuter["pct_within_week_part"]
        commuter_out = commuter[hour_out.columns].head(7)
    else:
        commuter_out = pd.DataFrame(columns=hour_out.columns)

    # -----------------------------
    # Day-of-week targeting
    # -----------------------------
    dow = dow.copy()
    if "dow_name" not in dow.columns and "dow" in dow.columns:
        dow["dow_name"] = dow["dow"].astype(str)

    _ensure_week_part(dow)
    _coerce_int(dow, "trips")

    if "year" in dow.columns:
        dow_tgt = _aggregate_across_years_for_targeting(dow, keys=["mode", "dow", "dow_name", "week_part"])
    else:
        if "pct_within_week_part" not in dow.columns or "pct_of_mode_year_trips" not in dow.columns:
            dow = _compute_pct_columns(dow, group_mode_col="mode")
        dow_tgt = dow

    dow_out_parts = []
    for wp in ["weekday", "weekend"]:
        sub = dow_tgt[dow_tgt["week_part"] == wp].copy()
        if sub.empty:
            continue
        sub = top_n(sub, args.top_dows, "trips")
        sub["window_type"] = "day_of_week_peak"
        sub["segment"] = wp
        sub["window_label"] = sub["dow_name"].astype(str)
        sub["priority_metric"] = sub["pct_within_week_part"]
        dow_out_parts.append(sub)

    dow_out = pd.concat(dow_out_parts, ignore_index=True) if dow_out_parts else pd.DataFrame(columns=hour_out.columns)
    dow_out = dow_out[hour_out.columns]

    # -----------------------------
    # Month targeting (FIXED: percentage-consistent)
    # -----------------------------
    month = month.copy()
    _coerce_int(month, "trips")
    month["trips_per_day"] = pd.to_numeric(month.get("trips_per_day", pd.NA), errors="coerce")
    month["year"] = pd.to_numeric(month.get("year", pd.NA), errors="coerce")
    month["month"] = pd.to_numeric(month.get("month", pd.NA), errors="coerce")

    month = month.dropna(subset=["year", "month"]).copy()
    month["year"] = month["year"].astype(int)
    month["month"] = month["month"].astype(int)

    # Compute % of year trips for each year (no NaNs unless year total is 0)
    year_totals = month.groupby(["mode", "year"])["trips"].transform("sum").replace({0: pd.NA})
    month["pct_of_mode_year_trips"] = (month["trips"] / year_totals) * 100.0

    # For month_peak, segment is "overall" so "within week_part" == same denominator
    month["pct_within_week_part"] = month["pct_of_mode_year_trips"]

    month_top = (
        month.dropna(subset=["trips_per_day"])
             .sort_values("trips_per_day", ascending=False)
             .head(args.top_months)
             .copy()
    )
    month_top["window_type"] = "month_peak"
    month_top["segment"] = "overall"
    month_top["window_label"] = month_top["year"].astype(str) + "-" + month_top["month"].astype(str).str.zfill(2)

    # Rank months by trips_per_day (still good “seasonality intensity”)
    month_top["priority_metric"] = month_top["trips_per_day"]

    month_out = month_top[hour_out.columns].copy()

    # -----------------------------
    # Combine outputs + actions
    # -----------------------------
    out = pd.concat([hour_out, commuter_out, dow_out, month_out], ignore_index=True)

    def recommend_action(row):
        if row["window_type"] == "hour_peak" and row["segment"] == "weekday":
            return "Acquisition + product upsell (commute peaks); consider safety nudge"
        if row["window_type"] == "hour_peak" and row["segment"] == "weekend":
            return "Leisure targeting + safety messaging"
        if row["window_type"] == "commuter_hours":
            return "Strong commuter acquisition + product bundle"
        if row["window_type"] == "day_of_week_peak":
            return "Campaign scheduling + staffing/ops alignment"
        if row["window_type"] == "month_peak":
            return "Seasonal campaign planning"
        return "General"

    out["recommended_action"] = out.apply(recommend_action, axis=1)

    out_path = out_dir / "axa_target_windows.csv"
    out.to_csv(out_path, index=False)

    print(f"Mode: {mode}")
    print(f"Wrote: {out_path}")
    print(out.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
