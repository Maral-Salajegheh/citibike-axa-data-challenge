#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
from typing import Optional, List

import numpy as np
import pandas as pd

REQUIRED = [
    "citibike_trips_by_year.csv",
    "citibike_trips_by_month.csv",
    "citibike_trips_by_dow.csv",
    "citibike_trips_by_hour.csv",
]


def parse_months_spec(spec: Optional[str]) -> Optional[List[int]]:
    """
    Parse month spec like:
      "1,2,3"
      "1-3"
      "1,2,3,10-12"
    Returns sorted unique months [1..12], or None if spec is empty/None.
    """
    if spec is None:
        return None
    s = str(spec).strip()
    if not s:
        return None

    months: set[int] = set()
    parts = re.split(r"\s*,\s*", s)
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            a_i = int(a.strip())
            b_i = int(b.strip())
            lo, hi = (a_i, b_i) if a_i <= b_i else (b_i, a_i)
            for m in range(lo, hi + 1):
                months.add(m)
        else:
            months.add(int(p))

    bad = [m for m in months if m < 1 or m > 12]
    if bad:
        raise ValueError(f"--yoy-months contains invalid months: {bad}. Valid range is 1..12.")

    return sorted(months)


def months_signature(months: List[int]) -> str:
    return ",".join(str(m) for m in sorted(months))


def pick_best_runs(summaries_root: Path) -> pd.DataFrame:
    """
    Find all run folders and pick the best run for each (mode, year).
    Best = max months_covered, then max trips.
    Returns: run_tag, mode, year, trips, months_covered
    """
    run_dirs = [d for d in summaries_root.glob("y*_mode*") if d.is_dir()]
    rows = []

    for d in run_dirs:
        year_f = d / "citibike_trips_by_year.csv"
        if not year_f.exists():
            continue

        dfy = pd.read_csv(year_f)
        if not set(["mode", "year", "trips"]).issubset(dfy.columns):
            continue

        # months coverage (optional but helpful to pick the "best" run)
        month_f = d / "citibike_trips_by_month.csv"
        if month_f.exists():
            dfm = pd.read_csv(month_f)
            if set(["mode", "year", "month"]).issubset(dfm.columns):
                cov = (
                    dfm.groupby(["mode", "year"], as_index=False)["month"]
                    .nunique()
                    .rename(columns={"month": "months_covered"})
                )
                dfy = dfy.merge(cov, on=["mode", "year"], how="left")
            else:
                dfy["months_covered"] = pd.NA
        else:
            dfy["months_covered"] = pd.NA

        dfy["run_tag"] = d.name
        rows.append(dfy[["run_tag", "mode", "year", "trips", "months_covered"]])

    if not rows:
        raise SystemExit(f"No usable run summaries found under {summaries_root}")

    all_runs = pd.concat(rows, ignore_index=True)
    all_runs["_months_cov_sort"] = all_runs["months_covered"].fillna(-1)

    best = (
        all_runs.sort_values(
            ["mode", "year", "_months_cov_sort", "trips"],
            ascending=[True, True, False, False],
        )
        .drop_duplicates(subset=["mode", "year"], keep="first")
        .drop(columns=["_months_cov_sort"], errors="ignore")
        .reset_index(drop=True)
    )
    return best


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required summary file: {path}")
    return pd.read_csv(path)


def ensure_year_mode(df: pd.DataFrame, year: int, mode: str) -> pd.DataFrame:
    """Some summary CSVs (dow/hour) may not include year; add it from the run metadata."""
    out = df.copy()
    if "mode" not in out.columns:
        out["mode"] = mode
    if "year" not in out.columns:
        out["year"] = int(year)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries-root", default="summaries")
    ap.add_argument("--out-dir", default="summaries/_compare")
    ap.add_argument(
        "--yoy-months",
        default=None,
        help="Optional: compute YoY only for these months (e.g. '1,2,3' or '1-3' or '1,2,3,10-12'). "
             "If not set, YoY is computed only when consecutive years have identical month coverage.",
    )
    args = ap.parse_args()

    yoy_months = parse_months_spec(args.yoy_months)
    yoy_sig = months_signature(yoy_months) if yoy_months else None

    root = Path(args.summaries_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = pick_best_runs(root)
    best.to_csv(out_dir / "selected_best_runs.csv", index=False)

    # IMPORTANT: best may contain multiple rows with the SAME run_tag (e.g., a single run folder
    # summarizing multiple years). If we iterate over best.iterrows(), we will read that run's
    # CSVs multiple times and DUPLICATE data. Always read per unique run_tag.
    run_tags = best["run_tag"].dropna().unique().tolist()

    # ---- MONTH (combined) ----
    month_rows = []
    for run_tag in run_tags:
        d = root / run_tag
        dfm = safe_read_csv(d / "citibike_trips_by_month.csv")
        dfm["run_tag"] = run_tag
        month_rows.append(dfm)

    month_all = pd.concat(month_rows, ignore_index=True)

    # percent within each (mode, year)
    year_totals = (
        month_all.groupby(["mode", "year"], as_index=False)["trips"]
        .sum()
        .rename(columns={"trips": "year_trips"})
    )
    month_all = month_all.merge(year_totals, on=["mode", "year"], how="left")
    month_all["pct_of_mode_year_trips"] = (month_all["trips"] / month_all["year_trips"]) * 100.0
    month_all.to_csv(out_dir / "citibike_trips_by_month_ALL.csv", index=False)

    # month-of-year across all years (seasonality)
    season = (
        month_all.groupby(["mode", "month"], as_index=False)["trips"]
        .sum()
        .sort_values(["mode", "month"])
    )
    season["pct_of_mode_all_trips"] = season.groupby("mode")["trips"].transform(lambda s: s / s.sum() * 100.0)
    season.to_csv(out_dir / "citibike_trips_by_monthOfYear_ALL.csv", index=False)

    # ---- YEAR (combined) ----
    year_all = best.sort_values(["mode", "year"]).copy()
    year_all["months_covered"] = pd.to_numeric(year_all.get("months_covered", np.nan), errors="coerce")

    # Compute YoY using month-level data (optionally filtered to yoy_months)
    if yoy_months:
        month_yoy = month_all[month_all["month"].isin(yoy_months)].copy()
        yoy_basis = f"months={yoy_sig}"
    else:
        month_yoy = month_all.copy()
        yoy_basis = "full_coverage_match_required"

    yoy_agg = (
        month_yoy.groupby(["mode", "year"], as_index=False)
        .agg(yoy_trips=("trips", "sum"), yoy_months_covered=("month", "nunique"))
    )

    # Build a signature of the months actually present after filtering
    yoy_months_set_map = {}
    for (m, y), sub in month_yoy.groupby(["mode", "year"]):
        months = sorted(pd.to_numeric(sub["month"], errors="coerce").dropna().astype(int).unique().tolist())
        yoy_months_set_map[(str(m), int(y))] = months_signature(months) if months else np.nan
    yoy_agg["yoy_months_set"] = [
        yoy_months_set_map.get((str(r["mode"]), int(r["year"])), np.nan) for _, r in yoy_agg.iterrows()
    ]

    year_all = year_all.merge(yoy_agg, on=["mode", "year"], how="left")
    year_all["yoy_basis"] = yoy_basis

    year_all["yoy_pct"] = np.nan
    year_all["trips_per_month"] = np.nan
    year_all["trips_per_month_yoy_pct"] = np.nan

    for mode in year_all["mode"].dropna().unique():
        mode_data = year_all[year_all["mode"] == mode].copy().sort_values(["year"])

        for i in range(1, len(mode_data)):
            prev_idx = mode_data.index[i - 1]
            curr_idx = mode_data.index[i]

            prev = year_all.loc[prev_idx]
            curr = year_all.loc[curr_idx]

            if yoy_months:
                # require both years contain ALL requested months
                prev_ok = (prev.get("yoy_months_set", np.nan) == yoy_sig) and (int(prev.get("yoy_months_covered", 0)) == len(yoy_months))
                curr_ok = (curr.get("yoy_months_set", np.nan) == yoy_sig) and (int(curr.get("yoy_months_covered", 0)) == len(yoy_months))
                comparable = prev_ok and curr_ok
            else:
                # require identical coverage year-to-year (month set equality)
                comparable = (
                    pd.notna(prev.get("yoy_months_set", np.nan))
                    and pd.notna(curr.get("yoy_months_set", np.nan))
                    and str(prev["yoy_months_set"]) == str(curr["yoy_months_set"])
                )

            prev_trips = prev.get("yoy_trips", np.nan)
            curr_trips = curr.get("yoy_trips", np.nan)

            if comparable and pd.notna(prev_trips) and float(prev_trips) > 0 and pd.notna(curr_trips):
                year_all.loc[curr_idx, "yoy_pct"] = ((float(curr_trips) / float(prev_trips)) - 1.0) * 100.0
            else:
                prev_y = int(prev["year"])
                curr_y = int(curr["year"])
                prev_cov = prev.get("yoy_months_set", np.nan)
                curr_cov = curr.get("yoy_months_set", np.nan)
                print(
                    f"⚠️  Cannot compute YoY for {mode} {curr_y}: comparing {curr_cov} to {prev_cov} (prev year {prev_y})"
                )

        # trips_per_month for the same YoY basis
        for idx in mode_data.index:
            row = year_all.loc[idx]
            denom = float(len(yoy_months)) if yoy_months else float(row.get("yoy_months_covered", np.nan))
            if pd.notna(row.get("yoy_trips", np.nan)) and np.isfinite(denom) and denom > 0:
                year_all.loc[idx, "trips_per_month"] = float(row["yoy_trips"]) / denom

        # trips_per_month_yoy_pct only where yoy_pct is valid
        mode_data2 = year_all.loc[mode_data.index].copy().sort_values(["year"])
        for i in range(1, len(mode_data2)):
            prev_idx = mode_data2.index[i - 1]
            curr_idx = mode_data2.index[i]
            if pd.notna(year_all.loc[curr_idx, "yoy_pct"]):
                prev_v = year_all.loc[prev_idx, "trips_per_month"]
                curr_v = year_all.loc[curr_idx, "trips_per_month"]
                if pd.notna(prev_v) and float(prev_v) > 0 and pd.notna(curr_v):
                    year_all.loc[curr_idx, "trips_per_month_yoy_pct"] = ((float(curr_v) / float(prev_v)) - 1.0) * 100.0

    year_all.to_csv(out_dir / "citibike_trips_by_year_ALL.csv", index=False)

    # ---- DOW (combined) ----
    dow_rows = []
    for run_tag in run_tags:
        d = root / run_tag
        dfd = safe_read_csv(d / "citibike_trips_by_dow.csv")
        dfd["run_tag"] = run_tag

        # In older outputs, these sometimes lacked year/mode.
        # If missing, we cannot correctly backfill without knowing which year it refers to,
        # but your pipeline now writes year column, so this should be fine.
        dow_rows.append(dfd)

    dow_all = pd.concat(dow_rows, ignore_index=True)

    # If year/mode columns are missing (legacy), try to backfill from best as a last resort.
    if "year" not in dow_all.columns or "mode" not in dow_all.columns:
        # Legacy fallback: expand per (mode,year) rows in best (can re-introduce duplication),
        # but only used if necessary.
        dow_rows = []
        for _, r in best.iterrows():
            run_tag = r["run_tag"]
            d = root / run_tag
            dfd = safe_read_csv(d / "citibike_trips_by_dow.csv")
            dfd = ensure_year_mode(dfd, int(r["year"]), str(r["mode"]))
            dfd["run_tag"] = run_tag
            dow_rows.append(dfd)
        dow_all = pd.concat(dow_rows, ignore_index=True)

    dow_year_totals = (
        dow_all.groupby(["mode", "year"], as_index=False)["trips"]
        .sum()
        .rename(columns={"trips": "year_trips"})
    )
    dow_all = dow_all.merge(dow_year_totals, on=["mode", "year"], how="left")
    dow_all["pct_of_mode_year_trips"] = (dow_all["trips"] / dow_all["year_trips"]) * 100.0
    dow_all.to_csv(out_dir / "citibike_trips_by_dow_ALL.csv", index=False)

    # ---- HOUR (combined) ----
    hour_rows = []
    for run_tag in run_tags:
        d = root / run_tag
        dfh = safe_read_csv(d / "citibike_trips_by_hour.csv")
        dfh["run_tag"] = run_tag
        hour_rows.append(dfh)

    hour_all = pd.concat(hour_rows, ignore_index=True)

    if "year" not in hour_all.columns or "mode" not in hour_all.columns:
        # Legacy fallback (see comment in DOW section)
        hour_rows = []
        for _, r in best.iterrows():
            run_tag = r["run_tag"]
            d = root / run_tag
            dfh = safe_read_csv(d / "citibike_trips_by_hour.csv")
            dfh = ensure_year_mode(dfh, int(r["year"]), str(r["mode"]))
            dfh["run_tag"] = run_tag
            hour_rows.append(dfh)
        hour_all = pd.concat(hour_rows, ignore_index=True)

    hour_year_totals = (
        hour_all.groupby(["mode", "year"], as_index=False)["trips"]
        .sum()
        .rename(columns={"trips": "year_trips"})
    )
    hour_all = hour_all.merge(hour_year_totals, on=["mode", "year"], how="left")
    hour_all["pct_of_mode_year_trips"] = (hour_all["trips"] / hour_all["year_trips"]) * 100.0
    hour_all.to_csv(out_dir / "citibike_trips_by_hour_ALL.csv", index=False)

    print("Saved comparison outputs to:", out_dir)
    print(" - citibike_trips_by_year_ALL.csv")
    print(" - citibike_trips_by_month_ALL.csv + citibike_trips_by_monthOfYear_ALL.csv")
    print(" - citibike_trips_by_dow_ALL.csv")
    print(" - citibike_trips_by_hour_ALL.csv")
    print(" - selected_best_runs.csv")
    if yoy_months:
        print(f"YoY computed on subset months: {yoy_sig}")


if __name__ == "__main__":
    main()
