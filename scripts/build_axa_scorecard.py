#!/usr/bin/env python3
"""
build_axa_scorecard.py

Build an AXA-ready station scorecard from a station risk/exposure table.

Notes:
- Exposure = trips
- Risk proxy = crashes_within_<R>m per 100k trips (raw)
- EB smoothing uses a Poisson-rate shrinkage:
    r0 = total_count/total_exposure   (within EB scope group)
    r_EB = (count + r0*m) / (exposure + m)
- Credibility: risk ranking is only for rows where exposure_trips >= --min-trips.
- Never ranks across modes (nyc/jc are processed separately).

Example (as invoked from Makefile via variables, no hardcoding needed):
  python scripts/build_axa_scorecard.py \
    --in-dir summaries/<RUN_TAG> \
    --out-dir summaries/<RUN_TAG> \
    --risk-file station_risk_exposure_plus_crashproximity_by_year_month.csv \
    --radius 500m \
    --eb-scope mode_year_month \
    --min-trips 5000
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2  # type: ignore
except Exception:
    chi2 = None


# -----------------------------
# Small helpers
# -----------------------------
def poisson_rate_ci(
    count: int,
    exposure: float,
    scale: float = 100_000.0,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """
    Exact Poisson CI for rate=(count/exposure)*scale.
    If scipy isn't available, return (rate, rate).
    """
    if exposure <= 0:
        return float("nan"), float("nan")

    rate = (count / exposure) * scale

    if chi2 is None:
        return rate, rate

    k = int(count)
    if k == 0:
        lam_lo = 0.0
    else:
        lam_lo = 0.5 * chi2.ppf(alpha / 2, 2 * k)
    lam_hi = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (k + 1))

    return (lam_lo / exposure * scale, lam_hi / exposure * scale)


def pct_rank_0_100(s: pd.Series) -> pd.Series:
    # stable percentile rank 0..100
    return s.rank(pct=True, method="average") * 100.0


def coerce_int(s: pd.Series, default: int = 0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)


def coerce_float(s: pd.Series, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(float)


def require_columns(df: pd.DataFrame, cols: Iterable[str], *, context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


# -----------------------------
# Radius parsing / discovery
# -----------------------------
_RADIUS_RE = re.compile(r"^\s*(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>m|km)?\s*$", re.IGNORECASE)


def parse_radius_to_m(raw: str) -> int:
    """
    Parse "500m", "500", "1km", "0.75km" -> integer meters.
    """
    s = str(raw).strip().lower()
    m = _RADIUS_RE.match(s)
    if not m:
        raise ValueError(f"Invalid --radius={raw!r}. Use like 250m, 750m, 750, 1km, or auto.")
    val = float(m.group("num"))
    unit = (m.group("unit") or "m").lower()
    meters = val * (1000.0 if unit == "km" else 1.0)
    if meters <= 0:
        raise ValueError(f"--radius must be > 0 (got {raw!r})")
    return int(round(meters))


def available_radii_in_df(df: pd.DataFrame) -> list[int]:
    radii: set[int] = set()
    for c in df.columns:
        mm = re.match(r"^crashes_within_(\d+)m(?:_per_100k_trips)?$", str(c))
        if mm:
            radii.add(int(mm.group(1)))
    return sorted(radii)


# -----------------------------
# EB smoothing
# -----------------------------
def eb_rate_per_trip(counts: pd.Series, exposures: pd.Series, m_prior: float) -> pd.Series:
    """
    EB posterior mean for Poisson rates:
      r0 = total_count/total_exposure
      r_EB = (count + r0*m) / (exposure + m)
    """
    exposures = exposures.astype(float)
    counts = counts.astype(float)

    total_exposure = float(exposures.sum())
    if total_exposure <= 0:
        return pd.Series(np.nan, index=counts.index)

    baseline_rate_per_trip = float(counts.sum() / total_exposure)
    prior_count = baseline_rate_per_trip * float(m_prior)

    return (counts + prior_count) / (exposures + float(m_prior))


def compute_optimal_eb_prior(counts: pd.Series, exposures: pd.Series) -> float:
    """
    Method-of-moments estimate for EB prior strength m (bounded).

    Fallback for small groups (<30 rows): m=20,000 (pseudo-trips).
    This is intentionally conservative and prevents tiny-denominator noise
    from dominating when the group is too small to estimate m stably.
    """
    valid = (exposures > 0) & (counts >= 0)
    n_valid = int(valid.sum())
    if n_valid < 30:
        print("  WARNING: <30 valid observations for EB prior estimation -> using m=20,000")
        return 20000.0

    counts_v = counts[valid].astype(float)
    exposures_v = exposures[valid].astype(float)
    rates = counts_v / exposures_v

    mean_rate = float(rates.mean())
    var_observed = float(rates.var(ddof=1))

    mean_exposure = float(exposures_v.mean())
    var_sampling = (mean_rate / mean_exposure) if mean_exposure > 0 else 0.0

    var_true = max(1e-10, var_observed - var_sampling)

    if var_true < 1e-10 or not np.isfinite(var_true):
        print("  INFO: No detectable true variance -> using strong prior m=100,000")
        return 100000.0

    m_est = (mean_rate**2) / var_true
    m_bounded = float(max(1000.0, min(100000.0, m_est)))

    print(f"  EB prior auto-calibration: m_est={m_est:.1f} -> m={m_bounded:.1f}")
    return m_bounded


# -----------------------------
# EB scope handling
# -----------------------------
@dataclass(frozen=True)
class EBScope:
    name: str
    keys: tuple[str, ...]

    @staticmethod
    def from_arg(arg: str) -> "EBScope":
        a = str(arg).strip().lower()
        if a == "mode":
            return EBScope("mode", ("mode",))
        if a == "mode_year":
            return EBScope("mode_year", ("mode", "year"))
        if a == "mode_year_month":
            return EBScope("mode_year_month", ("mode", "year", "month"))
        raise ValueError("Invalid --eb-scope. Use one of: mode, mode_year, mode_year_month")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build AXA partner scorecard from station risk exposure data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in-dir", required=True, help="Run summary directory, e.g. summaries/<RUN_TAG>")
    ap.add_argument("--out-dir", required=True, help="Output directory (usually same as --in-dir)")
    ap.add_argument(
        "--risk-file",
        default="station_risk_exposure_plus_crashproximity.csv",
        help="Input CSV filename inside --in-dir (default: station_risk_exposure_plus_crashproximity.csv)",
    )
    ap.add_argument(
        "--radius",
        default="500m",
        help="Crash proximity radius (e.g. 250m, 750m, 750, 1km) or 'auto'/'max' (max available).",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for confidence intervals (default 0.05 -> 95%% CI)")
    ap.add_argument("--min-trips", type=int, default=5_000, help="Minimum trips for credible risk ranking")
    ap.add_argument("--m-prior", type=float, default=None, help="EB prior strength (pseudo-trips). If omitted, auto-calibrated.")
    ap.add_argument(
        "--eb-scope",
        default="mode",
        help="EB grouping: mode (default), mode_year, mode_year_month",
    )
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    risk_path = in_dir / str(args.risk_file)
    if not risk_path.exists():
        raise FileNotFoundError(f"Missing required input: {risk_path}")

    df = pd.read_csv(risk_path)

    # Resolve radius
    radius_raw = str(args.radius).strip()
    if radius_raw.lower() in {"auto", "max"}:
        radii = available_radii_in_df(df)
        if not radii:
            raise ValueError(
                f"No crash proximity columns found in {risk_path.name}. "
                "Did you run summarize with NYPD + --radii-m?"
            )
        radius_m = max(radii)
        print(f"Auto-selected radius: {radius_m}m")
    else:
        radius_m = parse_radius_to_m(radius_raw)

    crash_col = f"crashes_within_{radius_m}m"
    raw_rate_col = f"{crash_col}_per_100k_trips"

    # EB scope selection
    scope = EBScope.from_arg(args.eb_scope)
    if scope.name in {"mode_year", "mode_year_month"}:
        require_columns(df, ["year"], context=f"--eb-scope {scope.name}")
    if scope.name == "mode_year_month":
        require_columns(df, ["month"], context=f"--eb-scope {scope.name}")

    # Validate required columns for scorecard
    required = [
        "mode",
        "start_station_id",
        "start_station_name",
        "station_lat",
        "station_lng",
        "trips",
        crash_col,
        raw_rate_col,
    ]
    require_columns(df, required, context=risk_path.name)

    # Clean types
    df["mode"] = df["mode"].astype(str)
    df["trips"] = coerce_int(df["trips"], default=0)
    df[crash_col] = coerce_int(df[crash_col], default=0)
    df[raw_rate_col] = coerce_float(df[raw_rate_col], default=0.0)

    if "year" in df.columns:
        df["year"] = coerce_int(df["year"], default=-1)
    if "month" in df.columns:
        df["month"] = coerce_int(df["month"], default=-1)

    # Base fields
    df["exposure_trips"] = df["trips"]
    df["crash_count"] = df[crash_col]
    df["risk_rate_per_100k_trips"] = df[raw_rate_col]

    # Poisson CI for raw rate
    ci = df.apply(
        lambda r: poisson_rate_ci(int(r["crash_count"]), float(r["exposure_trips"]), alpha=args.alpha),
        axis=1,
    )
    df["risk_rate_ci_low"] = [x[0] for x in ci]
    df["risk_rate_ci_high"] = [x[1] for x in ci]

    # Exposure percentile (within mode, to match original intent)
    df["exposure_index_pct"] = df.groupby("mode", dropna=False)["exposure_trips"].transform(pct_rank_0_100)

    min_trips = int(args.min_trips)
    out_parts: list[pd.DataFrame] = []

    # Never rank across modes: process each mode independently.
    for mode, g_mode in df.groupby("mode", dropna=False):
        g_mode = g_mode.copy()
        print(f"\n{'='*70}\nProcessing mode: {mode}\n{'='*70}")

        # Credibility for ranking
        sufficient = g_mode["exposure_trips"] >= min_trips
        g_mode["credibility_flag"] = "insufficient_data"
        g_mode.loc[sufficient, "credibility_flag"] = "credible"

        # Decide whether risk proxy has usable signal among credible rows
        if int(sufficient.sum()) < 10:
            risk_has_signal = False
        else:
            risk_has_signal = (g_mode.loc[sufficient, "risk_rate_per_100k_trips"].nunique(dropna=True) > 1)

        g_mode["risk_proxy_available"] = bool(risk_has_signal)

        if not risk_has_signal:
            # Exposure-only fallback (same philosophy as original)
            g_mode["eb_risk_rate_per_100k_trips"] = np.nan
            g_mode["risk_index_pct"] = np.nan
            g_mode["expected_incidents_proxy"] = np.nan

            g_mode["exposure_pct"] = g_mode["exposure_index_pct"] / 100.0
            g_mode["risk_pct"] = 0.0
            g_mode["axa_priority_score"] = g_mode["exposure_pct"]
            g_mode["scoring_strategy"] = "exposure_only_no_risk_signal"

            g_mode["prevention_hotspot"] = g_mode["exposure_index_pct"] >= 90.0
            g_mode["product_hotspot"] = g_mode["exposure_index_pct"] >= 80.0
            g_mode["acquisition_hotspot"] = g_mode["exposure_index_pct"] >= 70.0

            out_parts.append(g_mode)
            continue

        # EB within scope groups (inside mode)
        inner_keys = tuple(k for k in scope.keys if k != "mode")  # already split by mode
        if not inner_keys:
            grouped = [((), g_mode)]
        else:
            grouped = list(g_mode.groupby(list(inner_keys), dropna=False))

        eb_per_trip_all = pd.Series(index=g_mode.index, dtype=float)
        m_used_all = pd.Series(index=g_mode.index, dtype=float)

        for key, gg in grouped:
            gg = gg.copy()
            if args.m_prior is None:
                m_prior = compute_optimal_eb_prior(gg["crash_count"], gg["exposure_trips"])
            else:
                m_prior = float(args.m_prior)
                # don't spam prints for every subgroup if user explicitly set it

            eb_per_trip = eb_rate_per_trip(gg["crash_count"], gg["exposure_trips"], m_prior)
            eb_per_trip_all.loc[gg.index] = eb_per_trip
            m_used_all.loc[gg.index] = m_prior

            if inner_keys and args.m_prior is None:
                key_tuple = key if isinstance(key, tuple) else (key,)
                key_str = ", ".join(f"{k}={v}" for k, v in zip(inner_keys, key_tuple))
                print(f"  EB scope {scope.name}: {key_str} -> m={m_prior:.1f}")

        g_mode["eb_m_prior_used"] = m_used_all
        g_mode["eb_risk_rate_per_100k_trips"] = eb_per_trip_all * 100_000.0

        # Risk ranking ONLY for credible rows (within mode)
        g_mode["risk_index_pct"] = np.nan
        g_mode.loc[sufficient, "risk_index_pct"] = pct_rank_0_100(g_mode.loc[sufficient, "eb_risk_rate_per_100k_trips"])

        # Expected incidents proxy (freq × exposure)
        g_mode["expected_incidents_proxy"] = eb_per_trip_all * g_mode["exposure_trips"]

        g_mode["exposure_pct"] = g_mode["exposure_index_pct"] / 100.0
        g_mode["risk_pct"] = g_mode["risk_index_pct"].fillna(0.0) / 100.0

        # Priority score: percentile of expected incidents within mode
        g_mode["axa_priority_score"] = pct_rank_0_100(g_mode["expected_incidents_proxy"]) / 100.0
        g_mode["scoring_strategy"] = (
            f"eb_expected_incidents_mintrips{min_trips}_"
            f"{scope.name}_"
            + ("mpriorAUTO" if args.m_prior is None else f"mprior{int(args.m_prior)}")
        )

        # Hotspot heuristics (kept consistent with your original approach)
        g_mode["prevention_hotspot"] = (
            (g_mode["exposure_index_pct"] >= 80.0)
            & (g_mode["risk_pct"] >= 0.8)
            & (g_mode["credibility_flag"] == "credible")
        )
        g_mode["product_hotspot"] = g_mode["exposure_index_pct"] >= 80.0
        g_mode["acquisition_hotspot"] = (g_mode["exposure_index_pct"] >= 70.0) & (g_mode["risk_pct"] <= 0.3)

        out_parts.append(g_mode)

    out_all = pd.concat(out_parts, axis=0, ignore_index=True)

    # Output columns (include year/month if present)
    out_cols = [
        "mode",
        "year",
        "month",
        "start_station_id",
        "start_station_name",
        "station_lat",
        "station_lng",
        "exposure_trips",
        "crash_count",
        "risk_rate_per_100k_trips",
        "risk_rate_ci_low",
        "risk_rate_ci_high",
        "risk_proxy_available",
        "credibility_flag",
        "exposure_pct",
        "risk_pct",
        "axa_priority_score",
        "prevention_hotspot",
        "product_hotspot",
        "acquisition_hotspot",
        "exposure_index_pct",
        "eb_risk_rate_per_100k_trips",
        "risk_index_pct",
        "expected_incidents_proxy",
        "scoring_strategy",
        "eb_m_prior_used",
    ]
    out_cols = [c for c in out_cols if c in out_all.columns]
    out = out_all[out_cols].copy()

    out = out.sort_values(["axa_priority_score", "exposure_trips"], ascending=False).reset_index(drop=True)

    out_path = out_dir / f"axa_partner_scorecard_{radius_m}m.csv"
    out.to_csv(out_path, index=False)

    print(f"\nWrote: {out_path}")
    print(f"Total rows: {len(out):,}")
    if scope.name != "mode":
        print(f"EB scope used: {scope.name}")
    if "year" in out.columns or "month" in out.columns:
        print("NOTE: output is station-period rows if year/month were present in the input risk file.")


if __name__ == "__main__":
    main()
