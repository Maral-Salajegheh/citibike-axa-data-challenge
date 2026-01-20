#!/usr/bin/env python3
"""
download_tripdata.py

Robust Citi Bike tripdata downloader (NYC/JC) that does NOT rely on index.html.

Why:
- tripdata "index.html" is often a JS-based bucket browser (no ZIP names in HTML source)
- Instead, we read the public S3 listing XML:
    https://tripdata.s3.amazonaws.com/?list-type=2&max-keys=1000

Features:
- Lists bucket keys via XML (paginated)
- Selects files by mode/year/month patterns
- NYC yearly fallback (e.g., 2023-citibike-tripdata.zip)
- --dry-run and --debug for sanity checks

Examples:
  python scripts/download_tripdata.py --years 2023 --months 1 --mode nyc --out-dir /tmp/test --dry-run --debug
  python scripts/download_tripdata.py --years 2023 --months 1 2 3 --mode jc --out-dir data/raw/jc_2023
  python scripts/download_tripdata.py --years 2023 --months 1 2 3 4 5 6 7 8 9 10 11 12 --mode nyc \
      --out-dir data/raw/nyc_2023 --allow-yearly-fallback --force
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen


DEFAULT_BUCKET_URLS = (
    # Prefer virtual-hosted-style
    "https://tripdata.s3.amazonaws.com/",
    # Fallback path-style
    "https://s3.amazonaws.com/tripdata/",
)


@dataclass(frozen=True)
class ListingResult:
    bucket_url: str
    zip_filenames: List[str]


def _http_get_bytes(url: str, timeout: float) -> bytes:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (download_tripdata.py)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_get_text(url: str, timeout: float) -> str:
    return _http_get_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def _strip_ns(tag: str) -> str:
    # {namespace}Tag -> Tag
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _list_bucket_zip_filenames(
    bucket_url: str,
    timeout: float,
    debug: bool,
    max_pages: int = 200,
) -> List[str]:
    """
    Uses S3 ListObjectsV2:
      GET <bucket_url>/?list-type=2&max-keys=1000[&continuation-token=...]
    Returns unique sorted list of ZIP basenames found in <Key>.
    """
    bucket_url = bucket_url.rstrip("/") + "/"
    token: Optional[str] = None
    zips: List[str] = []
    seen = set()

    for page in range(1, max_pages + 1):
        url = f"{bucket_url}?list-type=2&max-keys=1000"
        if token:
            url += f"&continuation-token={quote(token, safe='')}"

        xml_text = _http_get_text(url, timeout=timeout)

        # Common failure modes show HTML / AccessDenied; detect early
        if "<ListBucketResult" not in xml_text and "<Error>" in xml_text:
            raise RuntimeError(f"S3 listing error from {bucket_url}: {xml_text[:400]}")
        if "<ListBucketResult" not in xml_text and "<html" in xml_text.lower():
            raise RuntimeError(
                f"S3 listing returned HTML (unexpected). First 400 chars: {xml_text[:400]}"
            )

        root = ET.fromstring(xml_text)

        # Collect keys
        for child in root.iter():
            if _strip_ns(child.tag) == "Key" and child.text:
                key = child.text.strip()
                if key.lower().endswith(".zip"):
                    fn = key.rsplit("/", 1)[-1]
                    if fn not in seen:
                        seen.add(fn)
                        zips.append(fn)

        is_truncated = False
        next_token = None
        for child in root.iter():
            t = _strip_ns(child.tag)
            if t == "IsTruncated" and child.text:
                is_truncated = child.text.strip().lower() == "true"
            elif t == "NextContinuationToken" and child.text:
                next_token = child.text.strip()

        if debug:
            print(f"[debug] bucket={bucket_url} page={page} zips_so_far={len(zips)} truncated={is_truncated}")

        if not is_truncated:
            break
        token = next_token
        if not token:
            break

    if debug:
        print(f"[debug] total_zip_files_found={len(zips)}")

    return sorted(zips)


def _fetch_listing(bucket_urls: Sequence[str], timeout: float, debug: bool) -> ListingResult:
    last_err: Optional[Exception] = None
    for b in bucket_urls:
        try:
            zips = _list_bucket_zip_filenames(b, timeout=timeout, debug=debug)
            if len(zips) < 50:
                raise RuntimeError(f"Listing succeeded but too few ZIPs ({len(zips)}).")
            if debug:
                print(f"[debug] listing_ok bucket={b} sample={zips[:20]}")
            return ListingResult(bucket_url=b, zip_filenames=zips)
        except Exception as e:
            last_err = e
            if debug:
                print(f"[debug] listing_failed bucket={b} err={e}")
            continue
    raise RuntimeError(f"Failed to list bucket from all candidates. Last error: {last_err}")


def _month_ints(months: Sequence[int]) -> List[int]:
    out = []
    for m in months:
        m = int(m)
        if not (1 <= m <= 12):
            raise ValueError(f"Month out of range 1..12: {m}")
        out.append(m)
    return out


def _pick_first_existing(options: Sequence[str], fn_map_lower_to_real: Dict[str, str]) -> Optional[str]:
    for o in options:
        key = o.lower()
        if key in fn_map_lower_to_real:
            return fn_map_lower_to_real[key]
    return None


def _build_selection(
    filenames: Sequence[str],
    years: Sequence[int],
    months: Sequence[int],
    mode: str,
    allow_yearly_fallback: bool,
) -> Tuple[List[str], List[str]]:
    years_i = [int(y) for y in years]
    months_i = _month_ints(months)
    mode_l = mode.lower()

    fn_map = {fn.lower(): fn for fn in filenames}

    def monthly_variants_nyc(y: int, m: int) -> List[str]:
        ym = f"{y}{m:02d}"
        return [
            f"{ym}-citibike-tripdata.csv.zip",
            f"{ym}-citibike-tripdata.zip",
        ]

    def monthly_variants_jc(y: int, m: int) -> List[str]:
        ym = f"{y}{m:02d}"
        return [
            f"JC-{ym}-citibike-tripdata.csv.zip",
            f"JC-{ym}-citibike-tripdata.zip",
        ]

    # NYC yearly archives commonly exist for older years (and sometimes are the only option)
    def yearly_variants_nyc(y: int) -> List[str]:
        return [
            f"{y}-citibike-tripdata.zip",
            f"{y}-citibike-tripdata.csv.zip",
        ]

    selected: List[str] = []
    missing_msgs: List[str] = []

    if mode_l in ("jc", "any"):
        miss_by_year: Dict[int, List[int]] = {}
        for y in years_i:
            for m in months_i:
                fn = _pick_first_existing(monthly_variants_jc(y, m), fn_map)
                if fn:
                    selected.append(fn)
                else:
                    miss_by_year.setdefault(y, []).append(m)
        for y, ms in sorted(miss_by_year.items()):
            missing_msgs.append(f"Missing JC {y} months: {sorted(ms)}")

    if mode_l in ("nyc", "any"):
        for y in years_i:
            miss_months: List[int] = []
            found_monthly = False
            for m in months_i:
                fn = _pick_first_existing(monthly_variants_nyc(y, m), fn_map)
                if fn:
                    selected.append(fn)
                    found_monthly = True
                else:
                    miss_months.append(m)

            if miss_months:
                if allow_yearly_fallback:
                    yfn = _pick_first_existing(yearly_variants_nyc(y), fn_map)
                    if yfn:
                        selected.append(yfn)
                        missing_msgs.append(
                            f"NYC {y}: monthly missing months={sorted(miss_months)} -> using yearly fallback {yfn}"
                        )
                    else:
                        missing_msgs.append(
                            f"Missing NYC {y} months: {sorted(miss_months)} (no yearly fallback found)"
                        )
                else:
                    missing_msgs.append(f"Missing NYC {y} months: {sorted(miss_months)}")

    # de-dupe in order
    out: List[str] = []
    seen = set()
    for fn in selected:
        if fn not in seen:
            seen.add(fn)
            out.append(fn)

    return out, missing_msgs


def _download_file(url: str, out_path: Path, timeout: float, retries: int, backoff: float, force: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f" - exists, skipping: {out_path}")
        return

    tmp_path = out_path.with_suffix(out_path.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 (download_tripdata.py)"})
            with urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as f:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            tmp_path.replace(out_path)
            print(f" - downloaded: {out_path.name}")
            return
        except Exception as e:
            if attempt == retries:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                raise RuntimeError(f"Failed to download {url} after {retries} attempts: {e}") from e

            sleep_s = backoff * (2 ** (attempt - 1))
            print(f" ! download failed ({attempt}/{retries}) {out_path.name} -> retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Download Citi Bike tripdata ZIPs (NYC/JC) via S3 XML listing.")
    ap.add_argument("--years", nargs="+", type=int, required=True)
    ap.add_argument("--months", nargs="+", type=int, required=True)
    ap.add_argument("--mode", choices=["nyc", "jc", "any"], default="nyc")
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--bucket-url", action="append", default=[], help="Override/add S3 bucket base URL(s). Can repeat.")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--backoff", type=float, default=1.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument(
        "--allow-yearly-fallback",
        action="store_true",
        help="If monthly NYC files are missing, use yearly archive (YYYY-citibike-tripdata.zip) if present.",
    )

    args = ap.parse_args(argv)

    bucket_urls = list(args.bucket_url) if args.bucket_url else []
    for u in DEFAULT_BUCKET_URLS:
        if u not in bucket_urls:
            bucket_urls.append(u)

    listing = _fetch_listing(bucket_urls=bucket_urls, timeout=args.timeout, debug=args.debug)
    filenames = listing.zip_filenames

    selected, missing_msgs = _build_selection(
        filenames=filenames,
        years=args.years,
        months=args.months,
        mode=args.mode,
        allow_yearly_fallback=args.allow_yearly_fallback,
    )

    for msg in missing_msgs:
        print(msg)

    if args.debug:
        # sanity: do we see the year string at all?
        for y in sorted(set(args.years)):
            hits = sum(1 for fn in filenames if str(y) in fn)
            print(f"[debug] filenames_containing_{y}={hits}")

    if not selected:
        print("No files selected. Check year/month availability, mode, and yearly fallback flag.")
        return 2

    print(f"Bucket used: {listing.bucket_url}")
    print(f"Selected {len(selected)} file(s) (mode={args.mode})")

    if args.dry_run:
        for fn in selected:
            url = urljoin(listing.bucket_url, fn)
            print(f" - would download: {url} -> {args.out_dir / fn}")
        return 0

    for fn in selected:
        url = urljoin(listing.bucket_url, fn)
        _download_file(
            url=url,
            out_path=args.out_dir / fn,
            timeout=args.timeout,
            retries=args.retries,
            backoff=args.backoff,
            force=args.force,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
