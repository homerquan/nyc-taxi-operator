"""Download official TLC files, recording successes and failures in a manifest."""
from __future__ import annotations

import argparse
import json
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://d37ci6vzurychx.cloudfront.net"
INDEX = "https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page"
ASSETS = {"taxi_zones.zip": f"{BASE}/misc/taxi_zones.zip", "taxi_zone_lookup.csv": f"{BASE}/misc/taxi_zone_lookup.csv"}


def monthly_url(month: str) -> str:
    if not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", month):
        raise ValueError(f"Invalid month: {month}")
    return f"{BASE}/trip-data/yellow_tripdata_{month}.parquet"


def months_between(start: str, end: str) -> list[str]:
    monthly_url(start); monthly_url(end)
    y, m = map(int, start.split("-"))
    output = []
    while f"{y:04d}-{m:02d}" <= end:
        output.append(f"{y:04d}-{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    if not output:
        raise ValueError("Start month must precede end month")
    return output


def session() -> requests.Session:
    s = requests.Session()
    retry = Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers["User-Agent"] = "NYCMobilityPredictor/0.1 (research demo)"
    return s


def published_months(s: requests.Session) -> list[str]:
    response = s.get(INDEX, timeout=60)
    response.raise_for_status()
    return sorted(set(re.findall(r"yellow_tripdata_(\d{4}-\d{2})\.parquet", response.text)), reverse=True)


def valid_file(path: Path) -> bool:
    try:
        if path.suffix == ".parquet":
            return pq.ParquetFile(path).metadata.num_rows > 0
        if path.suffix == ".zip":
            with zipfile.ZipFile(path) as z:
                return z.testzip() is None and any(x.endswith(".shp") for x in z.namelist())
        return path.stat().st_size > 50 and "LocationID" in path.read_text(errors="replace")[:500]
    except Exception:
        return False


def save_manifest(path: Path, entries: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(entries, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def download_one(s: requests.Session, url: str, target: Path, manifest: dict) -> bool:
    entry = {"url": url, "checked_at": datetime.now(timezone.utc).isoformat()}
    if target.exists() and valid_file(target):
        old = manifest.get(target.name, {})
        manifest[target.name] = {**old, **entry, "status": "valid", "size": target.stat().st_size}
        return True
    part = target.with_suffix(target.suffix + ".part")
    try:
        offset = part.stat().st_size if part.exists() else 0
        response = s.get(url, headers={"Range": f"bytes={offset}-"} if offset else {}, stream=True, timeout=(20, 120))
        if offset and response.status_code == 416:
            # The partial file is longer than the remote object or is stale.
            # Preserve it for inspection and retry cleanly instead of looping.
            part.replace(target.with_name(target.name + f".corrupt-{time.time_ns()}"))
            response = s.get(url, stream=True, timeout=(20, 120))
            offset = 0
        response.raise_for_status()
        if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            offset = 0
        with part.open("ab" if offset else "wb") as f:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)
        # Validate using the final extension while leaving an existing bad file untouched.
        candidate = target.with_name(target.stem + ".download" + target.suffix)
        part.replace(candidate)
        if not valid_file(candidate):
            candidate.replace(target.with_name(target.name + f".corrupt-{time.time_ns()}"))
            raise ValueError("Downloaded file failed integrity validation")
        candidate.replace(target)
        manifest[target.name] = {**entry, "status": "valid", "size": target.stat().st_size, "downloaded_at": datetime.now(timezone.utc).isoformat()}
        print(f"Downloaded {target.name}: {target.stat().st_size:,} bytes", flush=True)
        return True
    except (requests.RequestException, OSError, ValueError) as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        manifest[target.name] = {**entry, "status": "unavailable" if status == 404 else "error", "error": str(exc), "size": part.stat().st_size if part.exists() else 0}
        print(f"Could not download {target.name}: {exc}", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--latest-36", action="store_true")
    group.add_argument("--months", nargs="+", help="Explicit months, e.g. 2025-01 2025-02")
    group.add_argument("--dev", action="store_true", help="One most recently published month")
    parser.add_argument("--start", default="2023-10")
    parser.add_argument("--end", default="2026-09")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    root = args.data_dir / "raw"
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    s = session()
    if args.dev or args.latest_36:
        months = published_months(s)[:1 if args.dev else 36]
        if len(months) < (1 if args.dev else 36):
            raise SystemExit("TLC index did not expose enough published months")
    else:
        months = args.months or months_between(args.start, args.end)
    files = {f"yellow_tripdata_{month}.parquet": monthly_url(month) for month in months} | ASSETS
    ok = True
    for name, url in files.items():
        success = download_one(s, url, root / name, manifest)
        ok = success and ok
        save_manifest(manifest_path, manifest)
    if not ok:
        raise SystemExit("Some files failed or are unavailable. See the manifest; rerun to retry.")


if __name__ == "__main__":
    main()
