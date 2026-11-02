"""Validate TLC trips and build pickup-hour targets and completion-time observations."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import duckdb
import geopandas as gpd
import pandas as pd


def sql_string(value: Path | str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def build_zones(raw: Path, out: Path) -> pd.DataFrame:
    archive = (raw / "taxi_zones.zip").resolve()
    with zipfile.ZipFile(archive) as z:
        shape = next(name for name in z.namelist() if name.lower().endswith(".shp"))
    geo = gpd.read_file(f"zip://{archive}!{shape}")
    lookup = pd.read_csv(raw / "taxi_zone_lookup.csv")
    geo = geo.rename(columns={"LocationID": "zone_id"})
    geo["zone_id"] = geo["zone_id"].astype(int)
    lookup = lookup.rename(columns={"LocationID": "zone_id", "Borough": "borough", "Zone": "zone"})
    # Calculate centroids in the source's projected CRS, then transform to WGS84.
    centers = gpd.GeoSeries(geo.geometry.centroid, crs=geo.crs).to_crs(4326)
    geo = geo.to_crs(4326)
    geo["longitude"], geo["latitude"] = centers.x, centers.y
    geo = geo[["zone_id", "longitude", "latitude", "geometry"]].merge(lookup[["zone_id", "borough", "zone"]], on="zone_id", how="inner")
    geo = geo.dropna(subset=["geometry", "longitude", "latitude"]).sort_values("zone_id")
    out.mkdir(parents=True, exist_ok=True)
    geo.to_file(out / "zones.geojson", driver="GeoJSON")
    zones = pd.DataFrame(geo.drop(columns="geometry"))
    zones.to_parquet(out / "zones.parquet", index=False)
    return zones


def dst_bad(column: str) -> str:
    # TLC timestamps have no UTC offset. Ambiguous fall-back and nonexistent spring
    # hours cannot be resolved from the record, so discard them conservatively.
    return (f"((month({column}) = 3 AND dayofweek({column}) = 0 AND day({column}) BETWEEN 8 AND 14 AND hour({column}) = 2) "
            f"OR (month({column}) = 11 AND dayofweek({column}) = 0 AND day({column}) BETWEEN 1 AND 7 AND hour({column}) = 1))")


def process_month(con: duckdb.DuckDBPyConnection, source: Path, out: Path) -> dict:
    month = source.stem.rsplit("_", 1)[-1]
    schema = {r[0].lower(): r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet({sql_string(source)})").fetchall()}
    needed = ["tpep_pickup_datetime", "tpep_dropoff_datetime", "pulocationid", "dolocationid", "trip_distance", "fare_amount"]
    missing = [x for x in needed if x not in schema]
    if missing:
        raise ValueError(f"{source.name}: missing required columns: {missing}")
    def col(name: str, kind: str) -> str:
        return f'TRY_CAST("{schema[name]}" AS {kind})'
    con.execute(f"""CREATE OR REPLACE TEMP VIEW typed AS SELECT
        {col('tpep_pickup_datetime','TIMESTAMP')} AS pickup,
        {col('tpep_dropoff_datetime','TIMESTAMP')} AS dropoff,
        {col('pulocationid','INTEGER')} AS src, {col('dolocationid','INTEGER')} AS dst,
        {col('trip_distance','DOUBLE')} AS distance, {col('fare_amount','DOUBLE')} AS fare
        FROM read_parquet({sql_string(source)})""")
    con.execute(f"""CREATE OR REPLACE TEMP VIEW zoned AS SELECT *,
        epoch(timezone('America/New_York', pickup)) AS pickup_epoch,
        epoch(timezone('America/New_York', dropoff)) AS dropoff_epoch
        FROM typed WHERE pickup IS NOT NULL AND dropoff IS NOT NULL
        AND strftime(pickup, '%Y-%m') = {sql_string(month)}
        AND NOT {dst_bad('pickup')} AND NOT {dst_bad('dropoff')}
        AND src IN (SELECT zone_id FROM zones) AND dst IN (SELECT zone_id FROM zones)""")
    con.execute("""CREATE OR REPLACE TEMP VIEW clean AS SELECT *,
        (dropoff_epoch-pickup_epoch)/60.0 AS duration,
        CAST(floor(pickup_epoch/3600)*3600 AS BIGINT) AS pickup_hour,
        CAST((floor(dropoff_epoch/3600)+1)*3600 AS BIGINT) AS available_at,
        CAST(floor(minute(pickup)/15) AS INTEGER) AS departure_bin
        FROM zoned WHERE dropoff_epoch > pickup_epoch AND dropoff_epoch-pickup_epoch <= 10800
        AND distance BETWEEN 0 AND 100 AND fare BETWEEN 0 AND 500
        AND isfinite(distance) AND isfinite(fare)""")
    raw_count = con.execute("SELECT count(*) FROM typed").fetchone()[0]
    zone_count = con.execute("SELECT count(*) FROM zoned").fetchone()[0]
    clean_count = con.execute("SELECT count(*) FROM clean").fetchone()[0]
    target = out / "targets" / f"{month}.parquet"
    observed = out / "observations" / f"{month}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    observed.parent.mkdir(parents=True, exist_ok=True)
    target_tmp = target.with_suffix(".parquet.tmp")
    observed_tmp = observed.with_suffix(".parquet.tmp")
    con.execute(f"""COPY (SELECT pickup_hour, src, dst, count(*)::BIGINT AS trip_count,
        avg(duration) AS duration, avg(fare) AS fare, avg(distance) AS distance,
        count(*) FILTER (WHERE departure_bin=0)::BIGINT AS bin0,
        count(*) FILTER (WHERE departure_bin=1)::BIGINT AS bin1,
        count(*) FILTER (WHERE departure_bin=2)::BIGINT AS bin2,
        count(*) FILTER (WHERE departure_bin=3)::BIGINT AS bin3
        FROM clean GROUP BY pickup_hour, src, dst ORDER BY pickup_hour, src, dst)
        TO {sql_string(target_tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    con.execute(f"""COPY (SELECT available_at, pickup_hour, src, dst, count(*)::BIGINT AS trip_count,
        sum(duration) AS duration_sum, sum(fare) AS fare_sum, sum(distance) AS distance_sum,
        count(*) FILTER (WHERE departure_bin=0)::BIGINT AS bin0,
        count(*) FILTER (WHERE departure_bin=1)::BIGINT AS bin1,
        count(*) FILTER (WHERE departure_bin=2)::BIGINT AS bin2,
        count(*) FILTER (WHERE departure_bin=3)::BIGINT AS bin3
        FROM clean GROUP BY available_at, pickup_hour, src, dst ORDER BY available_at, pickup_hour, src, dst)
        TO {sql_string(observed_tmp)} (FORMAT PARQUET, COMPRESSION ZSTD)""")
    target_tmp.replace(target)
    observed_tmp.replace(observed)
    return {"source": source.name, "rows": raw_count, "invalid_time_zone_or_location": raw_count-zone_count,
            "invalid_duration_distance_or_fare": zone_count-clean_count, "accepted": clean_count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--months", nargs="*", help="Only process specified months")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    raw, out = args.data_dir / "raw", args.data_dir
    zones = build_zones(raw, out)
    files = sorted(raw.glob("yellow_tripdata_*.parquet"))
    if args.months:
        files = [raw / f"yellow_tripdata_{m}.parquet" for m in args.months]
    if not files:
        raise SystemExit("No monthly Parquet files found. Run the downloader first.")
    con = duckdb.connect()
    con.execute(f"SET threads={max(1,args.threads)}")
    con.register("zones", zones)
    reports = []
    for source in files:
        print(f"Processing {source.name}…", flush=True)
        reports.append(process_month(con, source, out))
        print(json.dumps(reports[-1]), flush=True)
    (out / "validation.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
