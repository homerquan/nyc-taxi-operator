import io
import json
import zipfile
from datetime import datetime
from pathlib import Path

import geopandas as gpd
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from shapely.geometry import Polygon

from mobility.download import download_one, monthly_url, months_between, save_manifest, valid_file
from mobility.graph import GraphContext, HOUR, candidates, hour_groups, lag_time
from mobility.inference import forecast, load_model, replay
from mobility.model import multitask_loss
from mobility.preprocess import build_zones, process_month
from mobility.training import label_available_by_boundary


def test_months_download_resume_and_manifest(tmp_path):
    assert len(months_between("2023-10", "2026-09")) == 36
    assert monthly_url("2025-01").endswith("yellow_tripdata_2025-01.parquet")
    with pytest.raises(ValueError):
        monthly_url("2025-13")
    content = io.BytesIO()
    pq.write_table(pa.table({"value": [1,2]}), content)
    payload = content.getvalue()
    target = tmp_path / "example.parquet"
    target.with_suffix(".parquet.part").write_bytes(payload[:15])
    class Response:
        status_code = 206
        headers = {"Content-Range": f"bytes 15-{len(payload)-1}/{len(payload)}"}
        def raise_for_status(self): pass
        def iter_content(self, _): yield payload[15:]
    class Session:
        def get(self, url, **kwargs):
            assert kwargs["headers"]["Range"] == "bytes=15-"
            return Response()
    manifest = {}
    assert download_one(Session(), "https://example.com/example.parquet", target, manifest)
    assert valid_file(target)
    save_manifest(tmp_path / "manifest.json", manifest)
    assert json.loads((tmp_path / "manifest.json").read_text())[target.name]["status"] == "valid"
    target.write_bytes(b"corrupt")
    assert not valid_file(target)


def test_geometry_lookup_and_wgs84(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    shape_dir = tmp_path / "shapes"
    shape_dir.mkdir()
    geo = gpd.GeoDataFrame({"LocationID": [1], "geometry": [Polygon([(990000,210000),(990100,210000),(990100,210100),(990000,210000)])]}, crs="EPSG:2263")
    geo.to_file(shape_dir / "taxi_zones.shp")
    with zipfile.ZipFile(raw / "taxi_zones.zip", "w") as archive:
        for path in shape_dir.iterdir():
            archive.write(path, f"nested/{path.name}")
    pd.DataFrame({"LocationID":[1],"Borough":["Manhattan"],"Zone":["Alpha"]}).to_csv(raw / "taxi_zone_lookup.csv",index=False)
    zones = build_zones(raw, tmp_path / "processed")
    assert zones.iloc[0].zone == "Alpha"
    assert -75 < zones.iloc[0].longitude < -72
    assert 40 < zones.iloc[0].latitude < 42


def test_cleaning_directed_edges_targets_and_availability(dataset):
    data, _, report = dataset
    assert report["rows"] == 7 and report["accepted"] == 4
    targets = pd.read_parquet(data / "processed/targets/2025-01.parquet")
    route = targets[(targets.src == 1) & (targets.dst == 2)].iloc[0]
    assert route.trip_count == 3 and route.duration == pytest.approx(80/3)
    assert route.fare == 20  # fare_amount, not total_amount
    assert [route.bin0,route.bin1,route.bin2,route.bin3] == [1,1,0,1]
    assert targets[(targets.src == 2) & (targets.dst == 1)].iloc[0].trip_count == 1
    obs = list(hour_groups(data, "observations"))
    assert [h for h,_ in obs] == sorted(h for h,_ in obs)
    first = obs[0][1]
    assert first[(first.src == 1) & (first.dst == 2)].trip_count.sum() == 2
    assert obs[1][1].trip_count.sum() == 1


def test_temporal_state_loss_and_checkpoint_determinism(dataset, checkpoint):
    data, zones, _ = dataset
    model, _, _, device = load_model(data, checkpoint)
    cutoff = int(datetime.fromisoformat("2025-01-01T11:00:00-05:00").timestamp())
    ctx = replay(model, zones, data, cutoff, device)
    assert ctx.counts(cutoff, np.array([0]), np.array([1]), (1,))[0,0] == 2
    obs = list(hour_groups(data, "observations"))
    with pytest.raises(ValueError, match="available"):
        ctx.observe(model, obs[-1][0], obs[-1][1], cutoff)
    target = next(hour_groups(data, "targets"))[1]
    chosen, src, dst = candidates(target, ctx, np.random.default_rng(1), 20, 1)
    positives = {(int(row.src), int(row.dst)) for row in target.itertuples()}
    assert all((ctx.ids[int(s)],ctx.ids[int(d)]) not in positives for s,d in zip(src[len(chosen):],dst[len(chosen):]))
    out = model.score(model.encode(ctx, cutoff), ctx, cutoff, src, dst)
    loss = multitask_loss(out, chosen)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.link.weight.grad is not None
    one, _, _ = forecast(data, checkpoint, cutoff, top_k=9)
    two, _, _ = forecast(data, checkpoint, cutoff, top_k=9)
    assert one == two
    with pytest.raises(ValueError, match="leak"):
        forecast(data, checkpoint, cutoff-2*HOUR)


def test_daily_and_weekly_lags_follow_local_time_across_dst():
    cutoff = int(datetime.fromisoformat("2025-03-10T09:00:00-04:00").timestamp())
    previous_week = int(datetime.fromisoformat("2025-03-03T09:00:00-05:00").timestamp())
    assert lag_time(cutoff, 168) == previous_week
    assert cutoff - previous_week == 167 * HOUR


def test_ambiguous_fall_hour_is_discarded(tmp_path):
    source = tmp_path / "yellow_tripdata_2025-11.parquet"
    pd.DataFrame({"tpep_pickup_datetime":pd.to_datetime(["2025-11-02 01:15", "2025-11-02 02:15"]),
                  "tpep_dropoff_datetime":pd.to_datetime(["2025-11-02 01:30", "2025-11-02 02:30"]),
                  "PULocationID":[1,1], "DOLocationID":[2,2], "trip_distance":[1.,1.], "fare_amount":[10.,10.]}).to_parquet(source)
    con = duckdb.connect()
    con.register("zones", pd.DataFrame({"zone_id":[1,2]}))
    report = process_month(con, source, tmp_path / "processed")
    assert report["accepted"] == 1
    assert report["invalid_time_zone_or_location"] == 1


def test_label_embargo_prevents_split_boundary_leakage():
    boundary = 100 * HOUR
    assert label_available_by_boundary(boundary - 4 * HOUR, boundary)
    assert not label_available_by_boundary(boundary - 3 * HOUR, boundary)
