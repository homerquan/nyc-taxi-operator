from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd
import pytest
import torch

from mobility.model import TaxiTGN
from mobility.preprocess import process_month


@pytest.fixture
def dataset(tmp_path: Path):
    data = tmp_path / "data"
    processed = data / "processed"
    processed.mkdir(parents=True)
    zones = pd.DataFrame({"zone_id": [1, 2, 3], "zone": ["Alpha", "Beta", "Gamma"], "borough": ["Manhattan"]*3,
                          "longitude": [-73.99, -73.98, -73.97], "latitude": [40.75, 40.76, 40.77]})
    zones.to_parquet(processed / "zones.parquet", index=False)
    features = [{"type":"Feature", "properties":r, "geometry":{"type":"Polygon", "coordinates":[[[r["longitude"],r["latitude"]], [r["longitude"]+0.001,r["latitude"]], [r["longitude"]+0.001,r["latitude"]+0.001], [r["longitude"],r["latitude"]]]]}} for r in zones.to_dict("records")]
    (processed / "zones.geojson").write_text(json.dumps({"type":"FeatureCollection", "features": features}))
    raw = data / "raw"
    raw.mkdir()
    trips = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2025-01-01 10:00", "2025-01-01 10:15", "2025-01-01 10:55", "2025-01-01 10:10", "2025-01-01 10:00", "2025-01-01 10:00", "2025-01-01 10:00"]),
        "tpep_dropoff_datetime": pd.to_datetime(["2025-01-01 10:20", "2025-01-01 10:45", "2025-01-01 11:25", "2025-01-01 10:35", "2025-01-01 10:20", "2025-01-01 09:30", "2025-01-01 10:20"]),
        "PULocationID": [1, 1, 1, 2, 999, 1, 1], "DOLocationID": [2, 2, 2, 1, 2, 2, 2],
        "trip_distance": [2., 3., 4., 1., 2., 2., 2.], "fare_amount": [10.,20.,30.,7.,10.,10.,-1.],
        "total_amount": [100.,200.,300.,70.,100.,100.,100.], "passenger_count": [1]*7,
    })
    path = raw / "yellow_tripdata_2025-01.parquet"
    trips.to_parquet(path, index=False)
    con = duckdb.connect()
    con.register("zones", zones)
    report = process_month(con, path, processed)
    con.close()
    return data, zones, report


@pytest.fixture
def checkpoint(dataset, tmp_path: Path):
    _, zones, _ = dataset
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    torch.manual_seed(2)
    model = TaxiTGN(zones, memory_dim=8, time_dim=4, hidden_dim=8)
    model.memory.reset_state()
    config = {"memory_dim":8, "time_dim":4, "hidden_dim":8, "seed":2}
    meta = {"zone_ids":[1,2,3], "config":config, "validation_end":1735743600, "test_end":1735758000, "start":1735743600}
    torch.save({"state_dict":model.state_dict(), "metadata":meta}, artifacts / "model.pt")
    return artifacts
