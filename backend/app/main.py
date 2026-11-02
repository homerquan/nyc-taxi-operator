"""HTTP interface to processed TLC data and replayed TGN forecasts."""
from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
from threading import RLock
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal
from backend.app.jobs import submit, get_job, active_jobs
from mobility.simulation import SimulationConfig, compare_policies

import duckdb
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from mobility.graph import HOUR, bounds, format_time
from mobility.inference import forecast
from mobility.preprocess import sql_string

app = FastAPI(title="NYC Mobility Forecast", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["GET", "POST"], allow_headers=["*"])
torch.set_num_threads(int(os.environ.get("NYC_TORCH_THREADS", "2")))


ROOT = Path(__file__).resolve().parents[2]
_forecast_lock = RLock()
_selected_paths = None
_training_paths = None


def paths():
    if _selected_paths is not None:
        return _selected_paths
    data = Path(os.environ.get("NYC_DATA_DIR", "data"))
    default_artifacts = "artifacts/cuda-full" if (ROOT / "artifacts/cuda-full/model.pt").is_file() else "artifacts"
    artifacts = Path(os.environ.get("NYC_ARTIFACTS_DIR", default_artifacts))
    return (data if data.is_absolute() else ROOT / data,
            artifacts if artifacts.is_absolute() else ROOT / artifacts)


def parse_cutoff(value: str) -> int:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError("timezone offset required")
        epoch = int(dt.timestamp())
        if epoch % HOUR or dt.microsecond:
            raise ValueError("must be aligned to an hour")
        return epoch
    except (ValueError, OverflowError) as exc:
        raise HTTPException(422, f"Invalid cutoff: {exc}") from exc


def data_ready(data):
    processed = data / "processed"
    return ((processed / "zones.geojson").is_file() and (processed / "zones.parquet").is_file()
            and any((processed / "targets").glob("*.parquet"))
            and any((processed / "observations").glob("*.parquet")))


def data_or_503():
    data, artifacts = paths()
    if not data_ready(data):
        raise HTTPException(503, "Processed TLC data is missing. Download and preprocess data first.")
    return data, artifacts


def model_or_503():
    data, artifacts = data_or_503()
    if not (artifacts / "model.pt").is_file():
        raise HTTPException(503, "No trained model checkpoint found. Run training first.")
    return data, artifacts


@app.get("/api/health")
def health():
    data, artifacts = paths()
    return {"status": "ready" if data_ready(data) and (artifacts / "model.pt").is_file() else "setup_required",
            "data_ready": data_ready(data), "model_ready": (artifacts / "model.pt").is_file()}


@app.get("/api/zones")
def zones():
    data, _ = data_or_503()
    return json.loads((data / "processed" / "zones.geojson").read_text())


@app.get("/api/metrics")
def metrics():
    _, artifacts = paths()
    path = artifacts / "metrics.json"
    return json.loads(path.read_text()) if path.is_file() else {"population": None, "models": {}, "message": "Run evaluation to generate held-out metrics."}


@app.get("/api/available-times")
def available_times():
    data, artifacts = model_or_503()
    meta = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=False)["metadata"]
    glob = data / "processed" / "targets" / "*.parquet"
    with duckdb.connect() as con:
        rows = con.execute(f"SELECT DISTINCT pickup_hour FROM read_parquet({sql_string(glob)}) WHERE pickup_hour >= ? AND pickup_hour < ? ORDER BY pickup_hour", [meta["validation_end"], meta.get("test_end", 2**62)]).fetchall()
    _, last = bounds(data)
    return {"times": [format_time(row[0]) for row in rows], "last_observed_pickup_hour": format_time(last),
            "model_selection_end": format_time(meta["validation_end"]), "quick_model": meta.get("quick", False)}


@lru_cache(maxsize=4)
def _cached_forecast(data: str, artifacts: str, checkpoint_mtime: int, cutoff: int):
    return forecast(Path(data), Path(artifacts), cutoff, top_k=100000)


def cached_forecast(data, artifacts, checkpoint_mtime, cutoff):
    with _forecast_lock:
        return _cached_forecast(data, artifacts, checkpoint_mtime, cutoff)


cached_forecast.cache_clear = _cached_forecast.cache_clear


@app.get("/api/forecast")
def get_forecast(as_of: str, horizon_minutes: int = 60, pickup_zone: int | None = None,
                 top_k: int = Query(100, ge=1, le=500), min_trips: float = Query(0, ge=0)):
    if horizon_minutes != 60:
        raise HTTPException(422, "Only the 60-minute horizon is supported")
    data, artifacts = model_or_503()
    cutoff = parse_cutoff(as_of)
    try:
        predictions, meta, summary = cached_forecast(str(data.resolve()), str(artifacts.resolve()), (artifacts / "model.pt").stat().st_mtime_ns, cutoff)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if pickup_zone is not None and pickup_zone not in meta["zone_ids"]:
        raise HTTPException(422, "Unknown pickup zone")
    matching = [p for p in predictions if p["predicted_trips"] >= min_trips and (pickup_zone is None or p["source_zone"] == pickup_zone)]
    _, last = bounds(data)
    return {"as_of": format_time(cutoff), "horizon_minutes": 60,
            "forecast_type": "historical_backtest" if cutoff <= last else "beyond_available_data",
            "data_notice": "TLC publishes monthly with a delay; this is not live traffic.",
            "predictions": matching[:top_k], "summary": summary}


@app.get("/api/actuals")
def actuals(as_of: str, pickup_zone: int | None = None):
    data, _ = data_or_503()
    cutoff = parse_cutoff(as_of)
    glob = data / "processed" / "targets" / "*.parquet"
    where = " AND src = ?" if pickup_zone is not None else ""
    params = [cutoff] + ([pickup_zone] if pickup_zone is not None else [])
    with duckdb.connect() as con:
        frame = con.execute(f"SELECT src AS source_zone, dst AS target_zone, trip_count, duration, fare, distance, bin0, bin1, bin2, bin3 FROM read_parquet({sql_string(glob)}) WHERE pickup_hour = ?{where} ORDER BY trip_count DESC", params).df()
    return {"as_of": format_time(cutoff), "available": not frame.empty, "total_trips": int(frame.trip_count.sum()), "routes": frame.to_dict(orient="records")}


@app.get("/api/trends")
def trends(as_of: str, zone_id: int | None = None, hours: int = Query(24, ge=1, le=168)):
    data, _ = data_or_503()
    cutoff = parse_cutoff(as_of)
    glob = data / "processed" / "targets" / "*.parquet"
    condition = " AND src = ?" if zone_id is not None else ""
    params = [cutoff-hours*HOUR, cutoff] + ([zone_id] if zone_id is not None else [])
    with duckdb.connect() as con:
        rows = con.execute(f"SELECT pickup_hour, sum(trip_count) FROM read_parquet({sql_string(glob)}) WHERE pickup_hour >= ? AND pickup_hour <= ?{condition} GROUP BY pickup_hour ORDER BY pickup_hour", params).fetchall()
    return {"zone_id": zone_id, "hours": [{"time": format_time(t), "actual_trips": int(count)} for t,count in rows]}


@app.get("/api/zone/{zone_id}/forecast")
def zone_forecast(zone_id: int, as_of: str, top_k: int = Query(10, ge=1, le=100)):
    data, artifacts = model_or_503()
    cutoff = parse_cutoff(as_of)
    try:
        predictions, meta, summary = cached_forecast(str(data.resolve()), str(artifacts.resolve()), (artifacts / "model.pt").stat().st_mtime_ns, cutoff)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if zone_id not in meta["zone_ids"]:
        raise HTTPException(404, "Unknown zone")
    return {"zone_id": zone_id, "as_of": format_time(cutoff),
            "outgoing": [p for p in predictions if p["source_zone"] == zone_id][:top_k],
            "incoming": [p for p in predictions if p["target_zone"] == zone_id][:top_k],
            "predicted_pickups": summary["zone_pickups"][str(zone_id)], "predicted_dropoffs": summary["zone_dropoffs"][str(zone_id)],
            "recent_actuals": trends(as_of, zone_id, 24)["hours"]}


@app.get("/api/route")
def route(as_of: str, source_zone: int, target_zone: int):
    data, artifacts = model_or_503()
    cutoff = parse_cutoff(as_of)
    try:
        predictions, meta, _ = cached_forecast(str(data.resolve()), str(artifacts.resolve()), (artifacts / "model.pt").stat().st_mtime_ns, cutoff)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(422, str(exc)) from exc
    if source_zone not in meta["zone_ids"] or target_zone not in meta["zone_ids"]:
        raise HTTPException(404, "Unknown zone")
    prediction = next(p for p in predictions if p["source_zone"] == source_zone and p["target_zone"] == target_zone)
    glob = data / "processed" / "targets" / "*.parquet"
    with duckdb.connect() as con:
        frame = con.execute(f"SELECT src AS source_zone, dst AS target_zone, trip_count, duration, fare, distance, bin0, bin1, bin2, bin3 FROM read_parquet({sql_string(glob)}) WHERE pickup_hour=? AND src=? AND dst=?", [cutoff, source_zone, target_zone]).df()
    return {"prediction": prediction, "actual": None if frame.empty else frame.iloc[0].to_dict()}


def model_profiles():
    data, artifacts = paths()
    profiles = {"current": (data, artifacts), "demo": (ROOT / "data/quick", ROOT / "artifacts"),
                "full": (ROOT / "data", ROOT / "artifacts/cuda-full")}
    for mode in ("quick", "full"):
        destination = ROOT / "artifacts" / f"dashboard-{mode}"
        profile = destination / "profile.json"
        if profile.is_file():
            try:
                recorded = json.loads(profile.read_text())
                profiles[f"trained_{mode}"] = (Path(recorded["data_dir"]), Path(recorded.get("artifacts_dir", destination)))
            except (OSError, ValueError, KeyError):
                pass
    return profiles


@app.get("/api/models")
def models():
    current = paths()
    result = []
    selected = "current"
    for key, (data, artifacts) in model_profiles().items():
        path = artifacts / "model.pt"
        item = {"id": key, "label": {"current": "Current configuration", "demo": "Quick demo", "full": "Full NYC model", "trained_quick": "Dashboard quick run", "trained_full": "Dashboard full run"}[key],
                "ready": path.is_file() and (data / "processed/zones.parquet").is_file()}
        if item["ready"]:
            meta = torch.load(path, map_location="cpu", weights_only=False)["metadata"]
            item.update(quick=meta.get("quick", False), best_epoch=meta.get("best_epoch"))
        result.append(item)
        if key != "current" and (data.resolve(), artifacts.resolve()) == tuple(p.resolve() for p in current):
            selected = key
    if selected != "current":
        result = [m for m in result if m["id"] != "current"]
    return {"current": selected, "models": result}


class ModelSelection(BaseModel):
    id: Literal["current", "demo", "full", "trained_quick", "trained_full"]


@app.post("/api/models/select")
def select_model(body: ModelSelection):
    global _selected_paths
    if any(j["status"] in ("queued", "running") for j in active_jobs()):
        raise HTTPException(409, "Wait for the active job to finish before switching models")
    if body.id not in model_profiles():
        raise HTTPException(422, "This training profile is not ready")
    selected = model_profiles()[body.id]
    if not (selected[1] / "model.pt").is_file() or not (selected[0] / "processed/zones.parquet").is_file():
        raise HTTPException(422, "This model/data profile is not ready")
    _selected_paths = selected
    cached_forecast.cache_clear()
    return {"id": body.id, "message": "Active model changed for this server process"}


class SimulationRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False, extra="forbid")
    as_of: str
    hours: int = Field(2, ge=1, le=6)
    fleet_size: int = Field(80, ge=1, le=300)
    fuel_price: float = Field(3.5, ge=0, le=20)
    mpg: float = Field(25, ge=1, le=100)
    price_multiplier: float = Field(1, ge=.1, le=5)
    demand_scale: float = Field(1, ge=.01, le=2)
    max_wait_minutes: float = Field(8, ge=1, le=30)
    driver_cost_hour: float = Field(0, ge=0, le=100)
    seed: int = Field(42, ge=0, le=2**31-1)


@app.post("/api/simulations", status_code=202)
def start_simulation(body: SimulationRequest):
    data, artifacts = model_or_503()
    cutoff = parse_cutoff(body.as_of)
    cfg = SimulationConfig(start=cutoff, **body.model_dump(exclude={"as_of"}))
    def work():
        predictions, _, _ = cached_forecast(str(data.resolve()), str(artifacts.resolve()),
                                             (artifacts / "model.pt").stat().st_mtime_ns, cutoff)
        return compare_policies(data, cfg, predictions)
    try:
        return submit("simulation", work)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    try:
        return get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, "Job not found; jobs expire or reset when the server restarts") from exc


class TrainingRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["quick", "full"] = "quick"
    epochs: int = Field(3, ge=1, le=100)
    device: Literal["auto", "cpu", "cuda"] = "auto"
    max_hours: int = Field(240, ge=48, le=744)
    resume: bool = False


@app.get("/api/training")
def training_status():
    data, artifacts = paths()
    result = {"data_dir": str(data), "artifacts_dir": str(artifacts),
              "cuda_available": torch.cuda.is_available(), "jobs": active_jobs()}
    run_artifacts = _training_paths[1] if _training_paths else artifacts
    result["training_artifacts_dir"] = str(run_artifacts)
    for name in ("progress", "training", "evaluation_progress"):
        path = run_artifacts / f"{name}.json"
        try:
            result[name] = json.loads(path.read_text()) if path.is_file() else None
        except (OSError, ValueError):
            result[name] = None
    if (artifacts / "model.pt").is_file():
        meta = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=False)["metadata"]
        result["model"] = {k: meta.get(k) for k in ("quick", "best_epoch", "start", "validation_end", "test_end", "runtime")}
    log = run_artifacts / "dashboard-training.log"
    if log.is_file():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-6000))
            result["log_tail"] = stream.read().decode(errors='replace')
    return result


@app.post("/api/training", status_code=202)
def start_training(body: TrainingRequest):
    global _training_paths
    data, _ = data_or_503()
    if body.mode == "quick" and not os.environ.get("NYC_DATA_DIR") and (ROOT / "data/quick/processed/zones.parquet").is_file():
        data = ROOT / "data/quick"
    if body.mode == "full" and not os.environ.get("NYC_DATA_DIR") and (ROOT / "data/processed/zones.parquet").is_file():
        data = ROOT / "data"
    destination = ROOT / "artifacts" / f"dashboard-{body.mode}"
    if body.resume:
        try:
            last_run = json.loads((destination / "last_run.json").read_text())
            artifacts = Path(last_run["artifacts_dir"])
            if str(data) != last_run["data_dir"]:
                raise ValueError("Resume data directory does not match the previous run")
        except (OSError, ValueError, KeyError) as exc:
            raise HTTPException(422, "No matching dashboard training run to resume") from exc
    else:
        artifacts = destination / f"run-{uuid4().hex[:12]}"
    def work():
        global _selected_paths, _training_paths
        _training_paths = (data, artifacts)
        artifacts.mkdir(parents=True, exist_ok=True)
        profile = {"data_dir": str(data), "artifacts_dir": str(artifacts), "mode": body.mode}
        (destination / "last_run.json").write_text(json.dumps(profile))
        command = [sys.executable, "-m", "mobility.training", "--data-dir", str(data),
                   "--artifacts-dir", str(artifacts), "--config", str(ROOT / "configs/full.json"),
                   "--epochs", str(body.epochs), "--device", body.device, "--log-every-hours", "24"]
        if body.mode == "quick":
            command += ["--quick", "--max-hours", str(body.max_hours), "--max-positives", "256"]
        if body.resume:
            command += ["--resume", str(artifacts / "training_state.pt")]
        with (artifacts / "dashboard-training.log").open('w') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if completed.returncode:
                raise ValueError('Training failed. Inspect the training log in the dashboard.')
            cached_forecast.cache_clear()
            evaluation = subprocess.run([sys.executable, "-m", "mobility.evaluation", "--data-dir", str(data),
                                         "--artifacts-dir", str(artifacts), "--device", body.device],
                                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if evaluation.returncode:
                raise ValueError('Training completed, but evaluation failed. Inspect the training log.')
        temporary_profile = destination / "profile.tmp"
        temporary_profile.write_text(json.dumps(profile))
        temporary_profile.replace(destination / "profile.json")
        _selected_paths = (data, artifacts)
        cached_forecast.cache_clear()
        return {"message": "Training and held-out evaluation complete. The new checkpoint is active; refresh the dashboard.",
                "artifacts_dir": str(artifacts)}
    try:
        return submit("training", work)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


# Production build can be served by the same process, with /api routes above.
frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="dashboard")
