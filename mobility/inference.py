"""Restore a checkpoint and replay only interactions observable at the cutoff."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
from threading import RLock

_sessions = {}
_session_lock = RLock()

import numpy as np
import torch

from mobility.graph import GraphContext, hour_groups, read_zones
from mobility.model import TaxiTGN
from mobility.training import device_for


def load_model(data_dir: Path, artifacts_dir: Path, device_name="cpu"):
    device = device_for(device_name)
    checkpoint = torch.load(artifacts_dir / "model.pt", map_location=device, weights_only=False)
    zones = read_zones(data_dir)
    meta = checkpoint["metadata"]
    if zones.zone_id.astype(int).tolist() != meta["zone_ids"]:
        raise ValueError("Checkpoint zones do not match processed geometry")
    cfg = meta["config"]
    model = TaxiTGN(zones, cfg["memory_dim"], cfg["time_dim"], cfg["hidden_dim"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval(); model.memory.reset_state()
    return model, zones, meta, device


def replay(model, zones, data_dir: Path, cutoff: int, device, start: int | None = None):
    model.eval(); model.memory.reset_state()
    ctx = GraphContext(zones, device)
    with torch.no_grad():
        for available_at, rows in hour_groups(data_dir, "observations", cutoff, start):
            ctx.observe(model, available_at, rows, cutoff)
    return ctx


def predict_pairs(model, ctx, cutoff, src, dst, batch_size=8192):
    collected = {"link_probability": [], "predicted_trips": [], "predicted_duration_min": [], "predicted_fare_usd": [], "departure_bin_probabilities": []}
    with torch.no_grad():
        z = model.encode(ctx, cutoff)
        for begin in range(0, len(src), batch_size):
            out = model.score(z, ctx, cutoff, src[begin:begin+batch_size], dst[begin:begin+batch_size])
            collected["link_probability"].extend(torch.sigmoid(out["link_logits"]).cpu().tolist())
            collected["predicted_trips"].extend(out["volume"].cpu().tolist())
            collected["predicted_duration_min"].extend(out["duration"].cpu().tolist())
            collected["predicted_fare_usd"].extend(out["fare"].cpu().tolist())
            collected["departure_bin_probabilities"].extend(torch.softmax(out["departure_logits"], dim=-1).cpu().tolist())
    return collected


def _forecast(data_dir: Path, artifacts_dir: Path, cutoff: int, source_zone=None, top_k=100, min_trips=0, device="cpu"):
    if cutoff % 3600:
        raise ValueError("Cutoff must be aligned to a whole hour")
    model, zones, meta, torch_device, ctx = replay_session(data_dir, artifacts_dir, cutoff, device)
    if cutoff < meta["validation_end"]:
        raise ValueError("Cutoff precedes the end of model selection and would leak future information")
    if cutoff % 3600:
        raise ValueError("Cutoff must be aligned to a whole hour")
    if source_zone is not None and source_zone not in ctx.index:
        raise ValueError("Unknown pickup zone")
    sources = [ctx.index[source_zone]] if source_zone is not None else np.arange(ctx.n)
    src = np.repeat(sources, ctx.n); dst = np.tile(np.arange(ctx.n), len(sources))
    pred = predict_pairs(model, ctx, cutoff, src, dst)
    result = []
    for i in range(len(src)):
        if pred["predicted_trips"][i] < min_trips:
            continue
        result.append({"source_zone": ctx.ids[int(src[i])], "target_zone": ctx.ids[int(dst[i])], **{k:v[i] for k,v in pred.items()}})
    result.sort(key=lambda x: x["predicted_trips"], reverse=True)
    volumes = np.asarray(pred["predicted_trips"])
    summary = {"predicted_total": float(volumes.sum()),
               "zone_pickups": {str(ctx.ids[i]): float(x) for i, x in enumerate(np.bincount(src, weights=volumes, minlength=ctx.n))},
               "zone_dropoffs": {str(ctx.ids[i]): float(x) for i, x in enumerate(np.bincount(dst, weights=volumes, minlength=ctx.n))}}
    return result[:top_k], meta, summary


def replay_session(data_dir, artifacts_dir, cutoff, device):
    """Reuse exact temporal state, invalidating when data/checkpoint changes.

    The trusted local snapshot contains only state from its recorded cutoff.
    Earlier requests rebuild from the start; later requests replay only newly
    available observations. No temporal state is carried backwards in time.
    """
    files = [artifacts_dir / "model.pt", data_dir / "processed" / "zones.parquet",
             *sorted((data_dir / "processed" / "observations").glob("*.parquet"))]
    signature = [(str(p.resolve()), p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    identity = hashlib.sha256(json.dumps([str(torch.__version__), device, signature]).encode()).hexdigest()
    session = _sessions.get(identity)
    if session and session[4] <= cutoff:
        model, zones, meta, ctx, previous = session
    else:
        model, zones, meta, torch_device = load_model(data_dir, artifacts_dir, device)
        if cutoff < meta["validation_end"]:
            raise ValueError("Cutoff precedes the end of model selection and would leak future information")
        if cutoff % 3600:
            raise ValueError("Cutoff must be aligned to a whole hour")
        previous, ctx = None, None
        snapshot = artifacts_dir / f"replay-{device}.pt"
        if snapshot.is_file():
            try:
                state = torch.load(snapshot, map_location=torch_device, weights_only=False)
                if state.get("identity") == identity and state["cutoff"] <= cutoff:
                    model.memory.load_state_dict(state["memory"])
                    model.memory.msg_s_store = state["msg_s_store"]
                    model.memory.msg_d_store = state["msg_d_store"]
                    ctx, previous = state["ctx"], state["cutoff"]
            except (OSError, RuntimeError, ValueError, KeyError, EOFError):
                pass  # A partial/old derived cache is safe to rebuild.
        if ctx is None:
            ctx = GraphContext(zones, torch_device)
    with torch.no_grad():
        for available_at, rows in hour_groups(data_dir, "observations", cutoff, meta["start"], after=previous):
            ctx.observe(model, available_at, rows, cutoff)
    _sessions.clear()  # Bound retained state to one active model/cutoff.
    _sessions[identity] = (model, zones, meta, ctx, cutoff)
    # Reuse the first held-out state after process restarts. Additional dates
    # stay in memory, preserving the useful earliest snapshot on disk.
    snapshot = artifacts_dir / f"replay-{device}.pt"
    if previous is None or not snapshot.is_file():
        temporary = snapshot.with_suffix(".tmp")
        try:
            torch.save({"identity": identity, "cutoff": cutoff, "ctx": ctx,
                        "memory": model.memory.state_dict(), "msg_s_store": model.memory.msg_s_store,
                        "msg_d_store": model.memory.msg_d_store}, temporary)
            temporary.replace(snapshot)
        except OSError:
            pass  # Read-only artifact storage can still serve predictions.
    return model, zones, meta, ctx.device, ctx


def forecast(data_dir: Path, artifacts_dir: Path, cutoff: int, source_zone=None, top_k=100,
             min_trips=0, device="cpu"):
    with _session_lock:
        return _forecast(data_dir, artifacts_dir, cutoff, source_zone, top_k, min_trips, device)
