"""Chronological TGN training with validation-based early stopping."""
from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from mobility.graph import GraphContext, HOUR, bounds, candidates, format_time, hour_groups, read_zones
from mobility.model import NORMALIZATION, TaxiTGN, multitask_loss
from mobility.download import months_between


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Install a CUDA-enabled PyTorch build and check the NVIDIA driver.")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this PyTorch environment.")
    return torch.device(name)


def runtime_info(device: torch.device) -> dict:
    info = {"device": str(device), "torch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda}
    if device.type == "cuda":
        info["gpu"] = torch.cuda.get_device_name(device)
        info["compute_capability"] = list(torch.cuda.get_device_capability(device))
    return info


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def save_checkpoint(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def epoch_seconds(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Split timestamps require an explicit UTC offset")
    result = int(parsed.timestamp())
    if result % HOUR:
        raise ValueError("Split timestamps must be on an hour boundary")
    return result


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def label_available_by_boundary(cutoff: int, boundary: int) -> bool:
    # Pickups occur throughout the hour and valid trips may last three hours.
    return cutoff + 4 * HOUR <= boundary


def walk(model, zones, data_dir, start, train_end, val_end, config, optimizer, device, on_progress=None, log_every=168):
    """Replay completed observations first, then predict each pickup hour."""
    model.train(); model.memory.detach(); model.memory.reset_state()
    ctx = GraphContext(zones, device)
    observed = iter(hour_groups(data_dir, "observations", val_end, start))
    pending = next(observed, None)
    train_losses, val_losses = [], []
    rng_train = np.random.default_rng(config["seed"])
    rng_val = np.random.default_rng(config["seed"] + 1)
    validating = False
    processed = 0
    for cutoff, rows in hour_groups(data_dir, "targets", val_end - 1, start):
        if cutoff < start:
            continue
        if cutoff >= train_end and not validating:
            # PyG flushes pending messages when switching to evaluation mode;
            # do not retain a graph from that flush into the next epoch.
            with torch.no_grad():
                model.eval()
            validating = True
        while pending is not None and pending[0] <= cutoff:
            ctx.observe(model, pending[0], pending[1], cutoff)
            pending = next(observed, None)
        is_train = cutoff < train_end
        # Pickups can finish up to three hours later. Exclude the final three
        # pickup hours of each fit/selection split so all labels are known by
        # the boundary. Continue replaying observations through the gap.
        boundary = train_end if is_train else val_end
        if not label_available_by_boundary(cutoff, boundary):
            continue
        selected, src, dst = candidates(rows, ctx, rng_train if is_train else rng_val,
                                         config["max_positives"], config["negative_ratio"])
        if not len(selected):
            continue
        with torch.set_grad_enabled(is_train):
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            z = model.encode(ctx, cutoff)
            output = model.score(z, ctx, cutoff, src, dst)
            loss = multitask_loss(output, selected, config.get("loss_weights"), len(rows), ctx.n * ctx.n)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                model.memory.detach()
        (train_losses if is_train else val_losses).append(float(loss.detach().cpu()))
        processed += 1
        if on_progress and log_every and processed % log_every == 0:
            on_progress("training" if is_train else "validation", cutoff, processed)
    if not train_losses or not val_losses:
        raise ValueError("The selected period must contain both training and validation hours")
    return float(np.mean(train_losses)), float(np.mean(val_losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--config", type=Path, default=Path("configs/full.json"))
    parser.add_argument("--quick", action="store_true", help="Use chronological 70/15/15 splits of available data")
    parser.add_argument("--max-hours", type=int, help="Limit quick mode to the first N hours for a smoke run")
    parser.add_argument("--max-positives", type=int, help="Maximum positive routes per training hour")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every-hours", type=int, default=168, help="Report progress every N processed hours; 0 disables periodic reports")
    parser.add_argument("--resume", type=Path, help="Resume at the next epoch from training_state.pt")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1:
        raise SystemExit("epochs and patience must be positive")
    if args.log_every_hours < 0:
        raise SystemExit("--log-every-hours must be nonnegative")
    if args.max_hours is not None and (not args.quick or args.max_hours < 10):
        raise SystemExit("--max-hours requires --quick and must be at least 10")
    config = json.loads(args.config.read_text())
    if args.max_positives:
        config["max_positives"] = args.max_positives
    start, last = bounds(args.data_dir)
    if args.quick:
        end = min(last + HOUR, start + args.max_hours*HOUR) if args.max_hours else last + HOUR
        hours = (end-start)//HOUR
        train_end = start + int(hours*0.70)*HOUR
        val_end = start + int(hours*0.85)*HOUR
    else:
        train_end = epoch_seconds(config["train_end"])
        val_end = epoch_seconds(config["validation_end"])
        end = epoch_seconds(config["test_end"])
        required = months_between("2023-10", "2026-09")
        missing = [m for m in required if not (args.data_dir / "targets" / f"{m}.parquet").is_file()
                   or not (args.data_dir / "observations" / f"{m}.parquet").is_file()]
        if missing or start > epoch_seconds("2023-10-01T00:00:00-04:00") or last < end-24*HOUR:
            raise SystemExit("The full split requires the complete October 2023–September 2026 dataset. Use --quick for partial data.")
    if not start < train_end < val_end < end:
        raise SystemExit("Invalid chronological split")
    torch.set_num_threads(max(1, args.threads))
    try:
        device = device_for(args.device)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    seed_all(config["seed"])
    zones = read_zones(args.data_dir)
    model = TaxiTGN(zones, config["memory_dim"], config["time_dim"], config["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    best, bad, log, best_epoch, first_epoch = float("inf"), 0, [], None, 1
    metadata = {"config": config, "normalization": NORMALIZATION, "start": start, "train_end": train_end, "validation_end": val_end, "test_end": end,
                "zone_ids": zones.zone_id.astype(int).tolist(), "quick": args.quick, "runtime": runtime_info(device)}
    if args.resume:
        if args.resume.resolve().parent != args.artifacts_dir.resolve():
            raise SystemExit("Resume with the same --artifacts-dir that contains training_state.pt")
        state = torch.load(args.resume, map_location=device, weights_only=False)
        for key in ("config", "normalization", "start", "train_end", "validation_end", "test_end", "zone_ids", "quick"):
            if state["metadata"].get(key) != metadata[key]:
                raise SystemExit(f"Cannot resume: checkpoint {key} does not match the current dataset or configuration")
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer_state"])
        # Checkpoints are written after validation. Mirror that module mode so
        # the next epoch's train transition matches an uninterrupted run.
        with torch.no_grad():
            model.eval()
        model.memory.reset_state()
        best, bad, log, best_epoch = state["best"], state["bad"], state["epochs"], state["best_epoch"]
        first_epoch = state["epoch"] + 1
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and state.get("cuda_rng"):
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        if first_epoch > args.epochs or bad >= args.patience:
            raise SystemExit("The checkpoint has already reached the requested epoch count or early-stopping limit")
    started = time.monotonic()
    print(json.dumps({"event": "start", "runtime": metadata["runtime"], "first_epoch": first_epoch, "epochs": args.epochs}), flush=True)

    def report(status, epoch, cutoff=None, hours=None):
        progress = {"status": status, "epoch": epoch, "cutoff": format_time(cutoff) if cutoff is not None else None,
                    "processed_hours": hours, "elapsed_seconds": round(time.monotonic()-started, 1), "runtime": metadata["runtime"]}
        write_json(args.artifacts_dir / "progress.json", progress)
        print(json.dumps({"event": "progress", **progress}), flush=True)

    epoch = first_epoch
    report("starting", epoch)
    try:
        for epoch in range(first_epoch, args.epochs + 1):
            train_loss, val_loss = walk(model, zones, args.data_dir, start, train_end, val_end, config, optimizer, device,
                                        on_progress=lambda phase, cutoff, hours: report(phase, epoch, cutoff, hours), log_every=args.log_every_hours)
            record = {"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss}
            log.append(record)
            print(json.dumps(record), flush=True)
            improved = val_loss < best
            if improved:
                best, bad, best_epoch = val_loss, 0, epoch
            else:
                bad += 1
            model.memory.detach()
            model.memory.reset_state()  # Replay observations when loading; never save validation memory.
            if improved:
                save_checkpoint(args.artifacts_dir / "model.pt", {"state_dict": model.state_dict(), "metadata": {**metadata, "best_epoch": epoch}})
            save_checkpoint(args.artifacts_dir / "training_state.pt", {"state_dict": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                            "metadata": metadata, "epoch": epoch, "best": best, "bad": bad, "best_epoch": best_epoch, "epochs": log,
                            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                            "python_rng": random.getstate(), "numpy_rng": np.random.get_state()})
            write_json(args.artifacts_dir / "training.json", {"metadata": metadata, "epochs": log})
            if bad >= args.patience:
                break
    except BaseException:
        report("interrupted", epoch)
        raise
    report("complete", epoch)
    print(f"Saved {args.artifacts_dir / 'model.pt'}; held-out test starts {format_time(val_end)}", flush=True)


if __name__ == "__main__":
    main()
