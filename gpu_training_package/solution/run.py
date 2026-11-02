#!/usr/bin/env python3
"""Prepare chronological splits, train/validate the TGN, then evaluate it."""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mobility.graph import HOUR, bounds, format_time  # noqa: E402
from mobility.preprocess import sql_string  # noqa: E402
from mobility.provenance import sha256_file  # noqa: E402
from mobility.training import epoch_seconds, write_json  # noqa: E402


def defaults() -> tuple[Path, Path, Path, Path]:
    bundled = HERE.parent / "environment" / "task_inputs"
    data = Path(os.environ.get("NYC_DATA_DIR", "/app/data" if Path("/app/data").is_dir() else bundled / "data"))
    manifest = Path(os.environ.get("NYC_DATA_MANIFEST", "/app/dataset_manifest.json" if Path("/app/dataset_manifest.json").is_file() else bundled / "dataset_manifest.json"))
    artifacts = Path(os.environ.get("NYC_ARTIFACTS_DIR", "/app/artifacts" if Path("/app").is_dir() else HERE.parent / "artifacts"))
    reports = Path(os.environ.get("NYC_REPORTS_DIR", "/app/reports" if Path("/app").is_dir() else HERE.parent / "reports"))
    return data, manifest, artifacts, reports


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=HERE, check=True)


def prepare(args: argparse.Namespace) -> dict:
    verifier = HERE.parent / "environment" / "verify_data.py"
    if not verifier.is_file():
        verifier = Path("/opt/nyc-taxi/verify_data.py")
    run([sys.executable, str(verifier), "check", "--data", str(args.data_dir), "--manifest", str(args.data_manifest)])
    config = json.loads(args.config.read_text())
    start, last = bounds(args.data_dir)
    if args.quick:
        end = min(last + HOUR, start + args.max_hours * HOUR) if args.max_hours else last + HOUR
        hours = (end - start) // HOUR
        train_end, val_end = start + int(hours * .70) * HOUR, start + int(hours * .85) * HOUR
    else:
        train_end, val_end, end = (epoch_seconds(config[k]) for k in ("train_end", "validation_end", "test_end"))
    if not (start < train_end - 3 * HOUR and train_end < val_end - 3 * HOUR and val_end < end <= last + HOUR):
        raise ValueError("Dataset does not support nonempty chronological train, validation, and test periods")
    ranges = {"train": (start, train_end - 3 * HOUR),
              "validation": (train_end, val_end - 3 * HOUR),
              "test": (val_end, end)}
    table = sql_string(args.data_dir / "targets" / "*.parquet")
    splits = {}
    with duckdb.connect() as connection:
        for name, (lo, hi) in ranges.items():
            n, trips, hours, first, final = connection.execute(
                f"SELECT count(*), coalesce(sum(trip_count),0), count(DISTINCT pickup_hour), min(pickup_hour), max(pickup_hour) FROM read_parquet({table}) WHERE pickup_hour >= ? AND pickup_hour < ?",
                [lo, hi],
            ).fetchone()
            if not n or not hours:
                raise ValueError(f"Empty {name} split")
            splits[name] = {"start": lo, "end_exclusive": hi, "start_local": format_time(lo),
                            "end_local_exclusive": format_time(hi), "route_rows": int(n),
                            "trips": int(trips), "hours_with_trips": int(hours),
                            "first_hour": int(first), "last_hour": int(final)}
    result = {"schema_version": 1, "quick": args.quick, "max_hours": args.max_hours,
              "data_manifest_sha256": sha256_file(args.data_manifest), "label_embargo_hours": 3,
              "boundaries": {"start": start, "train_end": train_end, "validation_end": val_end, "test_end": end},
              "splits": splits}
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.artifacts_dir / "splits.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def make_report(args: argparse.Namespace, split: dict) -> None:
    metrics = json.loads((args.artifacts_dir / "metrics.json").read_text())
    model = metrics["models"]["tgn"]["mrr"]
    recent = metrics["models"]["recent"]["mrr"]
    seasonal = metrics["models"]["seasonal"]["mrr"]
    if not all(math.isfinite(v) for v in (model, recent, seasonal)):
        raise ValueError("Evaluation produced a nonfinite ranking score")
    assessment = {"metric": "sampled_route_mrr", "tgn": model, "recent": recent,
                  "seasonal": seasonal, "improvement_over_best_history": model - max(recent, seasonal),
                  "improved_over_both": model > max(recent, seasonal),
                  "model_sha256": sha256_file(args.artifacts_dir / "model.pt"),
                  "metrics_sha256": sha256_file(args.artifacts_dir / "metrics.json"),
                  "data_manifest_sha256": split["data_manifest_sha256"],
                  "test_hours": metrics["population"]["hours"],
                  "test_start": metrics["population"]["test_start"],
                  "test_end_exclusive": metrics["population"]["test_end_exclusive"]}
    write_json(args.artifacts_dir / "assessment.json", assessment)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "report.md").write_text(
        "# NYC taxi TGN training and evaluation\n\n"
        f"Training, validation and test periods are recorded in `splits.json`; the label embargo is {split['label_embargo_hours']} hours. "
        "The best validation-loss checkpoint was evaluated on the later test period.\n\n"
        f"| Held-out sampled route MRR | Value |\n|---|---:|\n| TGN | {model:.6f} |\n"
        f"| Recent history | {recent:.6f} |\n| Seasonal history | {seasonal:.6f} |\n\n"
        f"Improvement over the better history baseline: {assessment['improvement_over_best_history']:.6f}. "
        f"Evaluated hours: {assessment['test_hours']}.\n\n"
        "MRR uses one observed-positive route and up to 20 sampled inactive same-origin destinations. "
        "It is not citywide demand accuracy, a driver outcome, or a profit estimate. "
        "Fare and trip-count errors are reported separately in `metrics.json`.\n"
    )
    print(json.dumps(assessment, indent=2), flush=True)


def main() -> None:
    data, manifest, artifacts, reports = defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["all", "prepare", "train", "evaluate"], default="all")
    parser.add_argument("--data-dir", type=Path, default=data)
    parser.add_argument("--data-manifest", type=Path, default=manifest)
    parser.add_argument("--artifacts-dir", type=Path, default=artifacts)
    parser.add_argument("--reports-dir", type=Path, default=reports)
    parser.add_argument("--config", type=Path, default=HERE / "configs" / "full.json")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every-hours", type=int, default=720)
    parser.add_argument("--resume", action="store_true", help="Continue from artifacts/training_state.pt")
    parser.add_argument("--quick", action="store_true", help="Use a small chronological smoke-test split")
    parser.add_argument("--max-hours", type=int, help="Limit quick-mode input hours")
    parser.add_argument("--max-positives", type=int, help="Limit training positives per hour")
    parser.add_argument("--eval-max-hours", type=int, help="Limit test hours for a smoke test; never accepted by the full verifier")
    args = parser.parse_args()
    if args.max_hours is not None and (not args.quick or args.max_hours < 40):
        parser.error("--max-hours requires --quick and must be at least 40")
    if args.eval_max_hours is not None and args.eval_max_hours < 1:
        parser.error("--eval-max-hours must be positive")
    split = prepare(args)
    if args.phase in {"all", "train"}:
        command = [sys.executable, "-m", "mobility.training", "--data-dir", str(args.data_dir),
                   "--artifacts-dir", str(args.artifacts_dir), "--config", str(args.config),
                   "--device", args.device, "--epochs", str(args.epochs), "--patience", str(args.patience),
                   "--threads", str(args.threads), "--log-every-hours", str(args.log_every_hours)]
        if args.resume:
            command += ["--resume", str(args.artifacts_dir / "training_state.pt")]
        if args.quick:
            command += ["--quick"]
        if args.max_hours is not None:
            command += ["--max-hours", str(args.max_hours)]
        if args.max_positives is not None:
            command += ["--max-positives", str(args.max_positives)]
        run(command)
    if args.phase in {"all", "evaluate"}:
        command = [sys.executable, "-m", "mobility.evaluation", "--data-dir", str(args.data_dir),
                   "--artifacts-dir", str(args.artifacts_dir), "--device", args.device,
                   "--threads", str(args.threads), "--log-every-hours", str(args.log_every_hours)]
        if args.eval_max_hours is not None:
            command += ["--max-hours", str(args.eval_max_hours)]
        run(command)
        make_report(args, split)


if __name__ == "__main__":
    main()
