"""Independently check input, split, CUDA training, and held-out ranking evidence."""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np


def read(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sql(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


def epoch(value: str) -> int:
    return int(datetime.fromisoformat(value).timestamp())


class Verifier:
    def __init__(self, context: dict):
        root = Path(context["tests_dir"]).resolve().parent
        workdir = Path(os.environ.get("OTTER_APP_DIR", context["workdir"]))
        default_data = Path("/app/data") if Path("/app/data").is_dir() else root / "environment/task_inputs/data"
        default_manifest = Path("/app/dataset_manifest.json") if Path("/app/dataset_manifest.json").is_file() else root / "environment/task_inputs/dataset_manifest.json"
        self.data = Path(os.environ.get("NYC_DATA_DIR", default_data))
        self.data_manifest = Path(os.environ.get("NYC_DATA_MANIFEST", default_manifest))
        self.artifacts = Path(os.environ.get("NYC_ARTIFACTS_DIR", workdir / "artifacts"))
        self.reports = Path(os.environ.get("NYC_REPORTS_DIR", workdir / "reports"))
        self._replayed = None

    def check_data(self):
        manifest = read(self.data_manifest)
        expected = manifest.get("files", {})
        require(manifest.get("schema_version") == 1 and expected, "Missing or invalid input inventory")
        actual = {p.relative_to(self.data).as_posix() for p in self.data.rglob("*") if p.is_file()}
        require(actual == set(expected), "Input file set differs from the inventory")
        for relative, info in expected.items():
            path = self.data / relative
            require(not path.is_symlink() and path.stat().st_size == info["bytes"] and sha256(path) == info["sha256"], f"Input changed: {relative}")
        months = {f"{year:04d}-{month:02d}" for year in range(2023, 2027) for month in range(1, 13)
                  if "2023-10" <= f"{year:04d}-{month:02d}" <= "2026-09"}
        for kind in ("targets", "observations"):
            required = {f"{kind}/{month}.parquet" for month in months}
            require(required <= set(expected), f"Expected all 36 monthly TLC {kind} files")
        require({"zones.parquet", "zones.geojson", "validation.json"} <= set(expected), "Missing zone assets or preprocessing validation")
        return f"Verified {len(expected)} input files, including 36 months of targets and observations"

    def check_splits(self):
        import duckdb
        s = read(self.artifacts / "splits.json")
        training = read(self.artifacts / "training.json")
        b, splits, metadata = s["boundaries"], s["splits"], training["metadata"]
        expected = {"train_end": epoch("2025-10-01T00:00:00-04:00"),
                    "validation_end": epoch("2026-04-01T00:00:00-04:00"),
                    "test_end": epoch("2026-10-01T00:00:00-04:00")}
        require(not s["quick"] and s["max_hours"] is None and not metadata["quick"], "Quick/smoke runs cannot pass the full task")
        require(all(b[k] == v and metadata[k] == v for k, v in expected.items()), "Incorrect chronological split boundaries")
        require(b["start"] == metadata["start"] and b["start"] < b["train_end"] < b["validation_end"] < b["test_end"], "Overlapping or unordered boundaries")
        require(s["data_manifest_sha256"] == sha256(self.data_manifest) and s["label_embargo_hours"] == 3, "Input binding or label embargo is missing")
        ranges = {"train": (b["start"], b["train_end"] - 3 * 3600),
                  "validation": (b["train_end"], b["validation_end"] - 3 * 3600),
                  "test": (b["validation_end"], b["test_end"])}
        with duckdb.connect() as con:
            for name, (start, end) in ranges.items():
                item = splits[name]
                require(item["start"] == start and item["end_exclusive"] == end and start < end, f"Invalid {name} period")
                row = con.execute(f"SELECT count(*), coalesce(sum(trip_count), 0), count(DISTINCT pickup_hour), min(pickup_hour), max(pickup_hour) FROM read_parquet({sql(self.data / 'targets/*.parquet')}) WHERE pickup_hour >= ? AND pickup_hour < ?", [start, end]).fetchone()
                require(tuple(row) == (item["route_rows"], item["trips"], item["hours_with_trips"], item["first_hour"], item["last_hour"]) and row[0] > 0, f"Incorrect {name} counts")
        return "Full chronological train/validation/test periods and three-hour label embargo verified"

    def check_cuda(self):
        training = read(self.artifacts / "training.json")
        runtime = training["metadata"]["runtime"]
        require(runtime.get("device", "").startswith("cuda") and runtime.get("cuda_runtime") and runtime.get("gpu"), "Training did not record a CUDA GPU")
        require(len(runtime.get("compute_capability", [])) == 2, "CUDA compute capability missing")
        return f"Training recorded {runtime['gpu']} with CUDA {runtime['cuda_runtime']}"

    def check_training(self):
        import torch
        training = read(self.artifacts / "training.json")
        logs = training["epochs"]
        require(logs and [x["epoch"] for x in logs] == list(range(1, len(logs) + 1)), "No sequential training epochs")
        require(all(math.isfinite(x["train_loss"]) and math.isfinite(x["validation_loss"]) for x in logs), "Missing or nonfinite train/validation loss")
        checkpoint = torch.load(self.artifacts / "model.pt", map_location="cpu", weights_only=True)
        require((self.artifacts / "training_state.pt").is_file(), "Missing resumable training state")
        best = min(logs, key=lambda x: x["validation_loss"])["epoch"]
        meta = checkpoint["metadata"]
        require(meta["best_epoch"] == best, "Checkpoint is not the best validation epoch")
        require(all(meta[k] == training["metadata"][k] for k in ("start", "train_end", "validation_end", "test_end", "config", "zone_ids")), "Checkpoint metadata disagrees with training")
        weights = checkpoint["state_dict"]
        require(any(k.startswith("memory.gru.") for k in weights) and any(k.startswith("attention.") for k in weights), "Checkpoint lacks TGN memory or temporal attention")
        require(all(torch.isfinite(v).all().item() for v in weights.values() if torch.is_floating_point(v)), "Nonfinite model weights")
        return f"{len(logs)} trained and validated epochs; saved best epoch {best}"

    def check_coverage(self):
        split = read(self.artifacts / "splits.json")
        metrics = read(self.artifacts / "metrics.json")
        population = metrics["population"]
        require(population.get("max_hours") is None, "Partial evaluation cannot pass the full task")
        require(population["hours"] == split["splits"]["test"]["hours_with_trips"] and population["hours"] > 0, "Held-out coverage is incomplete")
        require(epoch(population["test_start"]) == split["boundaries"]["validation_end"] and epoch(population["test_end_exclusive"]) == split["boundaries"]["test_end"], "Evaluation period is not the held-out test split")
        require(population["negatives_per_positive"] == 20 and population["max_positives_per_hour"] == 256, "Benchmark sampling protocol changed")
        require(metrics["provenance"]["model_sha256"] == sha256(self.artifacts / "model.pt"), "Evaluation does not match saved checkpoint")
        require(metrics["runtime"]["device"].startswith("cuda"), "Evaluation did not run on CUDA")
        return f"Complete held-out interval: {population['hours']} hourly groups"

    def replay_mrr(self):
        if self._replayed is not None:
            return self._replayed
        import pyarrow.parquet as pq
        metrics = read(self.artifacts / "metrics.json")
        names = ("tgn", "recent", "seasonal")
        readers = {n: pq.ParquetFile(self.artifacts / f"evaluation_{n}.parquet") for n in names}
        groups = readers["tgn"].num_row_groups
        require(groups == metrics["population"]["hours"] and all(p.num_row_groups == groups for p in readers.values()), "Missing or misaligned prediction groups")
        compare = ["cutoff", "group_id", "source_zone", "target_zone", "actual_trips"]
        columns = compare + ["link_probability"]
        totals = {n: 0.0 for n in names}
        positives = examples = 0
        first_cutoff = epoch(metrics["population"]["test_start"])
        end_cutoff = epoch(metrics["population"]["test_end_exclusive"])
        previous_cutoff = first_cutoff - 1
        requested_negatives = metrics["population"]["negatives_per_positive"]
        for index in range(groups):
            batches = {n: readers[n].read_row_group(index, columns=columns).to_pandas() for n in names}
            base = batches["tgn"]
            require(len(base) > 0 and base.cutoff.nunique() == 1, "Each prediction group must represent one hour")
            cutoff = int(base.cutoff.iloc[0])
            require(first_cutoff <= cutoff < end_cutoff and cutoff > previous_cutoff, "Repeated, unordered, or out-of-split evaluation hour")
            previous_cutoff = cutoff
            for n in names[1:]:
                require(base[compare].equals(batches[n][compare]), f"{n} used different evaluation candidates")
            actual = base.actual_trips.to_numpy()
            ids = base.group_id.to_numpy(dtype=np.int64)
            source = base.source_zone.to_numpy()
            positive = np.flatnonzero(actual > 0)
            negative = np.flatnonzero(actual == 0)
            count = len(positive)
            require(count > 0 and count <= 256 and len(positive) + len(negative) == len(base), "Invalid positive or negative observations")
            require(np.array_equal(np.sort(ids[positive]), np.arange(count)) and ids.min() >= 0 and ids.max() < count, "Invalid candidate group identifiers")
            require(not base.iloc[positive][["source_zone", "target_zone"]].duplicated().any(), "Duplicate positive route within an hour")
            pos_source = np.empty(count, dtype=source.dtype)
            pos_source[ids[positive]] = source[positive]
            require(np.array_equal(source[negative], pos_source[ids[negative]]), "A negative candidate has the wrong pickup zone")
            per_group = np.bincount(ids[negative], minlength=count)
            # In this bundled full test set every origin/hour has at least 20
            # inactive destination zones, so each group must contain all 20.
            require(np.all(per_group == requested_negatives), "Incorrect number of sampled negatives")
            pairs = base.iloc[negative][["group_id", "target_zone"]]
            require(not pairs.duplicated().any(), "Duplicate negative within a candidate group")
            for name in names:
                score = batches[name].link_probability.to_numpy(dtype=float)
                require(np.all(np.isfinite(score)) and np.all((score >= 0) & (score <= 1)), f"Invalid {name} scores")
                pos_score = np.empty(count)
                pos_score[ids[positive]] = score[positive]
                larger = (score[negative] > pos_score[ids[negative]]).astype(float)
                ties = (score[negative] == pos_score[ids[negative]]).astype(float)
                rank = 1 + np.bincount(ids[negative], weights=larger + 0.5 * ties, minlength=count)
                totals[name] += float(np.reciprocal(rank).sum())
            positives += count
            examples += len(base)
        require(positives > 0, "No test candidates")
        values = {n: totals[n] / positives for n in names}
        for n in names:
            recorded = metrics["models"][n]
            require(recorded["positive_routes"] == positives and recorded["examples"] == examples, f"Incorrect {n} evaluation population")
            require(math.isclose(recorded["mrr"], values[n], abs_tol=1e-10), f"Recorded {n} MRR does not match predictions")
        self._replayed = values
        return values

    def check_candidates(self):
        v = self.replay_mrr()
        return f"MRR recomputed from matched candidates: TGN {v['tgn']:.6f}, recent {v['recent']:.6f}, seasonal {v['seasonal']:.6f}"

    def check_ground_truth(self):
        import duckdb
        prediction = sql(self.artifacts / "evaluation_tgn.parquet")
        target = sql(self.data / "targets/*.parquet")
        with duckdb.connect() as con:
            total, wrong = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE e.actual_trips != coalesce(t.trip_count, 0))
                FROM read_parquet({prediction}) e LEFT JOIN read_parquet({target}) t
                ON e.cutoff=t.pickup_hour AND e.source_zone=t.src AND e.target_zone=t.dst""").fetchone()
        expected = read(self.artifacts / "metrics.json")["models"]["tgn"]["examples"]
        require(total == expected and wrong == 0, f"Prediction labels do not match input data ({wrong} mismatches)")
        return f"Verified {total} sampled labels against the TLC target data"

    def check_report(self):
        v = self.replay_mrr()
        report = (self.reports / "report.md").read_text()
        assessment = read(self.artifacts / "assessment.json")
        require(len(report) > 150 and all(x in report.lower() for x in ("validation", "seasonal", "mrr")), "Report is missing required topics")
        require(all(math.isclose(assessment[k], v[k], abs_tol=1e-10) for k in v), "Assessment scores disagree with predictions")
        diff = v["tgn"] - max(v["recent"], v["seasonal"])
        require(math.isclose(assessment["improvement_over_best_history"], diff, abs_tol=1e-10), "Incorrect improvement")
        require(assessment["improved_over_both"] == (diff > 0), "Incorrect pass/fail assessment")
        require(assessment["model_sha256"] == sha256(self.artifacts / "model.pt") and assessment["metrics_sha256"] == sha256(self.artifacts / "metrics.json"), "Assessment provenance mismatch")
        return "Report and assessment agree with independently recomputed scores"


def evaluate(context: dict) -> dict:
    verifier = Verifier(context)
    checks = [
        ("data_integrity", "data-preparation", verifier.check_data),
        ("chronological_splits", "data-preparation", verifier.check_splits),
        ("cuda_training", "model-training", verifier.check_cuda),
        ("training_validation_checkpoint", "model-training", verifier.check_training),
        ("evaluation_coverage", "held-out-evaluation", verifier.check_coverage),
        ("matched_candidates_and_mrr", "held-out-evaluation", verifier.check_candidates),
        ("ground_truth", "held-out-evaluation", verifier.check_ground_truth),
        ("report", "held-out-evaluation", verifier.check_report),
    ]
    results = []
    for cid, milestone, check in checks:
        try:
            detail, passed = check(), True
        except Exception as exc:
            detail, passed = f"{type(exc).__name__}: {exc}", False
        results.append({"id": cid, "milestone_id": milestone, "passed": passed, "detail": detail})
    # Missing or invalid evidence is a failing metric, never an assumed success.
    try:
        values = verifier.replay_mrr()
        improvement = values["tgn"] - max(values["recent"], values["seasonal"])
    except Exception:
        improvement = -1.0
    return {"criteria": results, "metrics": {"mrr_improvement_over_best_history": improvement}}
