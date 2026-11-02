"""Regression tests for the benchmark gate and independent ranking verifier."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from checks import Verifier  # noqa: E402
from pass_fail import evaluate_primary_metrics  # noqa: E402


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setenv("OTTER_APP_DIR", str(tmp_path))
    monkeypatch.setenv("NYC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("NYC_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    data = {"cutoff": [3600] * 6, "group_id": [0, 1, 0, 0, 1, 1],
            "source_zone": [1, 2, 1, 1, 2, 2], "target_zone": [2, 4, 3, 4, 5, 6],
            "actual_trips": [2, 1, 0, 0, 0, 0]}
    scores = {"tgn": [0.9, 0.7, 0.2, 0.1, 0.2, 0.1],
              "recent": [0.1, 0.3, 0.2, 0.3, 0.2, 0.6],
              "seasonal": [0.5, 0.3, 0.5, 0.1, 0.2, 0.4]}
    values = {"tgn": 1.0, "recent": (1 / 3 + 1 / 2) / 2,
              "seasonal": (1 / 1.5 + 1 / 2) / 2}
    for name, values_for_name in scores.items():
        pd.DataFrame({**data, "link_probability": values_for_name}).to_parquet(artifacts / f"evaluation_{name}.parquet", index=False)
    metrics = {"population": {"hours": 1, "negatives_per_positive": 2,
                              "test_start": "1970-01-01T01:00:00+00:00",
                              "test_end_exclusive": "1970-01-01T02:00:00+00:00"},
               "models": {name: {"mrr": value, "positive_routes": 2, "examples": 6} for name, value in values.items()}}
    (artifacts / "metrics.json").write_text(json.dumps(metrics))
    targets = tmp_path / "data/targets"
    targets.mkdir(parents=True)
    pd.DataFrame({"pickup_hour": [3600, 3600], "src": [1, 2], "dst": [2, 4], "trip_count": [2, 1]}).to_parquet(targets / "example.parquet", index=False)
    verifier = Verifier({"tests_dir": str(Path(__file__).resolve().parent), "workdir": tmp_path})
    return verifier, artifacts, targets


def test_recomputes_mrr_including_ties_and_checks_truth(run):
    verifier, _, _ = run
    scores = verifier.replay_mrr()
    assert scores == pytest.approx({"tgn": 1.0, "recent": 5 / 12, "seasonal": 7 / 12})
    assert "6 sampled labels" in verifier.check_ground_truth()


def test_rejects_different_candidates(run):
    verifier, artifacts, _ = run
    path = artifacts / "evaluation_seasonal.parquet"
    frame = pd.read_parquet(path)
    frame.loc[2, "target_zone"] = 9
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="different evaluation candidates"):
        verifier.replay_mrr()


def test_rejects_duplicate_positive_route(run):
    verifier, artifacts, _ = run
    for name in ("tgn", "recent", "seasonal"):
        path = artifacts / f"evaluation_{name}.parquet"
        frame = pd.read_parquet(path)
        frame.loc[1, ["source_zone", "target_zone"]] = [1, 2]
        frame.loc[[4, 5], "source_zone"] = 1
        frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Duplicate positive route"):
        verifier.replay_mrr()


def test_rejects_false_negative_label(run):
    verifier, _, targets = run
    path = targets / "example.parquet"
    frame = pd.read_parquet(path)
    frame.loc[len(frame)] = [3600, 1, 3, 1]
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="1 mismatches"):
        verifier.check_ground_truth()


def test_pass_gate_requires_strict_improvement():
    spec = {"pass_criteria": {"mrr_improvement_over_best_history": {"threshold": 0.0, "comparison": "gt"}}}
    for improvement, passed in [(-0.01, False), (0.0, False), (0.01, True)]:
        result = evaluate_primary_metrics({"mrr_improvement_over_best_history": improvement}, spec)
        assert result[0]["passed"] is passed


@pytest.fixture
def input_bundle(tmp_path, monkeypatch):
    data = tmp_path / "data"
    names = [f"{kind}/{year:04d}-{month:02d}.parquet"
             for kind in ("targets", "observations")
             for year in range(2023, 2027) for month in range(1, 13)
             if "2023-10" <= f"{year:04d}-{month:02d}" <= "2026-09"]
    names += ["zones.parquet", "zones.geojson", "validation.json"]
    files = {}
    for name in names:
        path = data / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = name.encode()
        path.write_bytes(payload)
        files[name] = {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "files": files}))
    monkeypatch.setenv("NYC_DATA_DIR", str(data))
    monkeypatch.setenv("NYC_DATA_MANIFEST", str(manifest))
    verifier = Verifier({"tests_dir": str(Path(__file__).resolve().parent), "workdir": tmp_path})
    return verifier, manifest


@pytest.mark.parametrize("missing", [None, "targets/2024-07.parquet", "observations/2026-09.parquet", "zones.parquet"])
def test_processed_bundle_requires_full_coverage_even_with_matching_inventory(input_bundle, missing):
    verifier, manifest = input_bundle
    if missing is None:
        assert "75 input files" in verifier.check_data()
        return
    (verifier.data / missing).unlink()
    inventory = json.loads(manifest.read_text())
    del inventory["files"][missing]
    manifest.write_text(json.dumps(inventory))
    with pytest.raises(ValueError, match="Expected all 36|Missing zone assets"):
        verifier.check_data()


def test_processed_bundle_rejects_changed_contents(input_bundle):
    verifier, _ = input_bundle
    path = verifier.data / "zones.parquet"
    path.write_bytes(b"x" * path.stat().st_size)
    with pytest.raises(ValueError, match="Input changed: zones.parquet"):
        verifier.check_data()
