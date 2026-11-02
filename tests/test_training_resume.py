"""An epoch-boundary resume must retain optimizer state and split protection."""
import json
import sys
import copy

import pandas as pd
import torch

from mobility import training
from mobility.graph import HOUR


def test_resuming_matches_uninterrupted_cpu_training(dataset, tmp_path, monkeypatch):
    data, _, _ = dataset
    target_path = data / "processed/targets/2025-01.parquet"
    obs_path = data / "processed/observations/2025-01.parquet"
    target = pd.read_parquet(target_path)
    obs = pd.read_parquet(obs_path)
    targets = []
    observations = []
    for hour in range(24):
        targets.append(target.assign(pickup_hour=target.pickup_hour + hour*HOUR))
        observations.append(obs.assign(pickup_hour=obs.pickup_hour + hour*HOUR, available_at=obs.available_at + hour*HOUR))
    pd.concat(targets).to_parquet(target_path, index=False)
    pd.concat(observations).to_parquet(obs_path, index=False)
    config = {"memory_dim":8, "time_dim":4, "hidden_dim":8, "negative_ratio":1, "max_positives":2, "learning_rate":0.001, "seed":42}
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    captured = []
    original_walk = training.walk
    def capture(*args, **kwargs):
        captured.append((copy.deepcopy(args[0].state_dict()), copy.deepcopy(args[7].state_dict()), args[0].training, torch.get_rng_state().clone()))
        return original_walk(*args, **kwargs)
    monkeypatch.setattr(training, "walk", capture)

    def run(destination, epochs, resume=None):
        args = ["training", "--data-dir", str(data), "--artifacts-dir", str(destination), "--config", str(config_path),
                "--quick", "--max-hours", "24", "--epochs", str(epochs), "--device", "cpu", "--threads", "1", "--log-every-hours", "0"]
        if resume:
            args += ["--resume", str(resume)]
        monkeypatch.setattr(sys, "argv", args)
        training.main()

    resumed = tmp_path / "resumed"
    uninterrupted = tmp_path / "uninterrupted"
    run(resumed, 1)
    run(resumed, 2, resumed / "training_state.pt")
    run(uninterrupted, 2)
    for key in captured[1][0]:
        assert torch.equal(captured[1][0][key], captured[3][0][key]), f"start of resumed epoch: {key}"
    assert captured[1][1]["param_groups"] == captured[3][1]["param_groups"]
    assert captured[1][2] == captured[3][2]
    assert torch.equal(captured[1][3], captured[3][3]), "torch RNG differs"
    for param_id, values in captured[1][1]["state"].items():
        for key, value in values.items():
            other = captured[3][1]["state"][param_id][key]
            assert torch.equal(value, other) if isinstance(value, torch.Tensor) else value == other
    a = torch.load(resumed / "training_state.pt", map_location="cpu", weights_only=False)
    b = torch.load(uninterrupted / "training_state.pt", map_location="cpu", weights_only=False)
    assert a["epoch"] == b["epoch"] == 2
    for key in a["state_dict"]:
        if not key.endswith("_assoc"):
            assert torch.equal(a["state_dict"][key], b["state_dict"][key]), key
    assert json.loads((resumed / "progress.json").read_text())["status"] == "complete"
