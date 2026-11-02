"""Exercise the actual PyG memory, attention, loss, and optimizer on CUDA."""
import numpy as np
import pytest
import torch

from mobility.graph import GraphContext, hour_groups
from mobility.model import TaxiTGN, multitask_loss
from mobility.training import device_for, runtime_info


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tgn_training_step_on_cuda(dataset):
    data, zones, _ = dataset
    device = device_for("cuda")
    model = TaxiTGN(zones, memory_dim=8, time_dim=4, hidden_dim=8).to(device)
    model.train()
    model.memory.reset_state()
    ctx = GraphContext(zones, device)
    timestamp, observations = next(hour_groups(data, "observations"))
    ctx.observe(model, timestamp, observations, timestamp)
    positives = next(hour_groups(data, "targets"))[1].iloc[:1]
    src = np.array([0, 0], dtype=np.int64)
    dst = np.array([1, 2], dtype=np.int64)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    result = model.score(model.encode(ctx, timestamp), ctx, timestamp, src, dst)
    loss = multitask_loss(result, positives)
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()
    assert result["volume"].device.type == "cuda"
    assert model.memory.memory.device.type == "cuda"
    assert model.link.weight.grad is not None and model.link.weight.grad.device.type == "cuda"
    assert model.memory.gru.weight_ih.grad is not None and model.memory.gru.weight_ih.grad.device.type == "cuda"
    assert any(p.grad is not None and p.grad.device.type == "cuda" for p in model.attention.parameters())
    assert torch.isfinite(loss).item()
    assert runtime_info(device)["gpu"]


def test_explicit_cuda_does_not_silently_fall_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        device_for("cuda")
