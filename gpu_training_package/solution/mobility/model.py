"""PyG TGN memory, temporal graph attention, and multi-task route forecasts."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, MeanAggregator

from mobility.graph import calendar_features

NORMALIZATION = {"coordinates_center": [40.7, -73.9], "coordinates_scale": 0.25,
                 "duration_minutes_scale": 30, "fare_usd_scale": 30, "distance_miles_scale": 10,
                 "centroid_distance_miles_scale": 10, "route_count_transform": "log1p"}


class TaxiTGN(nn.Module):
    def __init__(self, zones, memory_dim=48, time_dim=16, hidden_dim=64):
        super().__init__()
        if hidden_dim % 2:
            raise ValueError("hidden_dim must be even")
        coords = zones[["latitude", "longitude"]].to_numpy(dtype=np.float32)
        self.register_buffer("centroid_radians", torch.as_tensor(np.deg2rad(coords)))
        self.register_buffer("static_features", torch.as_tensor((coords - np.array([40.7, -73.9], dtype=np.float32)) / 0.25))
        self.memory = TGNMemory(len(zones), 4, memory_dim, time_dim,
                                message_module=IdentityMessage(4, memory_dim, time_dim), aggregator_module=MeanAggregator())
        self.node_proj = nn.Linear(memory_dim + 2, hidden_dim)
        self.attention = TransformerConv(hidden_dim, hidden_dim // 2, heads=2, edge_dim=4 + time_dim, dropout=0.0)
        self.shared = nn.Sequential(nn.Linear(hidden_dim*2 + 3 + 6 + 1, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.link = nn.Linear(hidden_dim, 1)
        self.volume = nn.Linear(hidden_dim, 1)
        self.duration = nn.Linear(hidden_dim, 1)
        self.fare = nn.Linear(hidden_dim, 1)
        self.departure = nn.Linear(hidden_dim, 4)

    def encode(self, ctx, cutoff: int):
        nodes = torch.arange(ctx.n, device=ctx.device)
        memory, _ = self.memory(nodes)
        x = self.node_proj(torch.cat([memory, self.static_features], dim=-1)).relu()
        edge_index, messages, age = ctx.edge_tensors(cutoff)
        edge_attr = torch.cat([messages, self.memory.time_enc(age)], dim=-1)
        return self.attention(x, edge_index, edge_attr).relu()

    def score(self, z, ctx, cutoff: int, src: np.ndarray, dst: np.ndarray):
        device = z.device
        s = torch.as_tensor(src, dtype=torch.long, device=device)
        d = torch.as_tensor(dst, dtype=torch.long, device=device)
        history = torch.as_tensor(np.log1p(ctx.counts(cutoff, src, dst)), device=device)
        calendar = torch.as_tensor(calendar_features(cutoff), device=device).expand(len(src), -1)
        a, b = self.centroid_radians[s], self.centroid_radians[d]
        delta = b - a
        hav = torch.sin(delta[:, 0]/2).square() + torch.cos(a[:, 0])*torch.cos(b[:, 0])*torch.sin(delta[:, 1]/2).square()
        distance = (2 * 3958.7613 * torch.asin(hav.clamp(0, 1).sqrt()) / 10).unsqueeze(-1)
        h = self.shared(torch.cat([z[s], z[d], history, calendar, distance], dim=-1))
        return {"link_logits": self.link(h).squeeze(-1),
                "volume": F.softplus(self.volume(h).squeeze(-1)) + 1e-6,
                "duration": F.softplus(self.duration(h).squeeze(-1))*30,
                "fare": F.softplus(self.fare(h).squeeze(-1))*30,
                "departure_logits": self.departure(h)}


def multitask_loss(output, positives, weights=None, population_positives=None, population_pairs=None):
    weights = weights or {"link": 1.0, "volume": 0.15, "duration": 1.0, "fare": 1.0, "departure": 0.5}
    n = len(positives)
    labels = torch.zeros_like(output["link_logits"]); labels[:n] = 1
    counts = torch.zeros_like(labels)
    counts[:n] = torch.as_tensor(positives.trip_count.to_numpy(copy=True), dtype=counts.dtype, device=counts.device)
    sample_weights = torch.ones_like(labels)
    if population_positives is not None and population_pairs is not None and n and len(labels) > n:
        # Correct for the deliberately balanced training sample. Without this,
        # the prevalence of routes and citywide volumes are badly inflated.
        sample_weights[:n] = population_positives / n
        sample_weights[n:] = (population_pairs - population_positives) / (len(labels)-n)
    sample_weights = sample_weights / sample_weights.sum().clamp_min(1)
    loss = weights["link"] * (F.binary_cross_entropy_with_logits(output["link_logits"], labels, reduction="none") * sample_weights).sum()
    loss = loss + weights["volume"] * (F.poisson_nll_loss(output["volume"], counts, log_input=False, reduction="none") * sample_weights).sum()
    if n:
        dur = torch.as_tensor(positives.duration.to_numpy(copy=True), dtype=counts.dtype, device=counts.device)
        fare = torch.as_tensor(positives.fare.to_numpy(copy=True), dtype=counts.dtype, device=counts.device)
        bins = torch.as_tensor(positives[["bin0", "bin1", "bin2", "bin3"]].to_numpy(copy=True), dtype=counts.dtype, device=counts.device)
        dist = bins / counts[:n, None].clamp_min(1)
        loss = loss + weights["duration"] * F.smooth_l1_loss(output["duration"][:n]/30, dur/30)
        loss = loss + weights["fare"] * F.smooth_l1_loss(output["fare"][:n]/30, fare/30)
        loss = loss - weights["departure"] * (dist * F.log_softmax(output["departure_logits"][:n], dim=-1)).sum(dim=-1).mean()
    return loss
