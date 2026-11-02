"""Chronological, bounded-memory readers and graph state for TLC aggregates."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import torch

from mobility.preprocess import sql_string

NY = ZoneInfo("America/New_York")
HOUR = 3600


def format_time(epoch: int) -> str:
    return datetime.fromtimestamp(int(epoch), NY).isoformat()


def lag_time(cutoff: int, hours: int) -> int:
    """Daily/weekly lags follow New York wall-clock time across DST."""
    if hours % 24:
        return cutoff - hours * HOUR
    local = datetime.fromtimestamp(int(cutoff), NY)
    lagged = local.replace(tzinfo=None) - timedelta(hours=hours)
    candidate = lagged.replace(tzinfo=NY, fold=local.fold)
    # Nonexistent local hours have no reliable TLC timestamp.
    if datetime.fromtimestamp(candidate.timestamp(), NY).replace(tzinfo=None) != lagged:
        return -1
    return int(candidate.timestamp())


def read_zones(data_dir: Path) -> pd.DataFrame:
    return pd.read_parquet(data_dir / "zones.parquet").sort_values("zone_id").reset_index(drop=True)


def bounds(data_dir: Path) -> tuple[int, int]:
    glob = data_dir / "targets" / "*.parquet"
    if not list(glob.parent.glob(glob.name)):
        raise FileNotFoundError(f"No processed targets in {glob.parent}")
    row = duckdb.connect().execute(f"SELECT min(pickup_hour), max(pickup_hour) FROM read_parquet({sql_string(glob)})").fetchone()
    return int(row[0]), int(row[1])


def hour_groups(data_dir: Path, kind: str, end: int | None = None, start: int | None = None):
    """Yield complete, ordered hour frames, without loading all rows into RAM."""
    if kind not in {"targets", "observations"}:
        raise ValueError(kind)
    col = "pickup_hour" if kind == "targets" else "available_at"
    glob = data_dir / kind / "*.parquet"
    con = duckdb.connect()
    conditions = []
    if end is not None:
        conditions.append(f"{col} <= {int(end)}")
    if start is not None:
        # Exclude trips picked up before the training dataset began, even if
        # their completion was reported after the start.
        conditions.append(f"pickup_hour >= {int(start)}")
    where = "WHERE " + " AND ".join(conditions) if conditions else ""
    tie_break = ", pickup_hour" if kind == "observations" else ""
    reader = con.execute(f"SELECT * FROM read_parquet({sql_string(glob)}) {where} ORDER BY {col}, src, dst{tie_break}").to_arrow_reader(65536)
    carry = pd.DataFrame()
    for batch in reader:
        frame = batch.to_pandas()
        if not carry.empty:
            frame = pd.concat([carry, frame], ignore_index=True)
        if frame.empty:
            continue
        last = frame[col].iloc[-1]
        carry = frame[frame[col] == last].copy()
        for timestamp, group in frame[frame[col] != last].groupby(col, sort=False):
            yield int(timestamp), group
    if not carry.empty:
        yield int(carry[col].iloc[0]), carry
    con.close()


class GraphContext:
    """Only observations available by the current cutoff may enter this state."""
    def __init__(self, zones: pd.DataFrame, device: torch.device, neighbor_hours: int = 6, max_edges_per_hour: int = 512):
        self.zones = zones
        self.device = device
        self.ids = zones.zone_id.astype(int).tolist()
        self.index = {zone: i for i, zone in enumerate(self.ids)}
        self.n = len(self.ids)
        self.history: dict[int, dict[int, int]] = {}
        self.route_stats: dict[int, np.ndarray] = {}
        self.edges: deque = deque()
        self.neighbor_hours = neighbor_hours
        self.max_edges_per_hour = max_edges_per_hour
        self.latest_observation = -1

    def observe(self, model, available_at: int, rows: pd.DataFrame, cutoff: int) -> None:
        if available_at > cutoff or available_at < self.latest_observation:
            raise ValueError("Observations must be chronological and available at the cutoff")
        self.latest_observation = available_at
        rows = rows[rows.src.isin(self.index) & rows.dst.isin(self.index)]
        if rows.empty:
            return
        src = np.array([self.index[int(x)] for x in rows.src], dtype=np.int64)
        dst = np.array([self.index[int(x)] for x in rows.dst], dtype=np.int64)
        counts = rows.trip_count.to_numpy(dtype=np.float32)
        msg = np.column_stack((np.log1p(counts), rows.duration_sum.to_numpy()/counts/30,
                               rows.fare_sum.to_numpy()/counts/30, rows.distance_sum.to_numpy()/counts/10)).astype(np.float32)
        for i, row in enumerate(rows.itertuples(index=False)):
            key = int(src[i]) * self.n + int(dst[i])
            hour = int(row.pickup_hour)
            bucket = self.history.setdefault(hour, {})
            bucket[key] = bucket.get(key, 0) + int(row.trip_count)
            stats = self.route_stats.setdefault(key, np.zeros(8, dtype=np.float64))
            stats += np.array([row.trip_count, row.duration_sum, row.fare_sum, row.distance_sum,
                               row.bin0, row.bin1, row.bin2, row.bin3])
        keep = np.argsort(-counts, kind="stable")[:self.max_edges_per_hour]
        self.edges.append((available_at, src[keep], dst[keep], msg[keep]))
        self.prune(cutoff)
        with torch.no_grad():
            model.memory.update_state(torch.as_tensor(src, device=self.device), torch.as_tensor(dst, device=self.device),
                                      torch.full((len(src),), available_at // HOUR, dtype=torch.long, device=self.device),
                                      torch.as_tensor(msg, device=self.device))

    def prune(self, cutoff: int) -> None:
        while self.edges and self.edges[0][0] < cutoff - self.neighbor_hours * HOUR:
            self.edges.popleft()
        for hour in list(self.history):
            if hour < cutoff - (4 * 7 * 24 + 2) * HOUR:
                del self.history[hour]

    def edge_tensors(self, cutoff: int):
        self.prune(cutoff)
        if not self.edges:
            return (torch.empty((2, 0), dtype=torch.long, device=self.device),
                    torch.empty((0, 4), device=self.device), torch.empty((0,), device=self.device))
        src = np.concatenate([x[1] for x in self.edges]); dst = np.concatenate([x[2] for x in self.edges])
        msgs = np.concatenate([x[3] for x in self.edges])
        age = np.concatenate([np.full(len(x[1]), (cutoff-x[0])/HOUR, dtype=np.float32) for x in self.edges])
        return (torch.as_tensor(np.stack([src, dst]), dtype=torch.long, device=self.device),
                torch.as_tensor(msgs, device=self.device), torch.as_tensor(age, device=self.device))

    def counts(self, cutoff: int, src: np.ndarray, dst: np.ndarray, lags=(1, 24, 168)) -> np.ndarray:
        keys = src * self.n + dst
        lagged = [lag_time(cutoff, lag) for lag in lags]
        return np.array([[self.history.get(hour, {}).get(int(k), 0) for hour in lagged] for k in keys], dtype=np.float32)

    def baseline(self, cutoff: int, src: np.ndarray, dst: np.ndarray, seasonal: bool):
        lags = (168, 336, 504, 672) if seasonal else (1, 2, 3)
        volume = self.counts(cutoff, src, dst, lags).mean(axis=1)
        duration, fare, bins = [], [], []
        for key in src*self.n+dst:
            s = self.route_stats.get(int(key), np.zeros(8))
            count = s[0]
            duration.append(s[1]/count if count else 0.0)
            fare.append(s[2]/count if count else 0.0)
            bins.append((s[4:8]/count).tolist() if count else [0.25]*4)
        return volume, np.asarray(duration), np.asarray(fare), np.asarray(bins)


def calendar_features(timestamp: int) -> np.ndarray:
    t = datetime.fromtimestamp(int(timestamp), NY)
    hour = t.hour * 2 * np.pi / 24
    day = t.weekday() * 2 * np.pi / 7
    month = (t.month - 1) * 2 * np.pi / 12
    return np.array([np.sin(hour), np.cos(hour), np.sin(day), np.cos(day), np.sin(month), np.cos(month)], dtype=np.float32)


def candidates(rows: pd.DataFrame, ctx: GraphContext, rng: np.random.Generator, max_positives: int, negative_ratio: int):
    """Sample positives and uniform valid negatives absent from the entire target hour."""
    rows = rows[rows.src.isin(ctx.index) & rows.dst.isin(ctx.index)]
    if len(rows) > max_positives:
        selected = rows.iloc[np.sort(rng.choice(len(rows), max_positives, replace=False))]
    else:
        selected = rows
    occupied = {(ctx.index[int(r.src)], ctx.index[int(r.dst)]) for r in rows.itertuples(index=False)}
    pos_src = np.array([ctx.index[int(x)] for x in selected.src], dtype=np.int64)
    pos_dst = np.array([ctx.index[int(x)] for x in selected.dst], dtype=np.int64)
    wanted = min(len(pos_src) * negative_ratio, ctx.n * ctx.n - len(occupied))
    occupied_ids = {s * ctx.n + d for s, d in occupied}
    negative_ids: set[int] = set()
    while len(negative_ids) < wanted:
        count = min(ctx.n * ctx.n, max(32, (wanted-len(negative_ids))*2))
        for candidate in rng.choice(ctx.n * ctx.n, count, replace=False):
            key = int(candidate)
            if key not in occupied_ids:
                negative_ids.add(key)
                if len(negative_ids) == wanted:
                    break
    ordered = np.array(sorted(negative_ids), dtype=np.int64)
    return selected, np.concatenate([pos_src, ordered // ctx.n]), np.concatenate([pos_dst, ordered % ctx.n])
