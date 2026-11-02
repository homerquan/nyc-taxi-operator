"""Regenerate held-out forecasts and compare the TGN to two causal baselines."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from mobility.graph import GraphContext, format_time, hour_groups
from mobility.inference import load_model, predict_pairs
from mobility.preprocess import sql_string
from mobility.provenance import sha256_file
from mobility.training import runtime_info, write_json


def evaluation_candidates(rows, ctx, rng, max_positives, negatives):
    rows = rows[rows.src.isin(ctx.index) & rows.dst.isin(ctx.index)]
    if len(rows) > max_positives:
        selected = rows.iloc[np.sort(rng.choice(len(rows), max_positives, replace=False))]
    else:
        selected = rows
    occupied = {(ctx.index[int(r.src)], ctx.index[int(r.dst)]) for r in rows.itertuples(index=False)}
    src = [ctx.index[int(x)] for x in selected.src]
    dst = [ctx.index[int(x)] for x in selected.dst]
    negative_src, negative_dst, groups = [], [], []
    for source in src:
        valid = [d for d in range(ctx.n) if (source, d) not in occupied]
        sample = rng.choice(valid, min(len(valid), negatives), replace=False)
        groups.append(list(range(len(selected) + len(negative_dst), len(selected) + len(negative_dst) + len(sample))))
        negative_src.extend([source] * len(sample)); negative_dst.extend(sample.tolist())
    return selected, np.array(src + negative_src, dtype=np.int64), np.array(dst + negative_dst, dtype=np.int64), groups


def metric_summary(path: Path, ranks: list[float]) -> dict:
    # Compute summaries from disk, including exact tie-aware ROC AUC and AP.
    # This avoids retaining all sampled examples for a full six-month test.
    with duckdb.connect() as con:
        source = f"read_parquet({sql_string(path)})"
        row = con.execute(f"""SELECT count(*), count(*) FILTER (WHERE actual_trips > 0),
          avg(abs(greatest(predicted_trips, 1e-6)-actual_trips)),
          sqrt(avg(power(greatest(predicted_trips, 1e-6)-actual_trips, 2))),
          avg(2 * CASE WHEN actual_trips=0 THEN greatest(predicted_trips, 1e-6)
              ELSE actual_trips * ln(actual_trips/greatest(predicted_trips, 1e-6))
              - actual_trips + greatest(predicted_trips, 1e-6) END),
          avg(abs(predicted_duration-actual_duration)) FILTER (WHERE actual_trips>0),
          avg(abs(predicted_fare-actual_fare)) FILTER (WHERE actual_trips>0),
          avg(departure_tvd) FILTER (WHERE actual_trips>0) FROM {source}""").fetchone()
        auc, ap = con.execute(f"""WITH grouped AS (
            SELECT link_probability, count(*) FILTER (WHERE actual_trips>0)::DOUBLE AS p,
                   count(*) FILTER (WHERE actual_trips=0)::DOUBLE AS n FROM {source} GROUP BY link_probability
        ), ranked AS (
            SELECT *, coalesce(sum(n) OVER (ORDER BY link_probability ASC ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS below,
                   sum(p) OVER (ORDER BY link_probability DESC) AS p_above,
                   sum(p+n) OVER (ORDER BY link_probability DESC) AS all_above FROM grouped
        ) SELECT sum(p*(below+n/2))/nullif(sum(p)*sum(n),0),
                 sum(p*p_above/nullif(all_above,0))/nullif(sum(p),0) FROM ranked""").fetchone()
    return {"examples": row[0], "positive_routes": row[1],
            "mrr": float(np.mean([1/r for r in ranks])) if ranks else None,
            "hits_at_10": float(np.mean([r <= 10 for r in ranks])) if ranks else None,
            "roc_auc": auc, "average_precision": ap,
            "volume_mae": row[2], "volume_rmse": row[3], "poisson_deviance": row[4],
            "duration_mae": row[5], "fare_mae": row[6], "departure_total_variation": row[7]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu")
    parser.add_argument("--max-hours", type=int, help="Optional limit for a smoke evaluation")
    parser.add_argument("--max-positives", type=int, default=256)
    parser.add_argument("--negatives", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--log-every-hours", type=int, default=720, help="Report replay/evaluation progress every N hours; 0 disables periodic reports")
    args = parser.parse_args()
    if args.max_positives < 1 or args.negatives < 1 or args.log_every_hours < 0 or (args.max_hours is not None and args.max_hours < 1):
        raise SystemExit("Sampling and hour limits must be positive")
    torch.set_num_threads(max(1, args.threads))
    model, zones, meta, device = load_model(args.data_dir, args.artifacts_dir, args.device)
    ctx = GraphContext(zones, device)
    observed = iter(hour_groups(args.data_dir, "observations", meta["test_end"], meta["start"]))
    pending = next(observed, None)
    rng = np.random.default_rng(meta["config"]["seed"] + 2)
    names = ["tgn", "recent", "seasonal"]
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    writers = {name: None for name in names}
    ranks = {name: [] for name in names}
    processed = 0
    replayed = 0
    started = time.monotonic()
    runtime = runtime_info(device)

    def report(status, cutoff=None):
        progress = {"status": status, "cutoff": format_time(cutoff) if cutoff is not None else None,
                    "replayed_hours": replayed, "evaluated_hours": processed,
                    "elapsed_seconds": round(time.monotonic()-started, 1), "runtime": runtime}
        write_json(args.artifacts_dir / "evaluation_progress.json", progress)
        print(json.dumps({"event": "evaluation_progress", **progress}), flush=True)

    report("starting")
    for cutoff, rows in hour_groups(args.data_dir, "targets", meta["test_end"] - 1, meta["start"]):
        while pending is not None and pending[0] <= cutoff:
            ctx.observe(model, pending[0], pending[1], cutoff)
            pending = next(observed, None)
        replayed += 1
        if cutoff < meta["validation_end"]:
            if args.log_every_hours and replayed % args.log_every_hours == 0:
                report("replaying", cutoff)
            continue
        selected, src, dst, groups = evaluation_candidates(rows, ctx, rng, args.max_positives, args.negatives)
        n = len(selected)
        if not n:
            continue
        p = predict_pairs(model, ctx, cutoff, src, dst)
        predictions = {"tgn": (np.array(p["link_probability"]), np.array(p["predicted_trips"]),
                               np.array(p["predicted_duration_min"]), np.array(p["predicted_fare_usd"]),
                               np.array(p["departure_bin_probabilities"]))}
        for name, seasonal in [("recent", False), ("seasonal", True)]:
            v, duration, fare, bins = ctx.baseline(cutoff, src, dst, seasonal)
            predictions[name] = (1-np.exp(-v), v, duration, fare, bins)
        actual_count = np.zeros(len(src)); actual_count[:n] = selected.trip_count.to_numpy()
        actual_duration = np.zeros(len(src)); actual_duration[:n] = selected.duration.to_numpy()
        actual_fare = np.zeros(len(src)); actual_fare[:n] = selected.fare.to_numpy()
        actual_bins = np.zeros((len(src), 4)); actual_bins[:n] = selected[["bin0", "bin1", "bin2", "bin3"]].to_numpy() / actual_count[:n, None]
        # Persist the candidate group, so the verifier can independently
        # reconstruct reciprocal ranks from the saved predictions.
        group_id = np.empty(len(src), dtype=np.int32)
        group_id[:n] = np.arange(n, dtype=np.int32)
        for i, indices in enumerate(groups):
            group_id[indices] = i
        for name, (link, volume, duration, fare, bins) in predictions.items():
            for i, indices in enumerate(groups):
                # Average rank for ties, following the one-positive-vs-many convention.
                scores = link[indices]
                ranks[name].append(float(1 + (scores > link[i]).sum() + 0.5*(scores == link[i]).sum()))
            frame = pd.DataFrame({"cutoff": cutoff, "group_id": group_id, "source_zone": np.array(ctx.ids)[src], "target_zone": np.array(ctx.ids)[dst],
                                  "actual_trips": actual_count, "actual_duration": actual_duration, "actual_fare": actual_fare,
                                  "link_probability": link, "predicted_trips": volume, "predicted_duration": duration,
                                  "predicted_fare": fare, "departure_tvd": np.abs(bins-actual_bins).sum(axis=1)/2})
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writers[name] is None:
                writers[name] = pq.ParquetWriter(args.artifacts_dir / f"evaluation_{name}.parquet", table.schema, compression="zstd")
            writers[name].write_table(table)
        processed += 1
        if args.log_every_hours and replayed % args.log_every_hours == 0:
            report("evaluating", cutoff)
        if args.max_hours and processed >= args.max_hours:
            break
    if not processed:
        raise SystemExit("No held-out hours found for this checkpoint")
    for writer in writers.values():
        writer.close()
    metrics = {"population": {"description": "Sampled positive routes and up to N same-origin routes absent in the same pickup hour; metrics are not citywide.",
                               "negatives_per_positive": args.negatives, "max_positives_per_hour": args.max_positives,
                               "hours": processed, "test_start": format_time(meta["validation_end"]), "test_end_exclusive": format_time(meta["test_end"]),
                               "seed": meta["config"]["seed"]+2, "max_hours": args.max_hours},
               "provenance": {"model_sha256": sha256_file(args.artifacts_dir / "model.pt")},
               "runtime": runtime, "models": {}}
    for name in names:
        metrics["models"][name] = metric_summary(args.artifacts_dir / f"evaluation_{name}.parquet", ranks[name])
    (args.artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    report("complete", cutoff)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
