# Data and model reference

[← Project overview](../README.md) · [Training](training.md) · [Development](development.md)

The source is NYC TLC yellow-taxi trip records. Data is historical and published
with a delay; the simulator models completed-trip demand rather than original
booking, cancellation, or rejection logs. Run CLI examples from the repository root.

## Dataset and splits

```bash
# October 2023 through September 2026, plus official zone geometry and lookup.
python -m mobility.download
# Alternative: discover and download the latest 36 months listed on the TLC page.
python -m mobility.download --latest-36 --data-dir data/latest

python -m mobility.preprocess --threads 4
python -m mobility.training --config configs/full.json --artifacts-dir artifacts/manual-full --epochs 10 --patience 3 --device auto
python -m mobility.evaluation --artifacts-dir artifacts/manual-full --device auto
```

The prescribed split is training October 2023–September 2025, validation October 2025–March 2026, and test April–September 2026. Full mode checks that all 36 target and observation files exist. The downloader resumes partial HTTP transfers when supported, verifies Parquet metadata, ZIP integrity, and the lookup CSV, and records status, URL, size, and timestamps in `data/raw/manifest.json`. Failed or unavailable downloads are reported and cause a nonzero exit; they are never silently replaced. Rerun the command to retry. Corrupt completed downloads are quarantined alongside the partial files.

`data/processed/validation.json` records accepted and dropped rows per input file. `data/processed/targets/` and `observations/` hold sparse directed Parquet aggregates; `zones.geojson` is transformed to WGS84 with names, boroughs, and projected-CRS centroids. The raw datasets, processed Parquet/GeoJSON files, and model checkpoints are versioned with Git LFS; runtime replay caches remain local.

## How the time boundary works

- TLC timestamps are naive New York local time. We discard records in the ambiguous fall-back hour and nonexistent spring-forward hour, convert the remaining timestamps to UTC, and display them in `America/New_York`. Daily and weekly lags use local wall-clock time across DST.
- Trips require valid zone geometries, timestamps in the file's pickup month, positive elapsed duration no greater than 180 minutes, distance in `[0, 100]` miles, and `fare_amount` in `[0, 500]` dollars. Nonfinite values are rejected. Because TLC has no unique trip ID, exact-looking duplicates are retained rather than deleting potentially real trips.
- Labels group trips by **pickup** hour and directed zone pair: count, mean duration, mean `fare_amount`, mean trip distance, and four pickup-time bins. `fare_amount` is the metered fare; `total_amount` includes additional charges and is not the target.
- A trip's destination, duration, fare, and distance enter graph state only after completion. To make this conservative at hourly cutoffs, a completion is made available at the next whole hour, including when the dropoff was exactly on a boundary. Previous-hour, same-local-hour-yesterday, and same-local-hour-in-prior-weeks counts include only trips completed and available at that cutoff. Labels may of course use the eventual completed trip: they are ground truth, not model inputs.
- The last three pickup hours in both the training and validation periods are excluded from fitting/selection because a valid trip can take up to three hours to finish. Observations continue to replay through these gaps. The API rejects forecasts before the validation selection boundary; this avoids presenting a model selected with later data as an earlier backtest. Checkpoints save model parameters, architecture, scaling metadata, split boundaries, and the earliest replay time. Memory itself is reset and reconstructed by replaying observations up to the requested cutoff. The training reference supplied with the request and the [PyG TGN example](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py) informed the memory, attention, chronological replay, and ranking design.

## Model and evaluation

`mobility/model.py` uses PyG `TGNMemory` with `IdentityMessage`/`MeanAggregator` and a `TransformerConv` over recent directed interactions. Mean aggregation handles the many interactions sharing each hourly timestamp without picking an arbitrary tied message. Static centroid features, great-circle distance between centroids, and cyclical hour/day/month features supplement the temporal memory. Separate heads predict link existence, nonnegative expected trip count, conditional duration, conditional metered fare, and four departure-bin probabilities. The configurable multi-task loss combines weighted BCE, Poisson likelihood, smooth-L1 attribute losses, and distributional cross-entropy; duration, fare, and departure losses apply only to positive routes. The fixed feature scales are saved in checkpoint metadata. The configuration JSON supports `loss_weights`, architecture sizes, seed, positive sample cap, negative ratio, and learning rate.

Training samples observed positive routes and uniformly samples valid zone pairs absent from that same hour. Positive and negative loss terms are weighted back to the population size so the deliberately balanced training sample does not by itself imply that half of all routes have trips. These unobserved pairs are negative **for that hour**, not invalid zones. Inference scores all ordered geometry-backed zone pairs in batches, including within-zone trips. Lines on the map show the highest-volume routes, while the citywide total sums all scored pairs.

Evaluation replays all prior completed observations and scores sampled held-out positives against up to 20 same-origin destinations absent in the same hour. It exports the sampled route-level forecasts and actuals to `artifacts/evaluation_*.parquet`, and writes `artifacts/metrics.json`. Metrics include one-versus-many MRR and Hits@10 (ties get average rank), ROC AUC, average precision, volume MAE/RMSE/Poisson deviance, positive-route duration and fare MAE, and departure-distribution total variation. These are **sampled-population** metrics, not citywide accuracy; ranking and classification scores depend on the negative sampling scheme. Change `--negatives`, `--max-positives`, or `--max-hours` to explore evaluation cost and sensitivity.

The recent baseline averages the last three completed-observation pickup hours; the seasonal baseline averages the same local hour over the preceding four weeks, treating unavailable history as zero. Both use only known historical route attributes for conditional duration, fare, and bin estimates. The dashboard compares their real held-out metrics with the TGN. Do not assume a small smoke checkpoint beats either baseline; inspect the saved results and use a longer run for a substantive assessment.
