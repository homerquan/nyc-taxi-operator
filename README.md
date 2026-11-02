# Motion / NYC

A local research demo that forecasts directed yellow-taxi trips between NYC TLC taxi zones for the next hour. It downloads official TLC Parquet files, builds pickup-hour labels and completion-time graph observations, trains a PyTorch Geometric Temporal Graph Network, compares it with two historical baselines, and displays the results in a React/MapLibre/deck.gl dashboard.

TLC data is published after a delay. These are historical backtests or forecasts based on the most recently processed monthly data, **not live traffic**. A small checkpoint is useful for checking the workflow; it is not a production-quality traffic model.

## Clone with datasets and checkpoints

Large TLC datasets, evaluation Parquet files, taxi-zone geometry/archives, and
PyTorch checkpoints use **Git LFS**. Install Git LFS before cloning:

```bash
git lfs install
git clone git@github.com:homerquan/nyc-taxi-operator.git
cd nyc-taxi-operator
git lfs pull
```

The checkout includes the full and quick datasets, the saved full model, and
held-out evaluation artifacts. Allow several GB for LFS downloads and the
checkout, plus space for Python/Node dependencies. To download source only,
clone with `GIT_LFS_SKIP_SMUDGE=1`; run `git lfs pull` when ready to fetch data.
Virtual environments, installed Node packages, frontend build output, derived
replay caches, and temporary dashboard training runs remain local.

## Fleet operations dashboard

The app now has three workspaces: **Fleet simulation**, **Demand forecast**, and **Model & training**.

With the existing environment, data, and checkpoints:

```bash
.venv/bin/python -m pip install -e '.[test]'
npm --prefix frontend install
npm --prefix frontend run build
.venv/bin/python scripts/serve.py
```

Open <http://127.0.0.1:8000>. The launcher works from any directory. The server prefers `artifacts/cuda-full/model.pt` when present and otherwise uses `artifacts/model.pt`. `--demo` selects the small January 2025 checkpoint and `data/quick`. Explicit `NYC_DATA_DIR` and `NYC_ARTIFACTS_DIR` override the defaults. Model selection in the UI applies to the running server; restart restores the environment/default selection.

In **Fleet simulation**, select a historical start, fleet size, fuel price, MPG, fare multiplier, and a 1–6 hour horizon. Advanced controls include market share, maximum pickup wait, driver cost per taxi-hour, and random seed. Click **Run comparison**, then play, pause, scrub, or switch between **1×, 10×, 25×, 50×, and 100×**. One real second advances 100 simulated seconds at 100×; the policies are computed in the background, so speed changes only playback. Static map geometry and historical hour options are reused during playback, and the clock pauses while its workspace or browser tab is hidden. The fleet map works without a basemap service. Vehicle positions interpolate zone centroids and are approximate, not street routes.

The engine uses **SimPy** (the discrete-event Python simulation library) to generate arrivals from TLC historical directed route counts and quarter-hour bins. Within-bin arrival times are seeded; fares, distances, and trip durations use recorded aggregate means. A demand scale of 1.0 replays the completed-trip market, which excludes canceled, rejected, or unrecorded demand. Fare multipliers change offered fares without modeling demand elasticity. Each selected hour must have data; windows over 50,000 modeled requests return a clear error instead of being silently sampled.

Both policies receive identical requests and forecast-weighted initial fleet locations. **First come** assigns the nearest feasible idle taxi in arrival order. **Forecast policy** uses TGN duration and demand forecasts, the current fare quote, fuel for loaded and empty miles, pickup deadlines, and the opportunity cost of occupying a taxi. It maximizes estimated net earnings per occupied minute, with a forecast destination-value credit. TGN durations are stabilized by a 28-day completed-trip prior strictly available before the start. Every 30 seconds, maximum-weight bipartite matching assigns the best positive-value batch with an explicit idle option. The forecast is frozen at the start, so future request labels never enter dispatch. The policy is a forecast-guided heuristic, not a trained reinforcement-learning agent or a globally optimal full-shift plan; it can earn less than the baseline.

The scorecard and timeline compare completed fares, fuel, driver cost, net revenue, accepted/completed/declined requests, and pickup wait. **Net revenue = completed metered fares − fuel committed for dispatched trips − configured driver shift cost.** Fuel is charged for the entire trip at dispatch and revenue on completion. Trips still in service at the horizon are displayed separately. Insurance, maintenance, tolls, taxes, and other costs are excluded. Quotes and fuel prices are editable assumptions, not current market data.

In **Model & training**, choose an existing full or demo checkpoint, start CPU/CUDA training, resume a matching epoch checkpoint, and inspect losses, progress, and logs. Dashboard training writes each fresh run to a separate subdirectory of `artifacts/dashboard-quick` or `artifacts/dashboard-full`, preserving existing checkpoints. Resume continues the last dashboard run in place. After both training and held-out evaluation succeed, the new profile becomes active; reload the dashboard. Quick runs use `data/quick` when available unless `NYC_DATA_DIR` is explicitly set. Full runs require the complete prescribed window. Jobs run in this single API process, one per kind, and reset on restart; completed job results are bounded to eight entries. Use one Uvicorn worker for this local app.

Forecast replay now reuses exact temporal state between consecutive hours and saves a derived `replay-cpu.pt` snapshot in the checkpoint directory. The first full-history replay can take several minutes; the UI runs simulations as background jobs while it prepares. Subsequent restarts can reuse the snapshot. Checkpoint/data changes invalidate it. Earlier cutoffs rebuild safely without carrying future memory backwards. The date selector is constrained to the checkpoint's held-out period, and the backend resolves project paths independently of the shell's directory.

Run a comparison from the command line:

```bash
.venv/bin/python scripts/simulate.py --as-of 2026-04-01T00:00:00-04:00 --hours 2 --fleet-size 80 --output /tmp/fleet-replay.json
```

Additional APIs: `GET /api/models`, `POST /api/models/select`, `GET/POST /api/training`, `POST /api/simulations`, and `GET /api/jobs/{id}`. Training and simulation submissions return HTTP 202 with a job ID; poll until `complete` or `failed`. Configuration values are validated, and overlapping jobs of the same kind return HTTP 409. This remains a local, unauthenticated research tool.

## Requirements

- Python 3.12, Node.js 20 or later, and npm.
- Several GB of disk for the raw 36-month download, more for intermediate Parquet files, evaluation exports, packages, and DuckDB temporary storage. Full training is compute-intensive; CUDA is supported, as is CPU. MPS can be selected where the installed PyTorch/PyG operations support it.
- Internet access to download TLC data and, in the browser, the keyless OpenFreeMap basemap and fonts.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
cd frontend && npm install && cd ..
```

For a machine with an existing compatible PyTorch installation, install the appropriate CPU, CUDA, or MPS build first following the [PyTorch installation instructions](https://pytorch.org/get-started/locally/). No paid map token is needed.

### NVIDIA DGX Spark / GB10

The GB10 is an ARM64 CUDA GPU. A CPU-only PyTorch wheel will make `--device auto` choose the CPU even when `nvidia-smi` sees the GPU. After the setup above, install a CUDA-enabled PyTorch build into the project virtual environment. For this Spark's CUDA 13 driver and Python 3.12, the following official wheel was verified:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu130 'torch==2.14.1+cu130'
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
```

Run an isolated GPU smoke train and evaluation without replacing an existing demo model:

```bash
python -m mobility.training --data-dir data/quick --artifacts-dir artifacts/cuda-smoke --quick --max-hours 240 --max-positives 256 --epochs 3 --device cuda
python -m mobility.evaluation --data-dir data/quick --artifacts-dir artifacts/cuda-smoke --device cuda
```

For the full processed window:

```bash
python -m mobility.training --data-dir data --artifacts-dir artifacts/cuda-full --config configs/full.json --epochs 10 --patience 3 --device cuda
python -m mobility.evaluation --data-dir data --artifacts-dir artifacts/cuda-full --device cuda
```

`--device cuda` fails with a clear error if CUDA is unavailable; `--device auto` prefers CUDA, then MPS, then CPU. Training prints the GPU name, compute capability, PyTorch and CUDA versions and records them in each checkpoint. It writes `progress.json` during each epoch and keeps both the best model and an epoch-boundary optimizer checkpoint. If interrupted, resume from the last completed epoch with the **same dataset and configuration**, and use the total desired epoch count:

```bash
python -m mobility.training --data-dir data --artifacts-dir artifacts/cuda-full --config configs/full.json --epochs 10 --patience 3 --device cuda --resume artifacts/cuda-full/training_state.pt
```

Use `--log-every-hours 24` for more frequent progress updates, or `0` to disable periodic reports. Evaluation also records its replay progress in `evaluation_progress.json`. A partially completed epoch is replayed after an interruption. CUDA kernels may produce slightly different floating-point results across runs or devices even with the same seed. The trained CUDA checkpoint can also be loaded on CPU for API serving; to serve it, set `NYC_ARTIFACTS_DIR=artifacts/cuda-full` alongside `NYC_DATA_DIR=data`.

## Quick, reproducible real-data run

This uses January 2025 and the first ten days, with a chronological 70/15/15 split. It has deliberately few training epochs and should only be used as a smoke test.

```bash
python -m mobility.download --months 2025-01 --data-dir data/quick
python -m mobility.preprocess --months 2025-01 --data-dir data/quick
python -m mobility.training --data-dir data/quick --quick --max-hours 240 --max-positives 256 --epochs 3 --device cpu
python -m mobility.evaluation --data-dir data/quick --max-positives 256 --negatives 20 --device cpu
```

In one terminal, serve the API and its checkpoint:

```bash
NYC_DATA_DIR=data/quick uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal, start the dashboard:

```bash
cd frontend
npm run dev
```

Open `http://localhost:5173`. The development server proxies `/api` to port 8000. Inference reconstructs temporal memory from the saved data; the first request for a selected hour can take time, especially with a full dataset. The four most recent forecasts and the current chronological replay state are cached in the API process.

`--dev` downloads the most recently published month instead; `--months 2025-01 2025-02 2025-03` selects up to three months for development. `--quick` without `--max-hours` uses the entire processed period with 70/15/15 chronological splits. The full and quick datasets/checkpoints can be kept separate with `--data-dir` and `--artifacts-dir`; set `NYC_DATA_DIR` and `NYC_ARTIFACTS_DIR` to the same locations when serving.

## Full requested window

```bash
# October 2023 through September 2026, plus official zone geometry and lookup.
python -m mobility.download
# Alternative: discover and download the latest 36 months listed on the TLC page.
python -m mobility.download --latest-36 --data-dir data/latest

python -m mobility.preprocess --threads 4
python -m mobility.training --config configs/full.json --epochs 10 --patience 3 --device auto
python -m mobility.evaluation --device auto
```

The prescribed split is training October 2023–September 2025, validation October 2025–March 2026, and test April–September 2026. Full mode checks that all 36 target and observation files exist. The downloader resumes partial HTTP transfers when supported, verifies Parquet metadata, ZIP integrity, and the lookup CSV, and records status, URL, size, and timestamps in `data/raw/manifest.json`. Failed or unavailable downloads are reported and cause a nonzero exit; they are never silently replaced. Rerun the command to retry. Corrupt completed downloads are quarantined alongside the partial files.

`data/processed/validation.json` records accepted and dropped rows per input file. `data/processed/targets/` and `observations/` hold sparse directed Parquet aggregates; `zones.geojson` is transformed to WGS84 with names, boroughs, and projected-CRS centroids. The raw files and artifacts are ignored by Git.

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

## API and deployment

FastAPI exposes `/api/health`, `/api/zones`, `/api/metrics`, `/api/available-times`, `/api/forecast`, `/api/actuals`, `/api/trends`, `/api/zone/{zone_id}/forecast`, and `/api/route`. Forecast timestamps require an explicit UTC offset and must be hour-aligned. The API scores all pairs and supports `pickup_zone`, `top_k`, and `min_trips`. Historical actuals come from processed target aggregates; predictions always come from the checkpoint. Forecasts beyond the last processed data are labeled and use stale available history.

For a simple single-host deployment, build the frontend and run the API from the project root; FastAPI serves `frontend/dist` at `/` and the API under `/api`:

```bash
cd frontend && npm run build && cd ..
NYC_DATA_DIR=data NYC_ARTIFACTS_DIR=artifacts uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Place a TLS-capable reverse proxy and appropriate access controls in front of the service for network use. Keep the processed data and model artifacts on persistent storage. This demo has no authentication or distributed job storage. Its bounded background queue and replay cache belong to one server process; cold replay of long histories can consume substantial CPU and memory. Avoid exposing it directly to untrusted traffic.

## Tests

```bash
pytest -q
cd frontend && npm test && npm run build
```

The tests use tiny synthetic records solely to check behavior: downloader and manifest handling, geometry conversion, cleaning and directed aggregation, strictly available observations, DST lag handling, negative sampling, forward/loss computation, checkpoint replay, deterministic inference, and API/component contracts. Real model scores must be generated separately from TLC data.
