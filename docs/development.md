# Development and API guide

[← Project overview](../README.md) · [Training](training.md) · [Data & model](data-and-model.md)

Run commands from the repository root with the Python virtual environment active.

## Local development

Run the API and Vite in separate terminals:

```bash
# Terminal 1
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

# Terminal 2
npm --prefix frontend run dev
```

Open [localhost:5173](http://localhost:5173). Vite proxies `/api` to port 8000.
Use `NYC_DATA_DIR` and `NYC_ARTIFACTS_DIR` to select a matching dataset and
checkpoint. The default is `data/` + `artifacts/cuda-full/` when that checkpoint
exists, otherwise `data/` + `artifacts/`. Model selection in the UI applies to
the running process; restart restores the environment/default selection.
The simulator’s fleet map works offline; forecast basemap tiles and fonts
use external services.

## Simulation and training jobs

| Endpoint | Purpose |
| --- | --- |
| `GET /api/models` | List available checkpoint profiles |
| `POST /api/models/select` | Select the active profile |
| `GET /api/training` | Read losses, progress, logs, and jobs |
| `POST /api/training` | Start training followed by evaluation |
| `POST /api/simulations` | Compare both policies on one seeded request stream |
| `GET /api/jobs/{id}` | Poll a submitted job |

Submissions return **202** with a job ID. Poll until `complete` or `failed`.
A second active job of the same kind returns **409**; invalid settings return
**422**. The process retains up to eight jobs. Jobs reset on server restart;
training checkpoints remain on disk. Use one Uvicorn worker.

## Command-line simulation

```bash
python scripts/simulate.py \
  --as-of 2026-04-01T08:00:00-04:00 \
  --hours 2 --fleet-size 80 \
  --output /tmp/fleet-replay.json
```

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Dataset/checkpoint missing after cloning | Run `git lfs pull`; LFS pointer files are not usable datasets |
| First full-model forecast is slow | Initial temporal replay can take several minutes; later requests reuse state |
| Want a faster smoke test | Start `python scripts/serve.py --demo` |
| CUDA unavailable | Install a compatible CUDA-enabled PyTorch build; see the [training guide](training.md) |
| Resume rejected | Keep the dataset/settings unchanged and increase the total epoch count |
| Simulation window rejected | Every hour needs history; use at most six hours and 50,000 modeled requests |

The first replay saves `replay-cpu.pt` alongside its checkpoint. Consecutive
hours reuse temporal state; checkpoint or observation changes invalidate it.
Earlier cutoffs rebuild without carrying future memory backward in time.
Playback reuses static map geometry and hour options, caches native formatters,
and pauses while its workspace or browser tab is hidden.

## API and deployment

FastAPI exposes `/api/health`, `/api/zones`, `/api/metrics`, `/api/available-times`, `/api/forecast`, `/api/actuals`, `/api/trends`, `/api/zone/{zone_id}/forecast`, and `/api/route`. Forecast timestamps require an explicit UTC offset and must be hour-aligned. The API scores all pairs and supports `pickup_zone`, `top_k`, and `min_trips`. Historical actuals come from processed target aggregates; predictions always come from the checkpoint. Forecasts beyond the last processed data are labeled and use stale available history.

For a simple single-host deployment, build the frontend and run the API from the project root; FastAPI serves `frontend/dist` at `/` and the API under `/api`:

```bash
npm --prefix frontend run build
NYC_DATA_DIR=data NYC_ARTIFACTS_DIR=artifacts/cuda-full uvicorn backend.app.main:app --host 0.0.0.0 --port 8000
```

Place a TLS-capable reverse proxy and appropriate access controls in front of the service for network use. Keep the processed data and model artifacts on persistent storage. This demo has no authentication or distributed job storage. Its bounded background queue and replay cache belong to one server process; cold replay of long histories can consume substantial CPU and memory. Avoid exposing it directly to untrusted traffic.

## Tests

```bash
pytest -q
npm --prefix frontend test
npm --prefix frontend run build
```

The tests use tiny synthetic records solely to check behavior: downloader and manifest handling, geometry conversion, cleaning and directed aggregation, strictly available observations, DST lag handling, negative sampling, forward/loss computation, checkpoint replay, deterministic inference, and API/component contracts. Real model scores must be generated separately from TLC data.
