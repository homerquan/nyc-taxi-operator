# NYC taxi temporal graph training

This package trains the PyTorch Geometric Temporal Graph Network developed in
the NYC taxi demo, using the bundled NYC TLC yellow taxi data. It follows the
same `solution/`, `environment/`, and `tests/` layout as `boardgame-translearn`.

`environment/task_inputs/data/` contains processed hourly targets and
completed-trip observations for all 36 months from October 2023 through
September 2026, plus zone assets and the preprocessing validation report:

```text
data/
  observations/       # 36 monthly Parquet files
  targets/            # 36 monthly Parquet files
  validation.json
  zones.geojson
  zones.parquet
```

`dataset_manifest.json` records the size and SHA-256 of all 75 included files.
Raw downloads and the separate quick-demo dataset are not bundled. The
`--quick` option uses a shorter time split over these same inputs. TLC trip
records are published by the
[NYC Taxi and Limousine Commission](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page).

## Requirements

- Linux with an NVIDIA GPU, a current NVIDIA driver, Docker Engine, Docker
  Compose, and the NVIDIA Container Toolkit. The container is pinned to the
  ARM64 NVIDIA PyTorch 25.11/CUDA 13 image used on the GB10 Spark.
- Enough storage for the 1 GB input bundle, the NVIDIA image, Docker layers,
  and model and evaluation artifacts. The full evaluation can write roughly
  1.4 GB of prediction files.
- Internet access while building the image to download pinned Python packages.
  Training and evaluation use the bundled inputs and do not download data.

Apple Silicon macOS cannot pass its Apple GPU to a CUDA container. Build and
run this NVIDIA configuration on the Spark or another compatible Linux host.

## Build and check CUDA

From the package root on Spark:

```bash
cd /home/homer/snorkel/nyc-taxi
mkdir -p artifacts reports logs
docker compose build
docker compose run --rm training python3 -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); assert torch.cuda.is_available()'
```

The Dockerfile deliberately keeps the NVIDIA image's CUDA-enabled PyTorch;
installing a plain CPU-only `torch` wheel would disable GB10 training.
Compose defaults to user and group ID 1000, matching the Spark account, so the
output files belong to the host user. On a Linux host with different IDs, set
`NYC_UID` and `NYC_GID` to the results of `id -u` and `id -g` before running.

## Train, validate, and test

```bash
docker compose run --rm training \
  bash /app/solution/solve.sh \
  --epochs 10 --patience 3 --device cuda
```

The pipeline verifies every bundled input, records chronological splits and a
three-hour label embargo, trains on the earliest period, validates each epoch,
saves the lowest-validation-loss checkpoint, and evaluates that checkpoint and
both historical baselines on the complete later test period. `--epochs` is the
maximum number of epochs; `--patience 3` stops after three consecutive epochs
without an improvement in validation loss.

The split is October 2023–September 2025 for training, October 2025–March 2026
for validation, and April–September 2026 for the held-out test. Train and
validation exclude their final three pickup hours from loss calculations so
that trip completion labels are available before the next split.

For a short **smoke test** that does not overwrite full-run artifacts:

```bash
docker compose run --rm training bash /app/solution/solve.sh \
  --quick --max-hours 120 --max-positives 16 --epochs 1 --eval-max-hours 4 \
  --artifacts-dir /app/artifacts/smoke --reports-dir /app/reports/smoke
```

The short run checks functionality only; the full verifier rejects its partial
split and evaluation. For a full run, omit all quick and limit options.
The periods are logical filters over the bundled monthly Parquet files; the
recorded boundaries and counts are in `artifacts/splits.json`.

To resume after an interrupted completed epoch, use the same training options
and artifact directory, increase `--epochs` if needed, and add `--resume`:

```bash
docker compose run --rm training bash /app/solution/solve.sh \
  --epochs 10 --patience 3 --device cuda --resume
```

An interrupted partial epoch is replayed from the last saved epoch. Resume
checks that the dataset split and model configuration match. If patience was
already exhausted, increase `--patience` to continue. Use `--phase prepare`,
`--phase train`, or `--phase evaluate` to run individual stages; evaluation
expects a trained checkpoint in the selected artifact directory.

Monitor `artifacts/progress.json`, `artifacts/evaluation_progress.json`, or
`nvidia-smi -l 2` from another terminal. The saved `model.pt` is the best
validation checkpoint; `training_state.pt` is the most recent resumable state.
`metrics.json`, the three `evaluation_*.parquet` files, `assessment.json`, and
`reports/report.md` contain the held-out results.

## Verify the result

After a **full** training and evaluation:

```bash
docker compose run --rm training bash /tests/test.sh
cat logs/reward.json
```

The reference-compatible harness writes `rubric_result.json`, `result.json`,
`criteria.json`, `reward.json`, and `reward.txt` in `logs/`. You can also run
the requested spelling directly:

```bash
docker compose run --rm training python3 /tests/pass_fall.py
```

`pass_fall.py` exits with status 0 for a pass and 1 for a failing result. The
standard Harbor-style `test.sh` writes the grade to `logs/reward.json` and
`logs/reward.txt`.

The rubric checks file integrity, separate chronological splits, CUDA training,
finite training and validation losses, the best checkpoint, complete held-out
coverage, matched candidate routes, reproduced MRR, and labels matched to the
input data. The pass/fail gate requires the TGN's sampled route MRR to exceed
**both** the recent and seasonal history baselines. A failed gate is reported
honestly; a better model is not guaranteed by training. The verifier recomputes
metrics from saved predictions and validates their input labels, but does not
independently rerun the full model inference.

MRR ranks an observed-active route against up to 20 sampled inactive routes
from the same origin and hour. It does not measure citywide demand accuracy,
fare accuracy, extra trips per driver, or profit. The public TLC data is
historical and does not supply live driver positions or taxi availability.

## Reference and credit

The implementation was informed by the TGN example in the Temporal Graph
Benchmark project: [TGB temporal link prediction example](https://github.com/shenyangHuang/TGB/blob/main/examples/linkproppred/tgbl-wiki/tgn.py?utm_source=chatgpt.com).

```bibtex
@article{huang2023temporal,
  title={Temporal graph benchmark for machine learning on temporal graphs},
  author={Huang, Shenyang and Poursafaei, Farimah and Danovitch, Jacob and Fey, Matthias and Hu, Weihua and Rossi, Emanuele and Leskovec, Jure and Bronstein, Michael and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={Advances in Neural Information Processing Systems},
  year={2023}
}
```
