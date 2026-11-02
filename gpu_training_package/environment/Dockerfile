FROM nvcr.io/nvidia/pytorch:25.11-py3@sha256:417cbf33f87b5378849df37983552cd1f8bc8b62fe1ceabe004de816a55dff21

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=42 \
    NYC_DATA_DIR=/app/data NYC_DATA_MANIFEST=/app/dataset_manifest.json \
    NYC_ARTIFACTS_DIR=/app/artifacts NYC_REPORTS_DIR=/app/reports
WORKDIR /app

COPY requirements.txt /tmp/nyc-requirements.txt
# Do not replace NVIDIA's GB10-compatible CUDA PyTorch with a CPU PyPI wheel.
RUN python3 -c 'import torch; print(torch.__version__)' > /tmp/nvidia-torch-version \
    && python3 -m pip install --no-cache-dir --no-deps -r /tmp/nyc-requirements.txt \
    && python3 -c 'from pathlib import Path; import torch, torch_geometric, duckdb, geopandas, pyarrow; assert torch.__version__ == Path("/tmp/nvidia-torch-version").read_text().strip(); assert torch.backends.cuda.is_built(); print("CUDA PyTorch", torch.__version__, "PyG", torch_geometric.__version__)'

COPY verify_data.py /opt/nyc-taxi/verify_data.py
COPY task_inputs/data/ /app/data/
COPY task_inputs/dataset_manifest.json /app/dataset_manifest.json
RUN python3 /opt/nyc-taxi/verify_data.py check --data /app/data --manifest /app/dataset_manifest.json \
    && mkdir -p /app/artifacts /app/reports
