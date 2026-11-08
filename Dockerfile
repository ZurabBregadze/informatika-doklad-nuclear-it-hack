FROM python:3.11-slim-bookworm

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
# CPU by default. A CUDA image is built with --build-arg TORCH_INDEX_URL=.../cu126.
RUN python -m pip install --no-cache-dir torch==2.14.1 --index-url "${TORCH_INDEX_URL}" \
    && python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check

COPY src/doe_p0/__init__.py src/doe_p0/fresnel.py src/doe_p0/model.py \
     src/doe_p0/phase.py src/doe_p0/weights.py src/doe_p0/inference.py ./src/doe_p0/
COPY infer.py ./
COPY weights/ ./weights/
RUN python -c "import hashlib,json; from pathlib import Path; from doe_p0.weights import load_weights; p=Path('weights/model.pt'); m=json.loads(p.with_suffix('.json').read_text()); assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256']; load_weights(p); print('BUNDLED MODEL OK')"

ENTRYPOINT ["python", "/app/infer.py"]
CMD ["--input-dir", "/input", "--output", "/output/doe_phase.png"]
