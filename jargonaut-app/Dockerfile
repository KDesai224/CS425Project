# Image for the Jargonaut app: the full pipeline behind a small web UI.
# Everything up to the torch install matches Grpproj/Dockerfile line for line, so
# Docker reuses that image's cached torch layer instead of downloading it again.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    HF_HOME=/workspace/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /workspace

RUN pip install torch==2.14.1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

# Triton, which PyTorch's native ops use inside Gemma 4's attention, compiles a
# small CUDA helper on first use and needs a C compiler. The slim base has none.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# The code is bind-mounted at /workspace by docker-compose.yml, not copied in,
# so editing it never needs a rebuild.
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
