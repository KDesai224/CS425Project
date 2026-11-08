# Training/evaluation image for the Jargonaut simplifier (Stage 3).
# The PyPI torch wheel bundles its own CUDA libraries, so a slim Python base is
# enough: the host only needs the NVIDIA Windows driver and Docker's WSL 2 backend.
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
# Kept after the pip layers so a rebuild reuses the cached torch install.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Code is bind-mounted at /workspace by docker-compose.yml, not copied in,
# so editing a script never needs a rebuild.
CMD ["bash"]
