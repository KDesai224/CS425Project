# Image for the DeBERTa evidence-extractor training notebook.
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

# The notebook and data/ are bind-mounted at /workspace by docker-compose.yml,
# not copied in, so editing the notebook never needs a rebuild.
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
