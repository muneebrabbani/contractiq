# ContractIQ -- serving image for OpenShift.
#
# Bakes in every model/dependency that would otherwise need internet access
# at runtime, because the deployment target only permits egress to
# api.openai.com once the pod is running:
#   - pip dependencies (from pyproject.toml, via PyPI)
#   - the spaCy en_core_web_sm model (redaction's address-NER step)
#   - the cross-encoder/ms-marco-MiniLM-L-6-v2 reranker (huggingface.co)
# Infra confirmed this cluster has direct internet egress (no forward proxy)
# at both build and run time, so none of the three downloads below need
# proxy configuration.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Tesseract OCR: not a pip package, required for the scanned-page fallback
# in ingestion/loaders.py (~85% of this corpus's pages, per ARCHITECTURE.md).
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifests first so the (slow) install layer is cached
# independently of application source-code changes.
COPY pyproject.toml ./
COPY src ./src

# Installs from pyproject.toml (not requirements.txt -- the latter is missing
# `streamlit` and a few others; pyproject.toml is the complete, current list)
# as a real package, so no PYTHONPATH hack is needed for `import contractiq`.
#
# torch is installed first from PyTorch's CPU-only index. Left to the default
# PyPI resolution, sentence-transformers pulls the CUDA build of torch plus
# ~2GB of nvidia-* wheels that this cluster has no GPU to use.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir .

ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Bake the two internet-downloaded models into the image so the running
# container never needs to reach spaCy's model host or huggingface.co.
RUN python -m spacy download en_core_web_sm
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Copy the rest of the repo (README, ARCHITECTURE.md, etc.) needed at runtime
# for reference/consistency; data/ and .env are excluded via .dockerignore.
COPY . .

# OpenShift's restricted-v2 SCC runs the container as an arbitrary UID from
# the namespace's allocated range, always in group 0 -- never hardcode a
# fixed non-root USER here. Instead make /app group-writable so whatever UID
# OpenShift assigns can still write into it (e.g. mounted PVC subpaths).
RUN chgrp -R 0 /app && chmod -R g=u,g+w /app

EXPOSE 8501

# --server.enableCORS=false / --server.enableXsrfProtection=false: the
# standard fix for Streamlit's WebSocket handshake otherwise failing behind
# a TLS-terminating reverse proxy (the OpenShift Route).
CMD ["streamlit", "run", "src/contractiq/ui/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
