FROM python:3.12-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
 && rm -rf /var/lib/apt/lists/*

# install uv
RUN pip install --no-cache-dir uv

# copy dependency files first (for cache)
COPY pyproject.toml uv.lock* requirements.txt* ./

ENV UV_PROJECT_ENVIRONMENT=/opt/venv

ENV UV_PYTHON=python3.12

# install python deps via uv
RUN uv sync --no-dev

# install additional requirements.txt if present
RUN if [ -f requirements.txt ]; then uv pip install --python /opt/venv/bin/python -r requirements.txt; fi

# copy source code
COPY . .

# env vars
ENV GRAPH_DIR=./data/graphs_new
ENV CHROMA_DIR=./chroma_db
ENV DEVICE=CPU
ENV TOP_N=3