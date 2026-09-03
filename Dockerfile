# syntax=docker/dockerfile:1
#
# QKash container image.
#
# The audited RPW extract is baked in at build time from the local build
# context: data/**/*.csv is untracked by design (see AGENTS.md), so this image
# can only be built on a machine that already holds the dataset.

########################  builder  ########################
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential covers any dependency without a cp312 manylinux wheel.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

########################  runtime  ########################
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8503 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 qkash

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# The project .streamlit/config.toml advertises the app at host "qkash" on
# 8503; that alias does not resolve off this workstation, so it is deliberately
# excluded from the image and replaced by the STREAMLIT_* environment above.
COPY --chown=qkash:qkash app.py ./
COPY --chown=qkash:qkash qkash/ ./qkash/
COPY --chown=qkash:qkash data/QKash.png ./data/
COPY --chown=qkash:qkash data/processed/remittance_east_africa_clean.csv ./data/processed/

USER qkash
EXPOSE 8503

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT:-$STREAMLIT_SERVER_PORT}/_stcore/health" || exit 1

# Render injects $PORT; locally it is unset and 8503 applies.
CMD ["sh", "-c", "exec streamlit run app.py --server.port ${PORT:-8503} --server.address 0.0.0.0"]
