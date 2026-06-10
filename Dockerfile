# ── Stage 1: dependency resolver ──────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Resolve and install prod dependencies into an isolated venv
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen --no-dev --no-install-project

# ── Stage 2: API runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS api

WORKDIR /app

# Non-root user
RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /app/.venv /app/.venv
COPY src/        src/
COPY api/        api/
COPY models/saved/ models/saved/

# Pre-create reports dir so the app can write figures
RUN mkdir -p reports/figures && chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1   \
    PYTHONUNBUFFERED=1          \
    MODEL_CHECKPOINT=models/saved/efficientnet_b0_best.pt \
    MODEL_DEVICE=cpu

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

# ── Stage 3: Dashboard runtime ─────────────────────────────────────────────
FROM python:3.11-slim AS dashboard

WORKDIR /app

RUN useradd --create-home --shell /bin/bash appuser

COPY --from=builder /app/.venv /app/.venv
COPY src/           src/
COPY api/           api/
COPY dashboard/     dashboard/
COPY models/saved/  models/saved/
COPY data/samples/  data/samples/

RUN mkdir -p reports/figures && chown -R appuser:appuser /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1   \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
