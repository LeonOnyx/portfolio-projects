# =============================================================
# Stage 1: Builder -- install Python dependencies
# =============================================================
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# =============================================================
# Stage 2: Runtime -- slim image with non-root user
# =============================================================
FROM python:3.11-slim AS runtime

# Non-root user (DEMO-04)
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# curl for entrypoint wait-for-ready check
RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/
COPY data/ data/
COPY docker/ docker/
COPY pyproject.toml .

# Create data directories that the app needs to write to
RUN mkdir -p data/assessments && \
    chmod +x docker/entrypoint.sh && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

ENTRYPOINT ["docker/entrypoint.sh"]
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
