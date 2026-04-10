#!/bin/bash
set -e

WEAVIATE_HOST="${WEAVIATE_HOST:-weaviate}"
WEAVIATE_PORT="${WEAVIATE_HTTP_PORT:-8080}"

echo "[entrypoint] Waiting for Weaviate at ${WEAVIATE_HOST}:${WEAVIATE_PORT}..."
MAX_WAIT=60
WAITED=0
until curl -sf "http://${WEAVIATE_HOST}:${WEAVIATE_PORT}/v1/.well-known/ready" > /dev/null 2>&1; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "[entrypoint] ERROR: Weaviate not ready after ${MAX_WAIT}s"
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
done
echo "[entrypoint] Weaviate ready (${WAITED}s)."

# Create collections (idempotent -- skips if they already exist)
echo "[entrypoint] Setting up Weaviate collections..."
python scripts/setup_weaviate.py 2>&1 || echo "[entrypoint] WARNING: Collection setup had errors (may already exist)"

echo "[entrypoint] Starting application..."
exec "$@"
