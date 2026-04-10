#!/usr/bin/env python
"""CLI script to create (or reset) Weaviate collections for the RAG pipeline.

Creates all four collections defined in ``src.rag.schemas`` with the
correct property schemas and self-provided vector configuration.

Usage:
    python scripts/setup_weaviate.py           # create collections
    python scripts/setup_weaviate.py --reset   # delete then recreate

Requires Weaviate to be running (``docker-compose up -d``).
"""

from __future__ import annotations

import argparse
import os
import sys

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so ``from src.…`` imports work
# regardless of the working directory when invoked.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import weaviate  # noqa: E402

from src.rag.schemas import COLLECTION_NAMES, create_all_collections  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create or reset Weaviate collections for the RAG pipeline.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Delete existing collections before creating (for dev workflow).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    host = os.environ.get("WEAVIATE_HOST", "localhost")
    port = int(os.environ.get("WEAVIATE_HTTP_PORT", "8080"))
    grpc_port = int(os.environ.get("WEAVIATE_GRPC_PORT", "50051"))

    try:
        with weaviate.connect_to_local(host=host, port=port, grpc_port=grpc_port) as client:
            if args.reset:
                print("Resetting collections...")
                for name in COLLECTION_NAMES:
                    if client.collections.exists(name):
                        client.collections.delete(name)
                        print(f"  Deleted: {name}")
                    else:
                        print(f"  Skipped (not found): {name}")

            # Check which collections already exist (idempotent creation)
            existing = [name for name in COLLECTION_NAMES if client.collections.exists(name)]
            missing = [name for name in COLLECTION_NAMES if name not in existing]

            if existing and not args.reset:
                print(f"Collections already exist (skipping): {existing}")

            if missing or args.reset:
                print("Creating collections...")
                try:
                    create_all_collections(client)
                except Exception as exc:
                    # Collections may partially exist; log and continue
                    print(f"  Note: {exc} (some collections may already exist)")

            # Verify each collection exists
            verified = []
            for name in COLLECTION_NAMES:
                if client.collections.exists(name):
                    collection = client.collections.get(name)
                    config = collection.config.get()
                    prop_count = len(config.properties)
                    print(f"  {name}: {prop_count} properties")
                    verified.append(name)
                else:
                    print(f"  WARNING: {name} not found after creation")

            print(f"\nVerified {len(verified)} collections: {verified}")

    except weaviate.exceptions.WeaviateConnectionError:
        print(
            "ERROR: Could not connect to Weaviate.\n"
            "Is Weaviate running? Start it with:\n"
            "  docker-compose up -d",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
