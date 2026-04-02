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

    try:
        with weaviate.connect_to_local() as client:
            if args.reset:
                print("Resetting collections...")
                for name in COLLECTION_NAMES:
                    if client.collections.exists(name):
                        client.collections.delete(name)
                        print(f"  Deleted: {name}")
                    else:
                        print(f"  Skipped (not found): {name}")

            print("Creating collections...")
            create_all_collections(client)

            # Verify each collection was created
            created = []
            for name in COLLECTION_NAMES:
                collection = client.collections.get(name)
                config = collection.config.get()
                prop_count = len(config.properties)
                print(f"  {name}: {prop_count} properties")
                created.append(name)

            print(f"\nCreated {len(created)} collections: {created}")

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
