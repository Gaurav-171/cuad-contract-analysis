#!/usr/bin/env python3
"""Bonus: semantic search over extracted clauses.

Usage:
    python search_clauses.py "what happens if a party breaches the contract"
    python search_clauses.py "cap on damages" -k 3
"""

import argparse
import json

from src import config, semantic_search


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic search over extracted clauses")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    if not config.EMBEDDINGS_JSON.exists():
        raise SystemExit("No embedding index found. Run `python run_pipeline.py` first.")

    for hit in semantic_search.search(args.query, args.top_k):
        print(json.dumps(hit, indent=2))


if __name__ == "__main__":
    main()
