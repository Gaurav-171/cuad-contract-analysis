#!/usr/bin/env python3
"""Run the full CUAD contract-analysis pipeline.

Usage:
    python run_pipeline.py                # 50 contracts (default)
    python run_pipeline.py -n 5           # smaller run for a quick test
    python run_pipeline.py --no-index     # skip the semantic-search index
"""

import argparse

from src import pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="CUAD contract analysis with LLMs")
    parser.add_argument("-n", "--num-contracts", type=int, default=None,
                        help="number of contracts to process (default: 50)")
    parser.add_argument("--no-index", action="store_true",
                        help="skip building the semantic-search embedding index")
    args = parser.parse_args()
    pipeline.run(num_contracts=args.num_contracts,
                 build_search_index=not args.no_index)


if __name__ == "__main__":
    main()
