from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from hmumu_transformer.data.shards import prepare_token_shards
from hmumu_transformer.preflight import check_dependencies, check_files_exist
from hmumu_transformer.utils.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to configs/experiment.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild shards even if metadata matches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_dependencies()
    cfg = load_experiment_config(args.config)
    files = []
    files += cfg["data"]["ggH_files"]
    files += cfg["data"]["VBF_files"]
    files += cfg["data"]["DY_files"]
    check_files_exist(files)

    prepared = prepare_token_shards(cfg, force_rebuild=args.force)
    print(json.dumps({"metadata_path": prepared.metadata_path, "splits": prepared.metadata["splits"]}, indent=2))


if __name__ == "__main__":
    main()
