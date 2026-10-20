"""Command line interface for NFT collection generation.

    nftgen generate collection.json --overwrite
    nftgen example --form collection
"""

from __future__ import annotations

import argparse
import json
import sys

from fractalgen.config import ConfigError

from .config import NftConfig, load_nft_config
from .generator import generate_collection

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nftgen", description="Generate an NFT collection from fractal renders.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="render the whole collection")
    generate.add_argument("config", help="collection JSON config")
    generate.add_argument("--overwrite", action="store_true", help="re-render existing tokens")
    generate.add_argument("--verbose", action="store_true")

    example = subparsers.add_parser("example", help="print an example collection config")
    example.add_argument("--indent", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "example":
            print(json.dumps(example_config(), indent=arguments.indent))
            return 0

        config = load_nft_config(arguments.config)
        tokens = generate_collection(config, overwrite=arguments.overwrite)
        if arguments.verbose:
            for token in tokens:
                print(f"{token.image_path.name}\t{token.metadata.get('render', {})}")
        print(f"wrote {len(tokens)} token(s) to {config.resolve_output()}")
        return 0
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"missing file: {error}", file=sys.stderr)
        return 2


def example_config() -> dict:
    return {
        "collection": {
            "name": "Fractals",
            "description": "Root-finding fractals, procedurally generated.",
            "size": 5,
            "start_index": 1,
            "seed": 2024,
        },
        "output": {"directory": "nft", "metadata": True},
        "composition": {
            "background": None,
            "scale": [0.45, 0.75],
            "rotation": [-12, 12],
            "flip": True,
            "offset": [0.0, 0.0],
        },
        "fractal": {
            "size": 800,
            "method": "newton",
            "variant": "standard",
            "iterations": 30,
            "polynomial": {"mode": "random", "max_degree": 7},
            "orbit_trap": {"image": None, "distance_mode": 1},
        },
        "colour": {"mode": "matplotlib", "name": "magma"},
        "image": {"return_type": "iteration", "shading": {"type": "matplotlib"}},
    }


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
