"""Batch NFT generation.

    from nftgen import load_nft_config, generate_collection

    config = load_nft_config("collection.json")
    tokens = generate_collection(config)

Each token is rendered by :mod:`fractalgen` (never the other way around), then
composed on the background and written as ``<index>.png`` plus a metadata JSON
document that describes the traits that were actually used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fractalgen import build_colormap, render_image
from fractalgen.render import load, save

from .compose import place, prepare_canvas
from .config import NftConfig

__all__ = ["GeneratedToken", "generate_collection", "generate_token"]


@dataclass
class GeneratedToken:
    index: int
    image_path: Path
    metadata_path: Path | None
    metadata: dict[str, Any]


def _token_seed(config: NftConfig, index: int) -> int:
    base = config.collection.seed if config.collection.seed is not None else 0
    return base + index


def _traits(config: NftConfig, index: int) -> list[dict[str, Any]]:
    fractal = config.fractal
    traits = [
        {"trait_type": "Method", "value": fractal.method},
        {"trait_type": "Variant", "value": fractal.variant},
        {"trait_type": "Iterations", "value": fractal.iterations},
        {"trait_type": "Polynomial", "value": fractal.polynomial.mode},
        {"trait_type": "Palette", "value": config.colour.name if config.colour.mode != "colors" else "custom"},
    ]
    if fractal.polynomial.mode == "random":
        traits.append({"trait_type": "Max degree", "value": fractal.polynomial.max_degree})
    if fractal.variant == "nova":
        traits.append({"trait_type": "Nova", "value": True})
    traits.append({"trait_type": "Seed", "value": _token_seed(config, index), "display_type": "number"})
    return traits


def generate_token(config: NftConfig, index: int, directory: Path) -> GeneratedToken:
    """Render and write a single token."""
    seed = _token_seed(config, index)
    render_config = config.token_config(index, seed)
    rendered = render_image(render_config)
    cmap = build_colormap(config.colour, config.base_dir)

    background_path = config.resolve(config.composition.background)
    background = load(background_path) if background_path is not None else None
    canvas_size = config.fractal.size
    canvas = prepare_canvas(background, canvas_size)

    rng = np.random.default_rng(seed)
    scale = rng.uniform(*config.composition.scale)
    rotation = rng.uniform(*config.composition.rotation)
    flip = bool(config.composition.flip and rng.random() < 0.5)
    composed = place(
        canvas,
        rendered.rgb,
        scale=scale,
        rotation=rotation,
        flip=flip,
        offset=config.composition.offset,
    )

    image_path = directory / f"{index}.png"
    save(composed, image_path)

    metadata: dict[str, Any] = {
        "name": f"{config.collection.name} #{index}",
        "description": config.collection.description,
        "image": image_path.name,
        "attributes": _traits(config, index),
        "render": {
            "method": render_config.fractal.method,
            "variant": render_config.fractal.variant,
            "seed": seed,
            "scale": round(scale, 4),
            "rotation": round(rotation, 2),
            "flip": flip,
            "palette": cmap.name if hasattr(cmap, "name") else config.colour.name,
        },
    }

    metadata_path: Path | None = None
    if config.write_metadata:
        metadata_path = directory / f"{index}.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return GeneratedToken(index=index, image_path=image_path, metadata_path=metadata_path, metadata=metadata)


def generate_collection(config: NftConfig, *, overwrite: bool = False) -> list[GeneratedToken]:
    """Generate every token of the collection and a ``collection.json`` manifest."""
    directory = config.resolve_output()
    directory.mkdir(parents=True, exist_ok=True)

    tokens: list[GeneratedToken] = []
    for offset in range(config.collection.size):
        index = config.collection.start_index + offset
        image_path = directory / f"{index}.png"
        if image_path.exists() and not overwrite:
            tokens.append(
                GeneratedToken(
                    index=index,
                    image_path=image_path,
                    metadata_path=(directory / f"{index}.json") if config.write_metadata else None,
                    metadata={},
                )
            )
            continue
        tokens.append(generate_token(config, index, directory))

    manifest = {
        "name": config.collection.name,
        "description": config.collection.description,
        "size": config.collection.size,
        "start_index": config.collection.start_index,
        "seed": config.collection.seed,
        "tokens": [
            {"index": token.index, "image": token.image_path.name, "metadata": None if token.metadata_path is None else token.metadata_path.name}
            for token in tokens
        ],
    }
    (directory / "collection.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return tokens
