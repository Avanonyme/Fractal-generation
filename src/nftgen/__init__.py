"""NFT collection generation built on top of :mod:`fractalgen`.

This package is deliberately separate: ``fractalgen`` renders fractals, this
package composes them into a numbered collection with metadata. Nothing in
``fractalgen`` imports ``nftgen``.
"""

from __future__ import annotations

from .config import CollectionConfig, CompositionConfig, NftConfig, load_nft_config
from .generator import GeneratedToken, generate_collection, generate_token

__all__ = [
    "CollectionConfig",
    "CompositionConfig",
    "GeneratedToken",
    "NftConfig",
    "generate_collection",
    "generate_token",
    "load_nft_config",
]
