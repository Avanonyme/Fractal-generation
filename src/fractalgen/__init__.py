"""Root-finding fractal generation.

The library is split so the numeric core can be used on its own::

    import numpy as np
    from fractalgen import FractalConfig, Polynomial, solve, colorize, build_colormap

    config = FractalConfig(size=1024)
    image = solve(config, Polynomial.from_roots([1, -1, 1j, -1j]))
    rgb = colorize(image.scalar_field("iteration"), build_colormap())

Rendering helpers (:func:`fractalgen.pipeline.render_image`,
:func:`fractalgen.pipeline.render_video`) sit on top of that core and know
nothing about NFTs; ``nftgen`` builds on those.
"""

from __future__ import annotations

from .colour import build_colormap
from .config import (
    AnimationConfig,
    ColourConfig,
    Config,
    ConfigError,
    FractalConfig,
    ImageConfig,
    OrbitTrapConfig,
    PolynomialConfig,
    RuntimeConfig,
    ShadingConfig,
    VideoConfig,
    load_config,
    save_config,
)
from .orbit import OrbitTrap
from .pipeline import RenderedImage, build_frame, render_image, render_video
from .polynomial import Polynomial
from .shading import shade
from .solver import FractalImage, choose_polynomial, pixel_grid, solve

__version__ = "2.0.0"

__all__ = [
    "__version__",
    "AnimationConfig",
    "ColourConfig",
    "Config",
    "ConfigError",
    "FractalConfig",
    "FractalImage",
    "ImageConfig",
    "OrbitTrap",
    "OrbitTrapConfig",
    "Polynomial",
    "RenderedImage",
    "PolynomialConfig",
    "RuntimeConfig",
    "ShadingConfig",
    "VideoConfig",
    "build_colormap",
    "build_frame",
    "choose_polynomial",
    "load_config",
    "pixel_grid",
    "render_image",
    "render_video",
    "save_config",
    "shade",
    "solve",
]
