"""Colour handling: colormaps from names, explicit colours, images and accents.

Only matplotlib is required. The image-palette and perceptual-uniformity helpers
need the ``palette`` extra (scikit-image, seaborn, extcolors, colorspacious) and
import it lazily, so the core renderer never pays for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, ListedColormap, to_hex, to_rgb, to_rgba

from . import optional
from ._numeric import as_float, as_int
from .config import ColourConfig, ConfigError

__all__ = [
    "build_colormap",
    "hex_to_rgb",
    "rgb_to_hex",
    "complementary",
    "triadic",
    "analogous",
    "split_complementary",
    "tetradic",
    "shades",
    "blend",
    "sample_colormap",
]

_MATPLOTLIB_FALLBACK = "viridis"


# --------------------------------------------------------------------------- #
# basic conversions
# --------------------------------------------------------------------------- #
def rgb_to_hex(rgb: Sequence[float]) -> str:
    """``(r, g, b)`` in 0..255 -> ``'#RRGGBB'``."""
    red, green, blue = (round(as_float(channel, "colour")) for channel in rgb[:3])
    return f"#{red:02X}{green:02X}{blue:02X}"


def hex_to_rgb(colour: str) -> tuple[int, int, int]:
    red, green, blue = (round(channel * 255) for channel in to_rgb(colour))
    return red, green, blue


def blend(first: Sequence[float], second: Sequence[float], alpha: float) -> tuple[int, int, int]:
    """Linear blend between two 0..255 colours."""
    return tuple(  # type: ignore[return-value]
        round((1 - alpha) * first[index] + alpha * second[index]) for index in range(3)
    )


def _to_rgb255(colour: str | Sequence[float]) -> tuple[int, int, int]:
    if isinstance(colour, str):
        return hex_to_rgb(colour)
    return tuple(round(as_float(channel, "colour")) for channel in colour[:3])  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# colour-scheme transforms (HSV based, pure stdlib math)
# --------------------------------------------------------------------------- #
def _shift_hue(colour: str | Sequence[float], degrees: float) -> str:
    import colorsys

    red, green, blue = (channel / 255 for channel in _to_rgb255(colour))
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    shifted = colorsys.hsv_to_rgb((hue + degrees / 360) % 1, saturation, value)
    return rgb_to_hex([channel * 255 for channel in shifted])


def complementary(colours: Sequence[str | Sequence[float]]) -> list[str]:
    return [rgb_to_hex([255 - channel for channel in _to_rgb255(colour)]) for colour in colours]


def triadic(colours: Sequence[str | Sequence[float]]) -> tuple[list[str], list[str]]:
    return [ _shift_hue(colour, 120) for colour in colours], [_shift_hue(colour, 240) for colour in colours]


def analogous(colours: Sequence[str | Sequence[float]], shift: float = 30.0) -> list[str]:
    return [_shift_hue(colour, shift) for colour in colours]


def split_complementary(
    colours: Sequence[str | Sequence[float]], shift: float = 30.0
) -> tuple[list[str], list[str]]:
    return (
        [_shift_hue(colour, 180 + shift) for colour in colours],
        [_shift_hue(colour, 180 - shift) for colour in colours],
    )


def tetradic(colours: Sequence[str | Sequence[float]]) -> tuple[list[str], list[str], list[str], list[str]]:
    families = ([_shift_hue(colour, 90 * step) for colour in colours] for step in range(4))
    return tuple(families)  # type: ignore[return-value]


def shades(colours: Sequence[str | Sequence[float]], steps: int = 5) -> list[str]:
    import colorsys

    palette: list[str] = []
    for colour in colours:
        red, green, blue = (channel / 255 for channel in _to_rgb255(colour))
        hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
        for step in range(steps + 1):
            scaled = colorsys.hsv_to_rgb(hue, saturation, value * step / steps)
            palette.append(rgb_to_hex([channel * 255 for channel in scaled]))
    return palette


# --------------------------------------------------------------------------- #
# palettes
# --------------------------------------------------------------------------- #
def _interpolate(colours: Sequence[tuple[int, int, int]], steps: int) -> list[str]:
    """Evenly interpolate a short colour list to ``steps`` colours."""
    if len(colours) < 2:
        return [rgb_to_hex(colours[0])] * steps
    palette: list[str] = []
    for index in range(steps):
        position = index / (steps - 1) * (len(colours) - 1)
        left = min(int(position), len(colours) - 2)
        fraction = position - left
        palette.append(rgb_to_hex(blend(colours[left], colours[left + 1], fraction)))
    return palette


def _palette_from_image(path: Path, n_colors: int) -> list[str]:
    optional.require("palette")
    import extcolors
    from PIL import Image

    with Image.open(path) as image:
        colours, _ = extcolors.extract_from_image(image.convert("RGB"), limit=n_colors)
    dominant = [tuple(colour[0]) for colour in colours][: max(2, n_colors)]
    return _interpolate(dominant, n_colors)


def _accent_palette(base: list[str], accent: str, n_colors: int) -> list[str]:
    """Extend ``base`` with blended accents of the requested colour scheme."""

    accenters = {
        "complementary": lambda colours: [complementary(colours)],
        "triadic": triadic,
        "analogous": analogous,
        "split-complementary": split_complementary,
        "tetradic": tetradic,
        "shades": shades,
    }
    if accent not in accenters:
        raise ConfigError(f"colour.accent: unknown scheme {accent!r}")
    produced = accenters[accent](base)
    palette = list(base)
    for family in produced:
        for index, colour in enumerate(family):
            partner = base[index % len(base)]
            palette.append(rgb_to_hex(blend(_to_rgb255(partner), _to_rgb255(colour), index / max(1, len(family) - 1))))
    return _interpolate([_to_rgb255(colour) for colour in palette], n_colors)


def sample_colormap(cmap: Colormap, count: int = 8) -> list[str]:
    """Sample a colormap into ``count`` hex colours."""
    return [to_hex(cmap(position)) for position in np.linspace(0, 1, count)]


def build_colormap(config: ColourConfig | None = None, base_dir: Path | str | None = None) -> Colormap:
    """Resolve a :class:`ColourConfig` into a matplotlib colormap object.

    Both arguments are optional so the numeric core stays usable without any
    configuration: ``build_colormap()`` returns ``viridis``.
    """
    config = config or ColourConfig()
    base_dir = Path(base_dir) if base_dir is not None else Path(".")
    from matplotlib import colormaps

    if config.mode == "matplotlib":
        if config.name not in colormaps:
            raise ConfigError(f"colour.name: unknown matplotlib colormap {config.name!r}")
        return colormaps[config.name]

    if config.mode == "seaborn":
        optional.require("seaborn")
        import seaborn as sns

        try:
            colours = sns.color_palette(config.name).as_hex()
        except ValueError as exc:
            raise ConfigError(f"colour.name: unknown seaborn palette {config.name!r}") from exc
        return _listed(colours, config)

    if config.mode == "colors":
        return _listed(list(config.colors), config)

    # accents
    if config.image is None:
        raise ConfigError("colour.image: required when mode = 'accents'")
    path = config.image if config.image.is_absolute() else base_dir / config.image
    if not path.exists():
        raise ConfigError(f"colour.image: file not found: {path}")
    base = _palette_from_image(path, max(4, config.n_colors // 16))
    accents = [_rgba(colour) for colour in _accent_palette(base, config.accent, config.n_colors)]
    return LinearSegmentedColormap.from_list("fractal-accents", accents, N=config.n_colors)


def _rgba(colour: str | Sequence[float]) -> tuple[float, ...]:
    """Normalise a colour to something matplotlib's ``to_rgba`` accepts."""
    if isinstance(colour, str):
        return to_rgba(colour)
    channels = [as_float(channel, "colour") for channel in colour[:4]]
    if len(channels) == 3:
        return (channels[0], channels[1], channels[2], 1.0)
    return (channels[0], channels[1], channels[2], channels[3])


def _listed(colours: Sequence[str | Sequence[float]], config: ColourConfig) -> Colormap:
    entries = [_rgba(colour) for colour in colours]
    if config.add_black:
        entries.insert(0, (0.0, 0.0, 0.0, 1.0))
    if len(entries) == 1:
        return ListedColormap(entries * 2, name="fractal-listed")
    return LinearSegmentedColormap.from_list("fractal-listed", entries, N=config.n_colors)
