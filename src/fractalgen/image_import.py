"""Turn a source image into a palette plus a raster "map modifier".

Three things come out of one picture:

* a **colour scheme** (dominant colours, dark to light) used as the colormap;
* a **raster** (small luminance bitmap) that the solver samples per pixel to
  modulate the damping of the root finder - the map modifier that makes the
  picture readable inside the fractal;
* a **trap bitmap** (the raster thresholded into a shape) used as an orbit trap,
  which echoes the silhouette as well.

The pipeline is dependency-light on purpose: numpy + Pillow only, both already
required. Palette extraction is a small deterministic k-means (seeded), and the
threshold is Otsu's method computed from a histogram, so this module never needs
scikit-image, seaborn or extcolors.

    from fractalgen import load_config
    from fractalgen.image_import import analyse_image, config_from_image

    analysis = analyse_image("photo.jpg", colours=6)
    analysis.colours            # ['#0b1024', '#33507a', ...]
    config = config_from_image(load_config("base.json"), "photo.jpg")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from ._numeric import as_float, as_int
from .colour import rgb_to_hex
from .config import ColourConfig, Config, ConfigError, MapConfig, OrbitTrapConfig

#: Source formats the CLI scans a folder for.
IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"})

__all__ = [
    "IMAGE_SUFFIXES",
    "ImageAnalysis",
    "analyse_image",
    "write_assets",
    "config_from_image",
    "otsu_threshold",
]


@dataclass(frozen=True)
class ImageAnalysis:
    """Everything derived from one source image."""

    #: Luminance raster, ``uint8`` of shape ``(raster_size, raster_size)``.
    raster: np.ndarray
    #: Dominant colours as ``#RRGGBB``, ordered dark to light.
    colours: tuple[str, ...]
    #: Colormap config built from :attr:`colours`.
    palette: ColourConfig
    #: Shape mask (raster above the Otsu threshold).
    trap_bitmap: np.ndarray
    #: Size of the source image, in pixels.
    source_size: tuple[int, int]

    @property
    def raster_colours(self) -> tuple[str, ...]:
        return self.colours


# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
def _load_rgb(path: Path) -> np.ndarray:
    if not path.exists():
        raise ConfigError(f"image: file not found: {path}")
    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.float64)
    except OSError as error:
        raise ConfigError(f"image: cannot read {path}: {error}") from error


def _dominant_colours(rgb: np.ndarray, count: int, seed: int = 0) -> tuple[str, ...]:
    """Deterministic k-means over a subsample of the pixels, ordered by luminance."""
    pixels = rgb.reshape(-1, 3)
    if pixels.shape[0] > 20000:
        rng = np.random.default_rng(seed)
        pixels = pixels[rng.choice(pixels.shape[0], 20000, replace=False)]

    # k-means++ style seeding on a fixed grid keeps the result reproducible
    # without pulling in scikit-learn.
    unique = np.unique(pixels.astype(np.uint8), axis=0).astype(np.float64)
    if unique.shape[0] <= count:
        centres = unique
    else:
        order = np.argsort(unique[:, 0] * 0.299 + unique[:, 1] * 0.587 + unique[:, 2] * 0.114)
        picks = np.linspace(0, order.size - 1, count)
        centres = unique[order[picks.round().astype(int)]]

    for _ in range(12):
        distances = ((pixels[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(distances, axis=1)
        updated = centres.copy()
        for index in range(centres.shape[0]):
            members = pixels[labels == index]
            if members.size:
                updated[index] = members.mean(axis=0)
        if np.allclose(updated, centres):
            centres = updated
            break
        centres = updated

    luminance = centres @ np.array([0.299, 0.587, 0.114])
    centres = centres[np.argsort(luminance)]
    return tuple(rgb_to_hex(colour) for colour in centres)


# --------------------------------------------------------------------------- #
# raster and shape mask
# --------------------------------------------------------------------------- #
def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold of an 8-bit array (histogram method, numpy only)."""
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ConfigError("image: empty raster")
    histogram, edges = np.histogram(np.clip(array, 0, 255), bins=bins, range=(0, 255))
    centres = (edges[:-1] + edges[1:]) / 2
    total = histogram.sum()
    if total == 0:
        return float(centres[len(centres) // 2])

    counts = np.cumsum(histogram)
    weights = counts / total
    means = np.cumsum(histogram * centres) / np.maximum(counts, 1)
    overall = float((histogram * centres).sum() / total)
    # Otsu's between-class variance: w0 * (mean0 - mean_total)^2 / (1 - w0).
    between = weights * (means - overall) ** 2 / np.maximum(1 - weights, 1e-12)
    return float(centres[int(np.nanargmax(between))])


def _rasterise(rgb: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("L")
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(resized, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def analyse_image(
    path: str | Path,
    *,
    colours: int = 6,
    raster_size: int = 96,
    seed: int = 0,
) -> ImageAnalysis:
    """Extract a palette, a raster and a shape mask from ``path``."""
    source = Path(path).expanduser()
    count = as_int(colours, "image.colours")
    size = as_int(raster_size, "image.raster_size")
    if not 2 <= count <= 64:
        raise ConfigError("image.colours: must be within 2..64")
    if not 8 <= size <= 1024:
        raise ConfigError("image.raster_size: must be within 8..1024")

    rgb = _load_rgb(source)
    raster = _rasterise(rgb, size)
    threshold = otsu_threshold(raster)
    trap_bitmap = raster > threshold
    if trap_bitmap.all() or not trap_bitmap.any():
        # A flat image has no shape to trap: fall back to the middle of its range.
        trap_bitmap = raster >= float(np.median(raster))

    palette = ColourConfig(mode="colors", colors=_dominant_colours(rgb, count, seed))
    return ImageAnalysis(
        raster=raster,
        colours=palette.colors,
        palette=palette,
        trap_bitmap=trap_bitmap,
        source_size=(rgb.shape[1], rgb.shape[0]),
    )


def write_assets(
    analysis: ImageAnalysis,
    directory: str | Path,
    stem: str = "fractal",
) -> tuple[Path, Path]:
    """Write the raster PNG and the palette JSON; return their paths."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    raster_path = target / f"{stem}_raster.png"
    palette_path = target / f"{stem}_palette.json"

    Image.fromarray(analysis.raster, mode="L").save(raster_path)
    palette_path.write_text(
        json.dumps(
            {
                "colours": list(analysis.colours),
                "source": {"width": analysis.source_size[0], "height": analysis.source_size[1]},
                "raster": raster_path.name,
            },
            indent=2,
        )
        + "\n"
    )
    return raster_path, palette_path


def config_from_image(
    config: Config,
    path: str | Path,
    *,
    colours: int = 6,
    raster_size: int = 96,
    strength: float = 1.4,
    blend: float = 0.8,
    output: str | Path | None = None,
    stem: str | None = None,
) -> Config:
    """Return a copy of ``config`` wired to the image's palette, map and trap.

    The raster asset is written next to the output (or the working directory when
    there is no output yet) and referenced by absolute path, so the config keeps
    working regardless of where the render is started from.
    """
    source = Path(path).expanduser()
    analysis = analyse_image(source, colours=colours, raster_size=raster_size)
    name = stem or source.stem

    target = Path(output) if output is not None else config.resolve_output()
    # With no output path configured, keep generated assets out of the working
    # directory: they belong next to a render, or in ./out.
    directory = target.parent if target is not None else Path.cwd() / "out"
    raster_path, _ = write_assets(analysis, directory, stem=name)

    fractal = config.fractal
    mapped = replace(
        fractal,
        orbit_trap=OrbitTrapConfig(image=raster_path, size=raster_size, distance_mode=1),
        map=MapConfig(
            image=raster_path,
            strength=as_float(strength, "image.strength"),
            size=raster_size,
            blend=as_float(blend, "image.blend"),
        ),
    )
    updated = replace(config, fractal=mapped, colour=analysis.palette)
    if target is not None:
        updated = replace(updated, output=target)
    return updated
