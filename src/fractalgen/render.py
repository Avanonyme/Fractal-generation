"""Array -> PNG rendering.

Rendering no longer goes through a matplotlib figure. The original code created
a ``Figure``, an ``Axes`` and called ``savefig`` for every intermediate image
(~100 ms each), which dominated run time for small fractals. Here the colormap
is applied with numpy and written with Pillow.

:func:`colorize` accepts

* a 2-D scalar field -> colormapped RGB,
* an ``(h, w, 3|4)`` array -> passed through (already coloured, e.g. LightSource).

so both shading families flow through the same save path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.colors import Colormap
from PIL import Image

__all__ = ["normalise", "colorize", "save", "load", "compose_over", "resize", "to_uint8"]


def normalise(values: np.ndarray) -> np.ndarray:
    """Scale a field into ``[0, 1]``; constant fields become zeros (not NaN)."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    if not finite.all():
        values = np.where(finite, values, np.nanmin(np.where(finite, values, np.nan)) if finite.any() else 0.0)
    low = np.float64(values.min())
    high = np.float64(values.max())
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def to_uint8(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array):
        array = np.real(array)
    return np.clip(array, 0, 255).astype(np.uint8)


def normalise_with(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    """Clip and scale a field to ``[0, 1]`` using a fixed range.

    Animations pass the range of their first frame so brightness stays constant
    from frame to frame instead of re-stretching to each frame's own min/max
    (which makes a zoom visibly pulse as the content brightens).
    """
    low, high = limits
    if high - low <= 1e-12:
        return np.zeros_like(np.asarray(values, dtype=np.float64))
    return np.clip((np.asarray(values, dtype=np.float64) - low) / (high - low), 0.0, 1.0)


def colorize(values: np.ndarray, cmap: Colormap, value_range: tuple[float, float] | None = None) -> np.ndarray:
    """Map a scalar field to ``(h, w, 3)`` uint8 through ``cmap``.

    3-D input is treated as an already-coloured image (LightSource shading
    returns float RGBA in ``[0, 1]``); it is rescaled instead of truncated.
    """
    array = np.asarray(values)
    if array.ndim == 3:
        channels = array[..., :3]
        if channels.dtype.kind == "f" and channels.max(initial=0) <= 1.0:
            channels = channels * 255
        return to_uint8(channels)
    scaled = normalise(array) if value_range is None else normalise_with(array, value_range)
    rgba = cmap(scaled)
    return to_uint8(rgba[..., :3] * 255)


def save(image: np.ndarray, path: str | Path) -> Path:
    """Write an RGB(A) uint8 array to ``path`` (``.png`` by default)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image)).save(path)
    return path


def load(path: str | Path, size: int | None = None) -> np.ndarray:
    """Read an image as RGB(A) uint8, optionally resized to ``size x size``."""
    with Image.open(Path(path)) as handle:
        picture = handle.convert("RGBA")
        if size is not None and picture.size != (size, size):
            picture = picture.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(picture)


def resize(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(to_uint8(image)).resize((size, size), Image.Resampling.BICUBIC))


def compose_over(foreground: np.ndarray, background: np.ndarray, alpha: int | np.ndarray = 255) -> np.ndarray:
    """Alpha-composite ``foreground`` centred on ``background``.

    ``alpha`` is either a scalar 0..255 or a per-pixel array; the foreground's own
    alpha channel (if any) is multiplied with it.
    """
    background = load(background) if isinstance(background, (str, Path)) else np.asarray(background)
    foreground = load(foreground) if isinstance(foreground, (str, Path)) else np.asarray(foreground)
    if background.ndim == 2:
        background = np.dstack([background] * 3)
    background = background[..., :3].astype(np.float64)

    height, width = background.shape[:2]
    if foreground.shape[:2] != (height, width):
        foreground = resize(foreground, max(height, width))
    if foreground.ndim == 2:
        foreground = np.dstack([foreground] * 3)

    weight = np.ones(foreground.shape[:2], dtype=np.float64)
    if foreground.shape[2] == 4:
        weight *= foreground[..., 3].astype(np.float64) / 255.0
    weight *= np.asarray(alpha, dtype=np.float64) / 255.0
    if weight.ndim == 0:
        weight = np.full(foreground.shape[:2], np.float64(weight))

    blended = foreground[..., :3].astype(np.float64) * weight[..., None] + background * (1 - weight[..., None])
    return to_uint8(blended)
