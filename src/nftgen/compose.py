"""Composition helpers: place a rendered fractal on a background canvas."""

from __future__ import annotations

import numpy as np
from PIL import Image

from fractalgen import render

__all__ = ["prepare_canvas", "place"]


def prepare_canvas(background: np.ndarray | None, size: int) -> np.ndarray:
    """Return an RGBA canvas of ``size x size`` (black when no background given)."""
    if background is None:
        return np.zeros((size, size, 4), dtype=np.uint8).__array__()
    canvas = np.asarray(background)
    if canvas.ndim == 2:
        canvas = np.dstack([canvas] * 3)
    if canvas.shape[0] != size or canvas.shape[1] != size:
        canvas = np.asarray(Image.fromarray(render.to_uint8(canvas)).resize((size, size), Image.Resampling.BICUBIC))
    if canvas.shape[2] == 3:
        alpha = np.full((size, size, 1), 255, dtype=np.uint8)
        canvas = np.concatenate([canvas, alpha], axis=2)
    return canvas


def place(
    canvas: np.ndarray,
    element: np.ndarray,
    *,
    scale: float = 0.55,
    rotation: float = 0.0,
    flip: bool = False,
    offset: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """Paste ``element`` centred on ``canvas`` with scaling, rotation and flip.

    ``offset`` is expressed as a fraction of the canvas size; ``element`` may be
    RGB or RGBA - RGB elements are treated as fully opaque.
    """
    canvas_image = Image.fromarray(render.to_uint8(canvas)).convert("RGBA")
    canvas_size = canvas_image.width

    element_array = render.to_uint8(element)
    if element_array.ndim == 2:
        element_array = np.dstack([element_array] * 3)
    if element_array.shape[2] == 3:
        alpha = np.full(element_array.shape[:2] + (1,), 255, dtype=np.uint8)
        element_array = np.concatenate([element_array, alpha], axis=2)

    target = max(1, round(canvas_size * scale))
    element_image = Image.fromarray(element_array).convert("RGBA").resize((target, target), Image.Resampling.BICUBIC)
    if flip:
        element_image = element_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rotation:
        element_image = element_image.rotate(rotation, expand=True, resample=Image.Resampling.BICUBIC)

    x = round((canvas_size - element_image.width) / 2 + offset[0] * canvas_size)
    y = round((canvas_size - element_image.height) / 2 + offset[1] * canvas_size)
    canvas_image.alpha_composite(element_image, (x, y))
    return np.asarray(canvas_image.convert("RGB"))
