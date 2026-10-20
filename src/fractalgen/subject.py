"""Subject masks: where the picture's principal shape is.

A bitmap orbit trap needs a *silhouette*, not a photograph. Thresholding a photo
at its median marks half the pixels and the resulting distance field is a haze -
`atrest.jpg` (a painterly forest with a derrick in the middle) has no tonal
separation between subject and background at all, so no threshold can find it.

What such a picture does have is **edges**: the derrick, the mushrooms, the beams
are all strongly outlined. So the default matte closes and fills an edge map into
a solid region, then keeps the largest connected component. It is deterministic,
needs no model, and is cheap (a few hundred milliseconds at 512 px).

Methods, cheapest first:

``edges``     the Canny map itself: thin outlines, good as a trap *shape*
``silhouette`` Canny -> dilate -> close -> fill -> largest blob (solid region)
``focus``     centre-weighted edge density: "the busy area in the middle"
``grabcut``   OpenCV GrabCut with a border-initialised rectangle (photos with a
              clear centre subject)
``rect``      an explicit rectangle; the honest answer for a compositional image
              where "the subject" is an authorial decision
``threshold`` Otsu on the luminance (flat graphics, logos)
``none``      the full frame (no matte)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, cast

import numpy as np
from scipy import ndimage

from . import optional
from ._numeric import as_float, as_int
from .config import ConfigError, SubjectConfig

__all__ = ["SubjectConfig", "matte"]


# --------------------------------------------------------------------------- #
# mattes
# --------------------------------------------------------------------------- #
def _largest_blob(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected component (scipy's stubs are loose here)."""
    if not mask.any():
        return mask
    labelled, count = cast(Any, ndimage.label(mask))
    if int(count) <= 1:
        return mask
    sizes = np.asarray(ndimage.sum(mask, labelled, index=range(1, int(count) + 1)))
    best = int(np.argmax(sizes)) + 1
    return cast(np.ndarray, labelled == best)


def _canny(luma: np.ndarray, config: SubjectConfig) -> np.ndarray:
    from skimage.feature import canny

    smooth = np.asarray(ndimage.gaussian_filter(luma, config.edge_sigma, mode="nearest"), dtype=np.float64)
    span = float(smooth.max() - smooth.min())
    normalised = (smooth - smooth.min()) / span if span > 0 else np.zeros_like(smooth)
    return np.asarray(canny(normalised, low_threshold=config.low_threshold, high_threshold=config.high_threshold))


def _silhouette_matte(luma: np.ndarray, config: SubjectConfig) -> np.ndarray:
    edges = _canny(luma, config)
    # Close the outlines into regions, then fill them: an open edge map would give
    # a trap shape that is all outline and no body.
    closed = ndimage.binary_dilation(edges, iterations=2)
    closed = ndimage.binary_closing(closed, structure=np.ones((5, 5)), iterations=2)
    return np.asarray(ndimage.binary_fill_holes(closed), dtype=bool)


def _focus_matte(luma: np.ndarray, config: SubjectConfig) -> np.ndarray:
    """Centre-weighted edge density.

    On a textured picture (foliage, crowds, brush strokes) an edge map is everywhere
    at once, so the useful automatic answer is not a silhouette but *where the
    picture is busy and centred*. It reads as a soft blob, which makes a decent
    trap region and an honest "we could not find a shape" default.
    """
    edges = _canny(luma, config)
    size = luma.shape[0]
    density = np.asarray(ndimage.gaussian_filter(edges.astype(np.float64), size / 10), dtype=np.float64)
    coordinates = np.mgrid[0:size, 0:size] / max(1, size - 1)
    centre_y, centre_x = coordinates
    prior = np.exp(-(((centre_x - 0.5) ** 2 + (centre_y - 0.55) ** 2) / (2 * 0.28**2)))
    score = density * prior
    threshold = np.percentile(score, 80)
    return np.asarray(ndimage.binary_fill_holes(score >= threshold), dtype=bool)


def _grabcut_matte(rgb: np.ndarray, config: SubjectConfig) -> np.ndarray:
    optional.require("cv2")
    import cv2

    height, width = rgb.shape[:2]
    mask = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = config.rect or (0.08, 0.08, 0.92, 0.92)
    rect = (
        round(x0 * width),
        round(y0 * height),
        max(1, round((x1 - x0) * width)),
        max(1, round((y1 - y0) * height)),
    )
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        cv2.cvtColor(np.ascontiguousarray(rgb.astype(np.uint8)), cv2.COLOR_RGB2BGR),
        mask,
        rect,
        background_model,
        foreground_model,
        5,
        cv2.GC_INIT_WITH_RECT,
    )
    return np.isin(mask, (cv2.GC_FGD, cv2.GC_PR_FGD))


def _rect_matte(shape: tuple[int, int], config: SubjectConfig) -> np.ndarray:
    height, width = shape
    x0, y0, x1, y1 = config.rect if config.rect is not None else (0.1, 0.1, 0.9, 0.9)
    mask = np.zeros(shape, dtype=bool)
    mask[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = True
    return mask


def _threshold_matte(luma: np.ndarray) -> np.ndarray:
    from .image_import import otsu_threshold

    threshold = otsu_threshold(luma)
    mask = luma > threshold
    # Keep the darker side when the subject is dark (the Otsu split is symmetric).
    return mask if mask.mean() <= 0.5 else ~mask


def matte(
    rgb: np.ndarray,
    size: int = 128,
    config: SubjectConfig | None = None,
) -> np.ndarray:
    """Return a soft subject mask, ``float`` in ``[0, 1]``, shaped ``(size, size)``.

    ``rgb`` is the picture as ``(h, w, 3)`` in whatever range; it is reduced to
    ``size`` internally, because a trap bitmap never needs full resolution.
    """
    from PIL import Image

    config = config or SubjectConfig()
    image = Image.fromarray(np.asarray(rgb, dtype=np.uint8)).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    small = np.asarray(image, dtype=np.float64)
    luma = small @ np.array([0.299, 0.587, 0.114])

    if config.method == "none":
        return np.ones((size, size), dtype=np.float64)
    if config.method == "rect":
        return _threshold_free_rect(size, config)
    if config.method == "threshold":
        binary = _threshold_matte(luma)
    elif config.method == "edges":
        binary = _canny(luma, config)
    elif config.method == "focus":
        binary = _focus_matte(luma, config)
    elif config.method == "grabcut" or (config.method == "auto" and optional.available("cv2")):
        try:
            binary = _grabcut_matte(small, config)
        except (ValueError, RuntimeError):  # pragma: no cover - OpenCV edge cases
            binary = _silhouette_matte(luma, config)
    else:
        binary = _silhouette_matte(luma, config)

    if config.method == "auto":
        coverage = float(np.mean(binary)) if binary.size else 0.0
        # A silhouette that covers almost nothing (or almost everything) means the
        # edge map did not close into a region: a textured picture. Fall back to
        # the busy-and-centred region instead of pretending we found a shape.
        if coverage < 0.02 or coverage > 0.60:
            binary = _focus_matte(luma, config)

    if config.keep_largest and config.method != "edges":
        # An edge map is many small components by design; reducing it to the
        # biggest one leaves a single speck.
        binary = _largest_blob(binary)
    if not binary.any():  # pragma: no cover - degenerate pictures
        binary = np.ones((size, size), dtype=bool)

    mask = binary.astype(np.float64)
    if config.expand:
        mask = np.asarray(ndimage.binary_dilation(mask > 0.5, iterations=config.expand), dtype=np.float64)
    if config.feather > 0:
        mask = np.asarray(ndimage.gaussian_filter(mask, config.feather, mode="nearest"), dtype=np.float64)
        peak = float(mask.max()) or 1.0
        mask = mask / peak
    if config.invert:
        mask = 1.0 - mask
    return np.clip(mask, 0.0, 1.0)


def _threshold_free_rect(size: int, config: SubjectConfig) -> np.ndarray:
    """Rect matte with the same feathering/expansion path as the others."""
    mask = _rect_matte((size, size), config).astype(np.float64)
    if config.feather > 0:
        mask = np.asarray(ndimage.gaussian_filter(mask, config.feather, mode="nearest"), dtype=np.float64)
        peak = float(mask.max()) or 1.0
        mask = mask / peak
    return np.clip(mask, 0.0, 1.0)
