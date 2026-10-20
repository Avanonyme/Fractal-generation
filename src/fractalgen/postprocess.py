"""Post-process shaders applied to a finished frame.

The banded look (``return_type: "iteration"``) is worth keeping, but colormapping
an iteration count is an iso-contour plot: inside a basin the count steps by
whole numbers, so the palette steps with it and the frame shows concentric
rings. Those rings are a *rendering* artifact, not structure, and they have a
signature the filters here can exploit:

* the colour step across a ring is small - a fraction of the palette;
* the step across real structure (a filament) is large.

A bilateral-style filter therefore removes the small steps while leaving the
filaments untouched. ``deband`` is that filter; ``unsharp`` restores a little
local contrast afterwards, which is usually lost when a small-radius blur is
used on a 1 px transition.

These are deliberately *post* filters: they run on the composed RGB frame, so
they work the same for every render type and for animation frames.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from . import optional
from ._numeric import as_float, as_int
from .config import ConfigError, PostprocessConfig
from .render import to_uint8

__all__ = ["DEBAND_DEFAULTS", "UNSHARP_DEFAULTS", "FILTERS", "apply", "apply_all", "resolve_params"]


DEBAND_DEFAULTS: dict[str, Any] = {
    #: Neighbourhood radius in pixels; 2-4 removes 1 px ring transitions.
    "radius": 3,
    #: Colour distance (0-255) above which a difference counts as structure.
    #: Keep it small: it must sit above a palette step, below a filament step.
    "sigma_color": 14.0,
    #: Spatial falloff; larger values blend further across a ring.
    "sigma_space": 2.5,
    #: Passes: each pass removes the next coarser ring.
    "passes": 1,
}

UNSHARP_DEFAULTS: dict[str, Any] = {
    #: Blur radius used to build the high-pass layer.
    "radius": 2.0,
    #: Strength of the restored detail (0.2-0.6 is subtle).
    "amount": 0.35,
    #: Detail smaller than this (0-255) is treated as noise and not boosted.
    "threshold": 2,
}


def _bilateral(rgb: np.ndarray, radius: int, sigma_color: float, sigma_space: float) -> np.ndarray:
    if optional.available("cv2"):
        import cv2

        diameter = 2 * radius + 1
        # OpenCV only implements bilateral filtering for uint8/float32 buffers.
        source = np.ascontiguousarray(to_uint8(rgb))
        return np.asarray(
            cv2.bilateralFilter(source, diameter, sigmaColor=sigma_color, sigmaSpace=sigma_space), dtype=np.uint8
        )

    # Fallback: a colour-guided blur. Weights are computed from the colour
    # distance to the centre pixel, which is the essence of the bilateral filter;
    # it is slower but needs no OpenCV.
    from scipy import ndimage

    array = np.asarray(rgb, dtype=np.float64)
    accumulated = np.zeros_like(array)
    weights = np.zeros(array.shape[:2])
    for offset_y in range(-radius, radius + 1):
        for offset_x in range(-radius, radius + 1):
            shifted = np.roll(np.roll(array, offset_y, axis=0), offset_x, axis=1)
            distance = np.abs(shifted - array).mean(axis=2)
            spatial = np.exp(-(offset_y**2 + offset_x**2) / (2 * sigma_space**2))
            weight = spatial * np.exp(-(distance**2) / (2 * sigma_color**2))
            accumulated += shifted * weight[..., None]
            weights += weight
    smoothed = accumulated / np.maximum(weights, 1e-9)[..., None]
    return smoothed


def deband(rgb: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Remove iso-contour ring steps while keeping real edges."""
    options = resolve_params("deband", params)
    smoothed = to_uint8(rgb)
    for _ in range(max(1, as_int(options["passes"], "postprocess.passes"))):
        smoothed = _bilateral(
            smoothed,
            radius=as_int(options["radius"], "postprocess.radius"),
            sigma_color=as_float(options["sigma_color"], "postprocess.sigma_color"),
            sigma_space=as_float(options["sigma_space"], "postprocess.sigma_space"),
        )
    return to_uint8(smoothed)


def unsharp(rgb: np.ndarray, params: Mapping[str, Any]) -> np.ndarray:
    """Restore local contrast after a de-banding pass."""
    from scipy import ndimage

    options = resolve_params("unsharp", params)
    array = np.asarray(rgb, dtype=np.float64)
    blurred = ndimage.gaussian_filter(array, as_float(options["radius"], "postprocess.radius"), mode="nearest")
    detail = array - blurred
    threshold = as_float(options["threshold"], "postprocess.threshold")
    detail = np.where(np.abs(detail) > threshold, detail, np.zeros_like(detail))
    amount = as_float(options["amount"], "postprocess.amount")
    return to_uint8(np.clip(array + amount * detail, 0, 255))


FILTERS = {"deband": deband, "unsharp": unsharp}

#: Allowed parameters per filter (unknown keys are rejected).
PARAMS = {"deband": DEBAND_DEFAULTS, "unsharp": UNSHARP_DEFAULTS}


def resolve_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Merge user parameters over a filter's defaults, rejecting typos."""
    if name not in PARAMS:
        raise ConfigError(f"postprocess {name!r}: unknown filter; known: {sorted(PARAMS)}")
    unknown = sorted(set(params) - set(PARAMS[name]))
    if unknown:
        raise ConfigError(
            f"postprocess {name}: unknown parameter(s) {unknown}; allowed: {', '.join(sorted(PARAMS[name]))}"
        )
    merged = dict(PARAMS[name])
    merged.update(params)
    return merged


def apply(step: PostprocessConfig, rgb: np.ndarray) -> np.ndarray:
    """Run one configured filter."""
    return FILTERS[step.name](rgb, step.params)


def apply_all(steps: Iterable[PostprocessConfig], rgb: np.ndarray) -> np.ndarray:
    for step in steps:
        rgb = apply(step, rgb)
    return rgb
