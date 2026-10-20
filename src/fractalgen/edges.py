"""Boundary (edge) extraction from the iteration field.

The original pipeline combined ``canny(conv) + sobel(conv) * -1 + canny(conv * -1)
+ sobel(conv)`` and then kept every non-zero pixel. The Sobel terms cancel
exactly and Canny's gradient magnitude is sign-invariant, so the whole
expression reduced to a (duplicated) Canny run. This module keeps the same
output while running each operator once.

scikit-image is optional: without it a Sobel-gradient threshold is used.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from . import optional

__all__ = ["detect_boundary"]


def _normalise(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    low, high = values.min(), values.max()
    if high - low <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / (high - low)


def detect_boundary(field: np.ndarray, low_threshold: float = 0.05, high_threshold: float = 0.15) -> np.ndarray:
    """Return a boolean edge mask for a 2-D scalar ``field``.

    Uses Canny on the normalised field when scikit-image is installed, and a
    Sobel-magnitude threshold otherwise.
    """
    normalised = _normalise(field)
    if optional.available("edges"):
        from skimage.feature import canny

        return np.asarray(canny(normalised, low_threshold=low_threshold, high_threshold=high_threshold), dtype=bool)

    gradient_y = ndimage.sobel(normalised, axis=0)
    gradient_x = ndimage.sobel(normalised, axis=1)
    magnitude = np.hypot(gradient_x, gradient_y)
    peak = magnitude.max()
    if peak <= 0:
        return np.zeros(normalised.shape, dtype=bool)
    return np.asarray(magnitude > 0.25 * peak, dtype=bool)
