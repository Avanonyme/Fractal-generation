"""Orbit traps: a distance field injected into the iteration.

An orbit trap is a monochrome bitmap. During the root-finding loop the current
``z`` is mapped into the bitmap and the smallest distance reached is kept. The
resulting distance field is what makes the "distance" render mode and the
relief shading possible.

The eight ``distance_mode`` values mirror the original project's ``get_distance``
variants; mode 1 (plain distance map) is the only one that is a true distance
field, the others are historical shape/filter mixes kept for visual continuity.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from scipy import ndimage

from . import optional
from ._numeric import as_int
from .config import OrbitTrapConfig

__all__ = ["OrbitTrap", "load_orbit_trap"]


def _read_bitmap(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((size, size)), dtype=float)


class OrbitTrap:
    """Precomputed orbit trap distance field.

    ``field`` holds the Euclidean distance of every pixel to the trapped shape,
    normalised to ``[-1, 1]`` (negative inside the shape), exactly like the
    original implementation, so downstream visuals stay comparable.
    """

    __slots__ = ("field", "mode")

    def __init__(self, field: np.ndarray, mode: int = 1) -> None:
        self.field = np.asarray(field, dtype=np.float64)
        self.mode = as_int(mode, "orbit_trap.distance_mode")

    @classmethod
    def from_bitmap(cls, bitmap: np.ndarray, mode: int = 1) -> "OrbitTrap":
        distance = np.asarray(ndimage.distance_transform_edt(np.logical_not(bitmap > 0)), dtype=np.float64)
        magnitude = cast(np.ndarray, np.abs(distance))
        with np.errstate(all="ignore"):
            normalised = np.where(distance == 0, 0.0, distance / magnitude)
        return cls(normalised, mode)

    def sample(self, z: np.ndarray) -> np.ndarray:
        """Distance value for each complex point of ``z`` (flattened output).

        The plane ``[-1, 1] x [-1, 1]`` is mapped onto the bitmap; points outside
        are clamped to the border, so the trap never produces out-of-range
        indices.
        """
        plane = np.asarray(z, dtype=np.complex128)
        width, height = self.field.shape
        x = (width - 1) / 2 * (np.clip(plane.real, -1, 1) + 1)
        y = (height - 1) / 2 * (np.clip(plane.imag, -1, 1) + 1)
        i = np.int64(x)
        j = np.int64(y)
        base = self.field[i, j]
        mode = self.mode

        if mode == 0:
            distance = base * plane
        elif mode == 1:
            distance = base
        elif mode == 2:
            distance = base * (1 - np.abs(x - i)) * (1 - np.abs(y - j))
        elif mode == 3:
            distance = base * np.arctan(plane)
        elif mode == 4:
            with np.errstate(all="ignore"):
                distance = base * np.sin(plane.real) * np.cos(plane.imag) / np.arctan(plane)
        elif mode == 5:
            with np.errstate(all="ignore"):
                distance = base * np.exp(-(plane**2) / (2 * base**2)) / np.sqrt(2 * np.pi * base**2)
        elif mode == 6:
            optional.require("edges")
            from skimage.filters import butterworth

            distance = butterworth(base * plane)
        elif mode == 7:
            optional.require("edges")
            from skimage.feature import canny

            distance = canny(base * plane)
        else:  # pragma: no cover - guarded by config validation
            raise ValueError(f"unknown orbit trap distance mode {mode}")

        # Distances are real: the shape variants above may produce complex
        # values, so take the magnitude before the running minimum.
        values = np.asarray(distance, dtype=np.complex128)
        magnitude = cast(np.ndarray, np.hypot(np.real(values), np.imag(values)))
        return magnitude.reshape(-1)


def load_orbit_trap(config: OrbitTrapConfig, base_dir: Path) -> OrbitTrap | None:
    """Load the configured trap, or return ``None`` when disabled."""
    if config.image is None:
        return None
    path = config.image if config.image.is_absolute() else Path(base_dir) / config.image
    if not path.exists():
        raise FileNotFoundError(f"orbit trap image not found: {path}")
    return OrbitTrap.from_bitmap(_read_bitmap(path, config.size), config.distance_mode)
