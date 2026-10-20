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

__all__ = ["OrbitTrap", "load_orbit_trap", "sample_colour"]


def _read_bitmap(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("L").resize((size, size)), dtype=float)


class OrbitTrap:
    """Precomputed orbit trap distance field.

    ``field`` holds the Euclidean distance of every pixel to the trapped shape,
    scaled to ``[0, 1]`` (0 inside the shape, 1 at the farthest point of the
    bitmap). The original project reduced it to its sign, which turned every
    trap render into a two-valued mask; keeping the magnitude makes "how close
    did the orbit get to the image" the actual field, which is what makes a
    trapped image readable.
    """

    __slots__ = ("field", "mode", "colour", "colour_mode", "scale", "offset", "wrap")

    def __init__(
        self,
        field: np.ndarray,
        mode: int = 1,
        colour: np.ndarray | None = None,
        colour_mode: str = "closest",
        scale: float = 1.0,
        offset: complex = 0j,
        wrap: bool = False,
    ) -> None:
        self.field = np.asarray(field, dtype=np.float64)
        self.mode = as_int(mode, "orbit_trap.distance_mode")
        #: Optional RGB copy of the trap image, ``(h, w, 3)`` in ``[0, 1]``.
        #: This is what makes a bitmap trap *paint* the picture: the orbit is
        #: sampled in the image and the colour at the closest approach is kept
        #: (iq's bitmap orbit traps, Carlson's artistic orbit traps).
        self.colour = None if colour is None else np.asarray(colour, dtype=np.float64)
        self.colour_mode = colour_mode
        #: Tiling factor: 3 means the picture covers a third of the frame.
        self.scale = float(scale)
        self.offset = complex(offset)
        self.wrap = bool(wrap)

    @classmethod
    def from_bitmap(
        cls,
        bitmap: np.ndarray,
        mode: int = 1,
        colour: np.ndarray | None = None,
        colour_mode: str = "closest",
        scale: float = 1.0,
        offset: complex = 0j,
        wrap: bool = False,
    ) -> "OrbitTrap":
        """Build a trap from any bitmap: pixels above its median are "inside".

        Thresholding at the median keeps photographs usable - a photo has no
        black background, so ``value > 0`` would mark every pixel as inside and
        produce an empty field.
        """
        values = np.asarray(bitmap, dtype=np.float64)
        if values.size == 0:
            from .config import ConfigError

            raise ConfigError("orbit_trap: the bitmap is empty")
        threshold = float(np.median(values))
        inside = values > threshold
        if not inside.any() or inside.all():
            # A flat image has no silhouette. Fall back to its (inverted) luminance
            # so a plain upload still renders instead of failing the whole request.
            span = float(values.max() - values.min())
            smooth = (values - values.min()) / span if span > 0 else np.zeros_like(values)
            return cls(1.0 - smooth, mode, colour, colour_mode, scale, offset, wrap)
        distance = np.asarray(ndimage.distance_transform_edt(np.logical_not(inside)), dtype=np.float64)
        span = float(distance.max())
        return cls(distance / span if span > 0 else distance, mode, colour, colour_mode, scale, offset, wrap)

    def image_coordinates(self, z: np.ndarray, domain=None) -> tuple[np.ndarray, np.ndarray]:
        """Map complex points to (row, column) in the trap image.

        ``scale`` tiles the picture: 3 means it covers a third of the frame and
        repeats, which is how a large trap image is made smaller.
        """
        plane = np.asarray(z, dtype=np.complex128).reshape(-1)
        height, width = self.field.shape

        if domain is None:
            x_min = y_min = -1.0
            span_x = span_y = 2.0
        else:
            x_min, y_min = domain.x_min, domain.y_min
            span_x, span_y = domain.width, domain.height

        shifted = (plane - self.offset) * self.scale
        column = (np.real(shifted) - x_min) / span_x * (width - 1)
        row = (np.imag(shifted) - y_min) / span_y * (height - 1)
        if self.wrap:
            column = np.mod(column, width)
            row = np.mod(row, height)
        else:
            column = np.clip(column, 0, width - 1)
            row = np.clip(row, 0, height - 1)
        return row, column

    @property
    def sample_mode(self) -> str:
        return "grid-wrap" if self.wrap else "nearest"

    def sample(self, z: np.ndarray, domain=None) -> np.ndarray:
        """Distance value for each complex point of ``z`` (flattened output).

        Two details decide whether a trapped image is actually readable:

        * **Orientation.** Bitmap rows are the imaginary axis, columns the real
          axis; the previous implementation indexed ``field[real, imag]``, which
          transposed every trap.
        * **Mapping.** The plane window comes from the render domain, not from a
          hard-coded ``[-1, 1]²``. With a fixed window, any zoom or pan clamped
          every orbit point to the border and the trap went flat.

        Sampling is bilinear: a nearest-neighbour lookup stamps the raster grid
        into the result as visible squares.
        """
        from scipy import ndimage

        plane = np.asarray(z, dtype=np.complex128).reshape(-1)
        width, height = self.field.shape
        row, column = self.image_coordinates(plane, domain)
        inside = (
            (column >= 0) & (column <= width) & (row >= 0) & (row <= height)
        )
        base = np.asarray(
            ndimage.map_coordinates(self.field, [row, column], order=1, mode=self.sample_mode),
            dtype=np.float64,
        )
        # Points outside the window must not pretend to be near the shape.
        far = float(self.field.max())
        base = np.where(inside, base, far)

        mode = self.mode
        if mode == 0:
            distance = base * plane
        elif mode == 1:
            distance = base
        elif mode == 2:
            distance = base * (1 - np.abs(column - np.round(column))) * (1 - np.abs(row - np.round(row)))
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


def sample_colour(trap: OrbitTrap, z: np.ndarray, domain=None) -> np.ndarray:
    """RGB of the trap image at each orbit point (``(n, 3)`` in ``[0, 1]``)."""
    from scipy import ndimage

    if trap.colour is None:
        return np.zeros((np.asarray(z).size, 3))
    plane = np.asarray(z, dtype=np.complex128).reshape(-1)
    row, column = trap.image_coordinates(plane, domain)
    channels = [
        ndimage.map_coordinates(trap.colour[..., channel], [row, column], order=1, mode=trap.sample_mode)
        for channel in range(3)
    ]
    return np.stack(channels, axis=1)


def load_orbit_trap(config: OrbitTrapConfig, base_dir: Path) -> OrbitTrap | None:
    """Load the configured trap, or return ``None`` when disabled."""
    if config.image is None:
        return None
    path = config.image if config.image.is_absolute() else Path(base_dir) / config.image
    if not path.exists():
        raise FileNotFoundError(f"orbit trap image not found: {path}")
    shape_path = path
    if config.mask is not None:
        shape_path = config.mask if config.mask.is_absolute() else Path(base_dir) / config.mask
        if not shape_path.exists():
            raise FileNotFoundError(f"orbit trap mask not found: {shape_path}")
    bitmap = _read_bitmap(shape_path, config.size)
    return OrbitTrap.from_bitmap(
        bitmap,
        config.distance_mode,
        _read_colour(path, config.size),
        config.colour_mode,
        config.scale,
        config.offset,
        config.wrap,
    )


def _read_colour(path: Path, size: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((size, size)), dtype=float) / 255.0
