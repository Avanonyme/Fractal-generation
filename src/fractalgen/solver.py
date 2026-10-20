"""Chunked active-set root-finding solver.

This is the module you use when you only want the fractal maths::

    from fractalgen import FractalConfig, Polynomial, solve

    result = solve(FractalConfig(size=512), Polynomial.random(6, np.random.default_rng(0)))
    result.iterations, result.roots, result.distance, result.boundary

Performance notes compared with the original implementation:

* **Horner evaluation** instead of ``sum(coefs[k] * z**k)`` (one multiply-add per
  term instead of a power).
* **Active-set masking**: converged pixels drop out of the iteration loop, so
  the cost scales with the number of still-active pixels rather than ``N^2``
  times the iteration count.
* **No memmap round-trip**: tiles are returned from workers directly. The old
  code wrote every tile to disk three times, then read it back, and those reads
  could fail silently because the ``executor.map`` iterator was never consumed.
* **Vectorised Nova coefficient mapping** instead of a Python double loop over
  every pixel and every coefficient.
"""

from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigError, FractalConfig
from .edges import detect_boundary
from .orbit import OrbitTrap, sample_colour
from .polynomial import Polynomial

__all__ = ["FractalImage", "solve", "pixel_grid", "choose_polynomial", "chunk_slices"]


@dataclass(frozen=True)
class FractalImage:
    """Result of one solve: raw fields plus the polynomial that produced them."""

    iterations: np.ndarray
    smooth_iterations: np.ndarray
    roots: np.ndarray
    roots_distance: np.ndarray
    distance: np.ndarray | None
    trap_colour: np.ndarray | None
    normal: np.ndarray | None
    boundary: np.ndarray
    polynomial: Polynomial
    config: FractalConfig

    @property
    def shape(self) -> tuple[int, int]:
        return self.iterations.shape

    @property
    def saturation(self) -> float:
        """Share of pixels that exhausted the iteration budget."""
        return np.mean(self.iterations > self.config.iterations).item()

    @property
    def converged_share(self) -> float:
        return 1.0 - self.saturation

    @property
    def zoom(self) -> float:
        """How far this frame is zoomed relative to a 2-unit-wide window."""
        return 2.0 / max(self.config.domain.width, 1e-300)

    def scalar_field(self, kind: str) -> np.ndarray:
        """Pick the field that should be colourised.

        ``iteration`` is the continuous count (no stepping); use
        ``iteration-steps`` for the raw integer counter.
        """
        if kind == "iteration":
            return self.smooth_iterations
        if kind == "iteration-steps":
            return self.iterations
        if kind == "root":
            return self.root_index().astype(np.float64)
        if kind == "smooth":
            return self.smooth_albedo()
        if kind == "distance":
            if self.distance is None:
                raise ValueError("render type 'distance' needs an orbit trap")
            return self.distance
        if kind == "trap":
            if self.trap_colour is None:
                raise ValueError("render type 'trap' needs an orbit trap image with colour")
            return self.trap_colour
        if kind == "boundary":
            return self.boundary.astype(np.float64)
        if kind == "roots":
            return np.real(self.roots)
        raise ValueError(f"unknown render type {kind!r}")

    def height_field(self, kind: str = "iteration", height: str = "auto") -> np.ndarray:
        """Field used for relief shading.

        Integer iteration counts are a step function, so shading them embosses
        every level boundary into a visible contour ("terracing"). The default
        therefore uses a continuous field:

        ``auto``             orbit-trap distance if there is one, else the log
                             distance to the attractor
        ``distance``         orbit-trap distance (requires a trap)
        ``roots``            log distance from each pixel to its attractor
        ``iteration``        log of the continuous iteration count
        ``iteration-steps``  raw integer counts (the old terraced look)
        """
        if height == "iteration-steps":
            return self.iterations
        if height == "distance":
            if self.distance is None:
                raise ValueError("shading.height='distance' needs an orbit trap")
            return self.distance
        if height == "roots":
            return self.log_roots_distance()
        if height == "iteration":
            return np.log1p(self.smooth_iterations)
        if height != "auto":
            raise ValueError(f"unknown shading height {height!r}")
        if self.distance is not None:
            return self.distance
        return self.log_roots_distance()

    def smooth_albedo(self) -> np.ndarray:
        """Basin hue plus a restrained inner gradient.

        One colormap step is reserved per root, so the value inside basin ``k``
        stays in ``[k, k + tint]`` and can never cross into the next hue: the
        colouring is piecewise constant *by construction*, which is the only way
        to guarantee no contour rings. ``tint`` (0.35 of a step) comes from the
        continuous iteration count and only adds depth.
        """
        index = self.root_index().astype(np.float64)
        count = self.smooth_iterations
        low, high = float(np.min(count)), float(np.max(count))
        tint = (count - low) / (high - low) if high - low > 1e-12 else np.zeros_like(count)
        return index + 0.35 * tint

    def root_index(self) -> np.ndarray:
        """Index of the root each pixel converged to; ``len(roots)`` if it did not.

        Colouring by root identity is the only colouring that cannot produce
        contour rings, because it is piecewise constant per basin.
        """
        roots = self.polynomial.roots
        distances = np.abs(self.roots.ravel()[:, None] - roots[None, :])
        index = np.argmin(distances, axis=1).reshape(self.shape)
        index[self.iterations > self.config.iterations] = len(roots)
        return index

    def log_roots_distance(self) -> np.ndarray:
        """``log10(|z0 - root|)``: the smooth "basin potential".

        Distance to the root a pixel converged to is continuous and has a
        continuous gradient across a basin, so shading it produces a smooth
        gradient. Iteration counts are a staircase (and their fractional
        "smoothing" is noisy), which is what produced visible contour rings.
        Non-converged pixels sit on the Julia set: they are clamped to the
        largest distance so they read as the ridge between basins.
        """
        distance = np.asarray(self.roots_distance, dtype=np.float64)
        finite = distance[np.isfinite(distance)]
        ceiling = np.percentile(finite, 99.5).item() if finite.size else 1.0
        ceiling = max(float(ceiling), 1e-9)
        clamped = np.clip(np.nan_to_num(distance, nan=ceiling, posinf=ceiling), 1e-9, ceiling)
        return np.log10(clamped)


def normalise_field(values: np.ndarray) -> np.ndarray:
    """Scale a field into ``[0, 1]`` (constant fields become zeros)."""
    array = np.asarray(values, dtype=np.float64)
    low, high = array.min(), array.max()
    if high - low <= 1e-12:
        return np.zeros_like(array)
    return (array - low) / (high - low)


def pixel_grid(size: int, domain) -> np.ndarray:
    """Complex pixel grid; rows are the imaginary axis, columns the real axis.

    The grid is built as ``centre + offset * width`` rather than from
    ``linspace(x_min, x_max)``: subtracting two nearly equal absolute
    coordinates loses precision at depth, while scaling a small offset does not.
    """
    offsets = np.arange(size, dtype=np.float64) / max(1, size) - 0.5
    centre = domain.center
    real = centre.real + offsets * domain.width
    imaginary = centre.imag + offsets * domain.height
    return real[np.newaxis, :] + 1j * imaginary[:, np.newaxis]


def chunk_slices(size: int, workers: int, minimum_tile: int = 64) -> list[tuple[slice, slice]]:
    """Split a square grid into at most ``workers`` tiles of at least ``minimum_tile``."""
    if workers <= 1 or size < 2 * minimum_tile:
        return [(slice(0, size), slice(0, size))]
    divisions = max(1, min(math.isqrt(workers), size // minimum_tile))
    if divisions <= 1:
        return [(slice(0, size), slice(0, size))]
    boundaries = np.linspace(0, size, divisions + 1).astype(int)
    return [
        (slice(row, next_row), slice(column, next_column))
        for row, next_row in zip(boundaries[:-1], boundaries[1:])
        for column, next_column in zip(boundaries[:-1], boundaries[1:])
    ]


# --------------------------------------------------------------------------- #
# iteration kernels
# --------------------------------------------------------------------------- #
def _newton(
    start: np.ndarray,
    polynomial: Polynomial,
    config: FractalConfig,
    trap: OrbitTrap | None,
    damping: complex | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    z = np.asarray(start, dtype=np.complex128).ravel().copy()
    iterations = np.full(z.shape, config.iterations + 1, dtype=np.float64)
    smooth = np.full(z.shape, config.iterations + 1, dtype=np.float64)
    previous = np.full(z.shape, np.inf)
    active = np.ones(z.shape, dtype=bool)
    trap_field = trap
    if trap_field is not None:
        # Seed with the pixel's own position: Newton collapses onto a root within
        # a few steps, so an orbit-only trap barely samples the image.
        distance = np.minimum(np.full(z.shape, np.inf), trap_field.sample(z, config.domain))
        trap_colour = (
            sample_colour(trap_field, z, config.domain) if trap_field.colour is not None else None
        )
        colour_sum = np.zeros_like(trap_colour) if trap_colour is not None else None
        colour_count = np.zeros(z.shape) if trap_colour is not None else None
    else:
        distance = None
        trap_colour = None
        colour_sum = None
        colour_count = None

    for step in range(1, config.iterations + 1):
        index = np.flatnonzero(active)
        if index.size == 0:
            break
        current = z[index]
        # A per-pixel damping array is aligned with the tile, so slice it with the
        # same active-set index the kernel is iterating over.
        local_damping = damping[index] if isinstance(damping, np.ndarray) else damping
        delta, stepped = _newton_step(polynomial, current, local_damping)
        current = current + delta
        z[index] = current

        magnitude = np.abs(delta)
        # A stalled point (zero derivative) must not count as converged: its
        # step is zero, which would otherwise look like perfect precision.
        converged = (magnitude < config.tolerance) & stepped
        if converged.any():
            finished = index[converged]
            iterations[finished] = step
            smooth[finished] = _smooth_iteration(step, magnitude[converged], config.tolerance, order=2)
            active[finished] = False
        previous[index] = magnitude

        if distance is not None and trap_field is not None:
            sampled = trap_field.sample(current, config.domain)
            closer = sampled < distance[index]
            distance[index] = np.minimum(distance[index], sampled)
            if trap_colour is not None and colour_sum is not None and colour_count is not None:
                # Orbit-trap colouring (iq's bitmap traps): the pixel takes the
                # trap image's colour sampled *along the orbit*.
                picked = sample_colour(trap_field, current, config.domain)
                mode = trap_field.colour_mode
                if mode == "closest":
                    if closer.any():
                        trap_colour[index[closer]] = picked[closer]
                elif mode == "last":
                    trap_colour[index] = picked
                else:  # average
                    colour_sum[index] += picked
                    colour_count[index] += 1.0
                    averaged: np.ndarray = colour_sum[index] / colour_count[index][:, None]
                    trap_colour[index] = averaged

    return iterations, smooth, z, distance, trap_colour


def _halley(
    start: np.ndarray,
    polynomial: Polynomial,
    config: FractalConfig,
    trap: OrbitTrap | None,
    damping: complex | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    z = np.asarray(start, dtype=np.complex128).ravel().copy()
    iterations = np.full(z.shape, config.iterations + 1, dtype=np.float64)
    smooth = np.full(z.shape, config.iterations + 1, dtype=np.float64)
    previous = np.full(z.shape, np.inf)
    active = np.ones(z.shape, dtype=bool)
    trap_field = trap
    if trap_field is not None:
        # Seed with the pixel's own position: Newton collapses onto a root within
        # a few steps, so an orbit-only trap barely samples the image.
        distance = np.minimum(np.full(z.shape, np.inf), trap_field.sample(z, config.domain))
        trap_colour = (
            sample_colour(trap_field, z, config.domain) if trap_field.colour is not None else None
        )
        colour_sum = np.zeros_like(trap_colour) if trap_colour is not None else None
        colour_count = np.zeros(z.shape) if trap_colour is not None else None
    else:
        distance = None
        trap_colour = None
        colour_sum = None
        colour_count = None

    for step in range(1, config.iterations + 1):
        index = np.flatnonzero(active)
        if index.size == 0:
            break
        current = z[index]
        local_damping = damping[index] if isinstance(damping, np.ndarray) else damping
        delta, stepped = _halley_step(polynomial, current, local_damping)
        current = current + delta
        z[index] = current

        magnitude = np.abs(delta)
        converged = (magnitude < config.tolerance) & stepped
        if converged.any():
            finished = index[converged]
            iterations[finished] = step
            smooth[finished] = _smooth_iteration(step, magnitude[converged], config.tolerance, order=2)
            active[finished] = False
        previous[index] = magnitude

        if distance is not None and trap_field is not None:
            sampled = trap_field.sample(current, config.domain)
            closer = sampled < distance[index]
            distance[index] = np.minimum(distance[index], sampled)
            if trap_colour is not None and colour_sum is not None and colour_count is not None:
                # Orbit-trap colouring (iq's bitmap traps): the pixel takes the
                # trap image's colour sampled *along the orbit*.
                picked = sample_colour(trap_field, current, config.domain)
                mode = trap_field.colour_mode
                if mode == "closest":
                    if closer.any():
                        trap_colour[index[closer]] = picked[closer]
                elif mode == "last":
                    trap_colour[index] = picked
                else:  # average
                    colour_sum[index] += picked
                    colour_count[index] += 1.0
                    averaged: np.ndarray = colour_sum[index] / colour_count[index][:, None]
                    trap_colour[index] = averaged

    return iterations, smooth, z, distance, trap_colour


def _smooth_iteration(step: int, residual: np.ndarray, tolerance: float, order: int = 2) -> np.ndarray:
    """Continuous iteration count for a *convergent* iteration.

    Convergent root finders are super-attracting, so the residual shrinks like
    ``u_{n+1} ~ u_n**order`` (quadratic for Newton, cubic for Halley). Solving
    ``ln u_n`` for the fractional step gives

        nu = n + log_order( ln(u) / ln(tolerance) )

    so ``u == tolerance`` yields exactly ``n`` and ``u == sqrt(tolerance)`` yields
    ``n - 1``. Mandelbrot escape-time smoothing does *not* transfer here (it
    assumes |z| -> infinity, not convergence), which is why the earlier
    interpolation of successive steps produced a noisy, banded field.

    ``residual`` is the step size at the iteration that satisfied the tolerance,
    i.e. the distance to the attracting root.
    """
    safe = np.maximum(np.abs(residual), 1e-300)
    with np.errstate(all="ignore"):
        ratio = np.log(safe) / np.log(tolerance)
        offset = np.log(np.maximum(ratio, 1e-300)) / np.log(order)
    offset = np.nan_to_num(offset, nan=0.0, posinf=0.0, neginf=0.0)
    return step + np.clip(offset, 0.0, 1.0)


def _newton_step(
    polynomial: Polynomial, z: np.ndarray, damping: complex | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``-damping * f(z) / f'(z)`` plus a mask of the points that moved.

    ``damping`` is a scalar, or one value per point when a map modifier scales it
    (see :class:`~fractalgen.config.MapConfig`).

    A zero derivative used to produce ``inf``/``nan`` pixels that poisoned every
    downstream array (and made tiled and single-tile runs disagree). Points on a
    multiple root keep their position and are reported as not stepped, so they
    never count as converged.
    """
    derivative = polynomial.derivative(z)
    step = np.zeros(z.shape, dtype=np.complex128)
    nonzero = derivative != 0
    factor = damping[nonzero] if isinstance(damping, np.ndarray) else damping
    with np.errstate(all="ignore"):
        step[nonzero] = -factor * polynomial(z[nonzero]) / derivative[nonzero]
    return step, nonzero


def _halley_step(
    polynomial: Polynomial, z: np.ndarray, damping: complex | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Halley step with the same zero-denominator protection as Newton."""
    value = polynomial(z)
    first = polynomial.derivative(z)
    second = polynomial.second_derivative(z)
    denominator = 2 * first**2 - second * value
    step = np.zeros(z.shape, dtype=np.complex128)
    nonzero = denominator != 0
    factor = damping[nonzero] if isinstance(damping, np.ndarray) else damping
    with np.errstate(all="ignore"):
        step[nonzero] = -factor * 2 * value[nonzero] * first[nonzero] / denominator[nonzero]
    return step, nonzero


_KERNELS = {"newton": _newton, "halley": _halley}


@lru_cache(maxsize=8)
def _map_raster(path: str, size: int) -> np.ndarray:
    """Luminance raster used as a map modifier, cached per (file, size).

    Every worker tile asks for the same file; re-reading and re-resizing the PNG
    per tile was measurably slower than the iteration itself on small renders.
    """
    from PIL import Image

    from scipy import ndimage

    with Image.open(path) as image:
        luminance = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    values = np.asarray(luminance, dtype=np.float64) / 255.0
    # A little smoothing keeps the modulation continuous across raster cells.
    return ndimage.gaussian_filter(values, max(0.8, size / 28), mode="nearest")


def _sample_raster(raster: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Bilinear lookup of ``raster`` at complex points in ``[-1, 1]^2``.

    Bilinear (not nearest) matters: a nearest-neighbour lookup stamps the raster's
    own grid into the fractal, which reads as square blocks rather than a picture.
    """
    from scipy import ndimage

    height, width = raster.shape
    columns = (np.real(z) + 1) / 2 * (width - 1)
    rows = (np.imag(z) + 1) / 2 * (height - 1)
    sampled = ndimage.map_coordinates(
        raster, [rows.ravel(), columns.ravel()], order=1, mode="nearest"
    )
    return np.asarray(sampled).reshape(np.shape(z))


def _warped_start(config: FractalConfig, start: np.ndarray) -> np.ndarray:
    """Displace the starting points along the image gradient.

    ``strength`` is scale-free: 1 means "shift by one raster cell", so the same
    value works for any resolution or zoom. Displacement is linear in strength, so
    - unlike modulating the damping - the fractal stays intact while its basin
    boundaries follow the picture.
    """
    from scipy import ndimage

    map_config = config.map
    if not map_config.enabled or map_config.image is None:
        return start
    raster = _map_raster(str(map_config.image), map_config.size)
    gradient_y, gradient_x = np.gradient(raster)
    peak = float(np.max(np.hypot(gradient_x, gradient_y)))
    if peak <= 0:
        return start

    domain = config.domain
    grid = np.asarray(start, dtype=np.complex128)
    columns = (np.real(grid) + 1) / 2 * (map_config.size - 1)
    rows = (np.imag(grid) + 1) / 2 * (map_config.size - 1)
    sampled_x = ndimage.map_coordinates(gradient_x, [rows.ravel(), columns.ravel()], order=1, mode="nearest")
    sampled_y = ndimage.map_coordinates(gradient_y, [rows.ravel(), columns.ravel()], order=1, mode="nearest")

    # One raster cell in plane units.
    cell = max(domain.width, domain.height) / max(1, map_config.size)
    displacement = (sampled_x + 1j * sampled_y) / peak * cell * map_config.strength
    return grid.reshape(-1) + displacement


def _pixel_damping(config: FractalConfig, start: np.ndarray) -> complex | np.ndarray:
    """Damping for the root finder, optionally scaled per pixel by the map image.

    A constant damping produces the usual symmetric basins; letting the map scale
    it locally deforms them, which is what makes the source image readable inside
    the fractal (see :class:`~fractalgen.config.MapConfig`).
    """
    map_config = config.map
    if not map_config.enabled or map_config.image is None:
        return config.damping
    raster = _map_raster(str(map_config.image), map_config.size)
    values = _sample_raster(raster, np.asarray(start, dtype=np.complex128))
    return config.damping * (1.0 + map_config.strength * (values - 0.5))


def _nova_start(start: np.ndarray, polynomial: Polynomial, config: FractalConfig) -> tuple[np.ndarray, Polynomial]:
    """Critical-point start used by the ``nova`` variant.

    Each pixel gets its own perturbed polynomial; the iteration starts at a root
    of its second derivative. The original code called ``np.roots`` once per
    pixel (millions of companion-matrix solves); here the critical point is found
    with a vectorised Newton loop, which is roughly three orders of magnitude
    faster and gives the same family of starting points.
    """
    rng = np.random.default_rng(config.polynomial.seed if config.polynomial.seed is not None else 0)
    family = polynomial.family(np.asarray(start).ravel(), rng, perturb=True)
    reference = Polynomial(family[0])
    if reference.degree >= 2:
        z = np.asarray(start, dtype=np.complex128).ravel().copy()
        second = Polynomial(reference.second_derivative_coefficients())
        third = Polynomial(second.derivative_coefficients())
        for _ in range(40):
            with np.errstate(all="ignore"):
                z = z - second(z) / third(z)
            if not np.all(np.isfinite(z)):
                break
        z = np.where(np.isfinite(z), z, np.asarray(start, dtype=np.complex128).ravel())
    else:
        z = np.asarray(start, dtype=np.complex128).ravel().copy()
    return z, polynomial


def _solve_tile(
    start: np.ndarray,
    polynomial: Polynomial,
    config: FractalConfig,
    trap: OrbitTrap | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    kernel = _KERNELS[config.method]
    shape = np.asarray(start).shape
    origin = np.asarray(start, dtype=np.complex128).reshape(-1)
    if config.variant == "nova":
        start, polynomial = _nova_start(start, polynomial, config)
    if config.map.mode == "warp":
        start = _warped_start(config, start)
        damping = config.damping
    else:
        damping = _pixel_damping(config, origin)
    iterations, smooth, roots, distance, trap_colour = kernel(start, polynomial, config, trap, damping)

    # Distance from each pixel to the attractor it reached, measured on the raw
    # values: this is the smooth "basin potential" used for shading, so it must
    # not inherit the root quantisation that root-identity colouring uses.
    reveal = np.abs(origin - roots)

    if config.round_digits:
        roots = np.round(roots, config.round_digits)
    distance = None if distance is None else distance.reshape(shape)
    if trap_colour is not None:
        trap_colour = trap_colour.reshape(shape[0], shape[1], 3) if len(shape) == 2 else trap_colour
    return (
        iterations.reshape(shape),
        smooth.reshape(shape),
        roots.reshape(shape),
        reveal.reshape(shape),
        distance,
        trap_colour,
    )


def solve(
    config: FractalConfig,
    polynomial: Polynomial | None = None,
    *,
    workers: int = 1,
    progress: bool = False,
) -> FractalImage:
    """Compute the fractal fields for ``config``.

    ``polynomial`` may be supplied to reuse a specific polynomial (animations do
    this so every frame keeps the same roots); otherwise one is created from the
    configuration (random polynomials are re-sampled until the image is
    well-conditioned).
    """
    if not config.domain.can_resolve:
        raise ConfigError(
            f"fractal.domain: window {config.domain.width:.3e} is below float64 resolution "
            f"({config.domain.resolution:.3e} per pixel); maximum zoom for this window is about "
            f"{config.domain.max_zoom:,.0f}x"
        )

    grid = pixel_grid(config.size, config.domain)
    trap = _load_trap(config)
    if polynomial is None:
        polynomial = choose_polynomial(config, trap, progress=progress)

    config = _apply_iteration_budget(config, polynomial, trap, progress)
    grid = pixel_grid(config.size, config.domain)

    if workers > 1 and not _processes_available():
        if progress:
            print("running serially: the __main__ module is not importable (no __file__)")
        workers = 1

    tiles = chunk_slices(config.size, workers)
    if len(tiles) == 1:
        iterations, smooth, roots, reveal, distance, trap_colour = _solve_tile(grid, polynomial, config, trap)
    else:
        iterations = np.empty(grid.shape, dtype=np.float64)
        smooth = np.empty(grid.shape, dtype=np.float64)
        roots = np.empty(grid.shape, dtype=np.complex128)
        reveal = np.empty(grid.shape, dtype=np.float64)
        trap_colour = np.zeros((*grid.shape, 3)) if trap is not None and trap.colour is not None else None
        distance = np.full(grid.shape, np.inf) if trap is not None else None
        arguments = [(grid[rows, columns], polynomial, config, trap) for rows, columns in tiles]
        with ProcessPoolExecutor(max_workers=min(len(tiles), workers)) as pool:
            results = list(pool.map(_solve_tile_star, arguments))
        for (rows, columns), tile in zip(tiles, results):
            tile_iterations, tile_smooth, tile_roots, tile_reveal, tile_distance, tile_colour = tile
            iterations[rows, columns] = tile_iterations
            smooth[rows, columns] = tile_smooth
            roots[rows, columns] = tile_roots
            reveal[rows, columns] = tile_reveal
            if trap_colour is not None and tile_colour is not None:
                trap_colour[rows, columns] = tile_colour
            if distance is not None and tile_distance is not None:
                distance[rows, columns] = tile_distance

    if distance is not None:
        distance = np.where(np.isfinite(distance), distance, 0.0)

    boundary = detect_boundary(iterations)
    normal = None
    if distance is not None:
        gradient_y, gradient_x = np.gradient(distance)
        normal = gradient_x + 1j * gradient_y

    if progress:
        print(f"solved {config.size}x{config.size} in {len(tiles)} tile(s)")

    image = FractalImage(
        iterations=iterations,
        smooth_iterations=smooth,
        roots=roots,
        roots_distance=reveal,
        distance=distance,
        trap_colour=trap_colour,
        normal=normal,
        boundary=boundary,
        polynomial=polynomial,
        config=config,
    )
    return image


def _processes_available() -> bool:
    """Whether worker processes can re-import the entry point.

    Spawn-based pools (the default on macOS and Windows) import ``__main__`` in
    every worker, so a script without a ``if __name__ == "__main__"`` guard, or
    an interactive/stdin session, cannot use them. Falling back to a serial run
    beats the original behaviour, where the pool died with a traceback per
    worker and the caller got no result at all.
    """
    import __main__

    path = getattr(__main__, "__file__", None)
    # ``python - <<EOF`` sets __file__ to "<stdin>", which spawn cannot import.
    return bool(path) and os.path.exists(path)


def _apply_iteration_budget(
    config: FractalConfig,
    polynomial: Polynomial,
    trap: OrbitTrap | None,
    progress: bool = False,
    saturation_limit: float = 0.05,
) -> FractalConfig:
    """Grow the iteration budget while most pixels still hit the cap.

    Deep zooms (and windows parked on the Julia set) need far more iterations
    than the default: with a fixed budget every pixel returns the "did not
    converge" value and the frame renders as one flat colour. Doubling stops as
    soon as 95 % of the pixels converge, or at ``max_iterations``.
    """
    if not config.auto_iterations:
        return config

    # Deep zooms need a few hundred extra iterations, not a few dozen: the
    # default ceiling is therefore generous (and still bounded).
    ceiling = config.max_iterations or max(config.iterations * 8, 1024)
    ceiling = min(max(ceiling, config.iterations), 8192)
    budget = config.iterations
    while True:
        probe = config.derive(iterations=budget)
        sample_size = min(96, config.size)
        probe = probe.derive(size=sample_size)
        iterations, _, _, _, _, _ = _solve_tile(pixel_grid(sample_size, probe.domain), polynomial, probe, trap)
        saturation = np.mean(iterations > budget).item()
        if progress:
            print(f"iteration budget {budget}: {100 * (1 - saturation):.1f}% converged")
        if saturation <= saturation_limit or budget >= ceiling:
            return config.derive(iterations=budget)
        budget = min(budget * 2, ceiling)


def _solve_tile_star(arguments) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Picklable wrapper for :func:`_solve_tile` (module-level for multiprocessing)."""
    return _solve_tile(*arguments)


def _load_trap(config: FractalConfig) -> OrbitTrap | None:
    from .orbit import load_orbit_trap

    base = Path(getattr(config, "base_dir", os.getcwd()))
    return load_orbit_trap(config.orbit_trap, base)


# --------------------------------------------------------------------------- #
# polynomial selection
# --------------------------------------------------------------------------- #
def _generated_area(polynomial: Polynomial, config: FractalConfig, sample: int = 100) -> float:
    """Share of the sampled plane that is inside the fractal body."""
    steps = min(config.iterations, 50)
    probe_config = config.derive(size=sample, iterations=steps)
    grid = pixel_grid(sample, config.domain)
    iterations, _, _, _, _, _ = _solve_tile(grid, polynomial, probe_config, None)
    return (iterations > iterations.mean()).mean().item()


def choose_polynomial(config: FractalConfig, trap: OrbitTrap | None = None, *, progress: bool = False) -> Polynomial:
    """Build the configured polynomial, resampling random ones until they fill the frame.

    A random polynomial can produce an almost empty or an almost uniform image;
    the original code rejected such samples based on the converged area. The
    acceptance window is configurable through ``polynomial.min_area``/``max_area``.
    """
    settings = config.polynomial
    rng = np.random.default_rng(settings.seed)
    if settings.mode != "random":
        return Polynomial.from_config(settings, rng)

    best: tuple[float, Polynomial] | None = None
    for attempt in range(1, settings.attempts + 1):
        candidate = Polynomial.random(settings.max_degree, rng)
        area = _generated_area(candidate, config)
        if settings.min_area <= area <= settings.max_area:
            if progress:
                print(f"polynomial accepted on attempt {attempt} (area {area:.2f})")
            return candidate
        # Keep the least degenerate sample in case no attempt is in range.
        penalty = min(abs(area - settings.min_area), abs(area - settings.max_area))
        if best is None or penalty < best[0]:
            best = (penalty, candidate)

    if progress and best is not None:
        print("polynomial conditioning window not reached; using closest sample")
    return best[1] if best is not None else Polynomial.random(settings.max_degree, rng)
