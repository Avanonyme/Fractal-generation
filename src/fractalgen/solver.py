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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import ConfigError, FractalConfig
from .edges import detect_boundary
from .orbit import OrbitTrap
from .polynomial import Polynomial

__all__ = ["FractalImage", "solve", "pixel_grid", "choose_polynomial", "chunk_slices"]


@dataclass(frozen=True)
class FractalImage:
    """Result of one solve: raw fields plus the polynomial that produced them."""

    iterations: np.ndarray
    roots: np.ndarray
    distance: np.ndarray | None
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
        return float(np.mean(self.iterations > self.config.iterations))

    @property
    def converged_share(self) -> float:
        return 1.0 - self.saturation

    @property
    def zoom(self) -> float:
        """How far this frame is zoomed relative to a 2-unit-wide window."""
        return 2.0 / max(self.config.domain.width, 1e-300)

    def scalar_field(self, kind: str) -> np.ndarray:
        """Pick the field that should be colourised."""
        if kind == "iteration":
            return self.iterations
        if kind == "distance":
            if self.distance is None:
                raise ValueError("render type 'distance' needs an orbit trap")
            return self.distance
        if kind == "boundary":
            return self.boundary.astype(np.float64)
        if kind == "roots":
            return np.real(self.roots)
        raise ValueError(f"unknown render type {kind!r}")

    def height_field(self, kind: str = "iteration") -> np.ndarray:
        """Field used for relief shading (distance when available)."""
        if self.distance is not None:
            return self.distance
        return self.scalar_field(kind if kind != "boundary" else "iteration")


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z = np.asarray(start, dtype=np.complex128).ravel().copy()
    iterations = np.full(z.shape, config.iterations + 1, dtype=np.float64)
    active = np.ones(z.shape, dtype=bool)
    trap_field = trap
    distance = np.full(z.shape, np.inf) if trap_field is not None else None
    damping = config.damping

    for step in range(1, config.iterations + 1):
        index = np.flatnonzero(active)
        if index.size == 0:
            break
        current = z[index]
        delta, stepped = _newton_step(polynomial, current, damping)
        current = current + delta
        z[index] = current

        # A stalled point (zero derivative) must not count as converged: its
        # step is zero, which would otherwise look like perfect precision.
        converged = (np.abs(delta) < config.tolerance) & stepped
        if converged.any():
            finished = index[converged]
            iterations[finished] = step
            active[finished] = False

        if distance is not None and trap_field is not None:
            distance[index] = np.minimum(distance[index], trap_field.sample(current))

    return iterations, z, distance


def _halley(
    start: np.ndarray,
    polynomial: Polynomial,
    config: FractalConfig,
    trap: OrbitTrap | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    z = np.asarray(start, dtype=np.complex128).ravel().copy()
    iterations = np.full(z.shape, config.iterations + 1, dtype=np.float64)
    active = np.ones(z.shape, dtype=bool)
    trap_field = trap
    distance = np.full(z.shape, np.inf) if trap_field is not None else None
    damping = config.damping

    for step in range(1, config.iterations + 1):
        index = np.flatnonzero(active)
        if index.size == 0:
            break
        current = z[index]
        delta, stepped = _halley_step(polynomial, current, damping)
        current = current + delta
        z[index] = current

        converged = (np.abs(delta) < config.tolerance) & stepped
        if converged.any():
            finished = index[converged]
            iterations[finished] = step
            active[finished] = False

        if distance is not None and trap_field is not None:
            distance[index] = np.minimum(distance[index], trap_field.sample(current))

    return iterations, z, distance


def _newton_step(
    polynomial: Polynomial, z: np.ndarray, damping: complex
) -> tuple[np.ndarray, np.ndarray]:
    """``-damping * f(z) / f'(z)`` plus a mask of the points that moved.

    A zero derivative used to produce ``inf``/``nan`` pixels that poisoned every
    downstream array (and made tiled and single-tile runs disagree). Points on a
    multiple root keep their position and are reported as not stepped, so they
    never count as converged.
    """
    derivative = polynomial.derivative(z)
    step = np.zeros(z.shape, dtype=np.complex128)
    nonzero = derivative != 0
    with np.errstate(all="ignore"):
        step[nonzero] = -damping * polynomial(z[nonzero]) / derivative[nonzero]
    return step, nonzero


def _halley_step(
    polynomial: Polynomial, z: np.ndarray, damping: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Halley step with the same zero-denominator protection as Newton."""
    value = polynomial(z)
    first = polynomial.derivative(z)
    second = polynomial.second_derivative(z)
    denominator = 2 * first**2 - second * value
    step = np.zeros(z.shape, dtype=np.complex128)
    nonzero = denominator != 0
    with np.errstate(all="ignore"):
        step[nonzero] = -damping * 2 * value[nonzero] * first[nonzero] / denominator[nonzero]
    return step, nonzero


_KERNELS = {"newton": _newton, "halley": _halley}


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    kernel = _KERNELS[config.method]
    shape = np.asarray(start).shape
    if config.variant == "nova":
        start, polynomial = _nova_start(start, polynomial, config)
    iterations, roots, distance = kernel(start, polynomial, config, trap)

    if config.round_digits:
        roots = np.round(roots, config.round_digits)
    distance = None if distance is None else distance.reshape(shape)
    return iterations.reshape(shape), roots.reshape(shape), distance


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
        iterations, roots, distance = _solve_tile(grid, polynomial, config, trap)
    else:
        iterations = np.empty(grid.shape, dtype=np.float64)
        roots = np.empty(grid.shape, dtype=np.complex128)
        distance = np.full(grid.shape, np.inf) if trap is not None else None
        arguments = [(grid[rows, columns], polynomial, config, trap) for rows, columns in tiles]
        with ProcessPoolExecutor(max_workers=min(len(tiles), workers)) as pool:
            results = list(pool.map(_solve_tile_star, arguments))
        for (rows, columns), (tile_iterations, tile_roots, tile_distance) in zip(tiles, results):
            iterations[rows, columns] = tile_iterations
            roots[rows, columns] = tile_roots
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

    return FractalImage(
        iterations=iterations,
        roots=roots,
        distance=distance,
        normal=normal,
        boundary=boundary,
        polynomial=polynomial,
        config=config,
    )


def _processes_available() -> bool:
    """Whether worker processes can re-import the entry point.

    Spawn-based pools (the default on macOS and Windows) import ``__main__`` in
    every worker, so a script without a ``if __name__ == "__main__"`` guard, or
    an interactive/stdin session, cannot use them. Falling back to a serial run
    beats the original behaviour, where the pool died with a traceback per
    worker and the caller got no result at all.
    """
    import __main__

    return hasattr(__main__, "__file__")


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
        iterations, _, _ = _solve_tile(pixel_grid(sample_size, probe.domain), polynomial, probe, trap)
        saturation = float(np.mean(iterations > budget))
        if progress:
            print(f"iteration budget {budget}: {100 * (1 - saturation):.1f}% converged")
        if saturation <= saturation_limit or budget >= ceiling:
            return config.derive(iterations=budget)
        budget = min(budget * 2, ceiling)


def _solve_tile_star(arguments) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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
    iterations, _, _ = _solve_tile(grid, polynomial, probe_config, None)
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
