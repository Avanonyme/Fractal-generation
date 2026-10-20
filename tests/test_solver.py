import numpy as np
import pytest

from fractalgen import FractalConfig, Polynomial, solve
from fractalgen.config import Domain
from fractalgen.solver import chunk_slices, pixel_grid


def _polynomial() -> Polynomial:
    return Polynomial.from_roots([1, -1, 1j, -1j])


def test_solve_finds_the_roots():
    config = FractalConfig(size=64, iterations=40)
    result = solve(config, _polynomial())
    assert result.iterations.shape == (64, 64)
    known = np.array([1, -1, 1j, -1j])
    converged = result.roots[result.iterations <= config.iterations]
    assert converged.size > 0
    for root in np.unique(np.round(converged, 4))[:4]:
        assert np.min(np.abs(known - root)) < 0.2


def test_iterations_are_bounded():
    config = FractalConfig(size=48, iterations=10)
    result = solve(config, _polynomial())
    assert result.iterations.min() >= 1
    # The budget may have grown (auto_iterations); it must match the report.
    assert result.iterations.max() <= result.config.iterations + 1
    assert 0.0 <= result.saturation <= 1.0


def test_auto_iterations_can_be_disabled():
    config = FractalConfig(size=48, iterations=8, auto_iterations=False)
    result = solve(config, _polynomial())
    assert result.config.iterations == 8
    assert result.iterations.max() <= 9


def test_boundary_mask_is_reasonable():
    result = solve(FractalConfig(size=96, iterations=30), _polynomial())
    share = result.boundary.mean()
    assert 0 < share < 0.5


def test_chunked_matches_single_tile():
    config = FractalConfig(size=64, iterations=20)
    single = solve(config, _polynomial(), workers=1)
    chunked = solve(config, _polynomial(), workers=4)
    assert np.array_equal(single.iterations, chunked.iterations)
    assert np.allclose(single.roots, chunked.roots)


def test_halley_runs():
    config = FractalConfig(size=32, method="halley", iterations=20)
    result = solve(config, _polynomial())
    assert np.isfinite(result.iterations).all()


def test_nova_variant_runs_and_is_seeded():
    config = FractalConfig(size=48, variant="nova", iterations=20)
    config = config.derive(polynomial=config.polynomial.__class__(mode="random", max_degree=5, seed=4))
    first = solve(config, None)
    second = solve(config, None)
    assert np.array_equal(first.iterations, second.iterations)


def test_distance_requires_orbit_trap():
    result = solve(FractalConfig(size=32), _polynomial())
    with pytest.raises(ValueError, match="orbit trap"):
        result.scalar_field("distance")


def test_chunk_slices_cover_the_grid():
    tiles = chunk_slices(256, workers=8, minimum_tile=64)
    covered = np.zeros((256, 256), dtype=int)
    for rows, columns in tiles:
        covered[rows, columns] += 1
    assert covered.min() == 1 and covered.max() == 1


def test_deep_zoom_keeps_detail():
    """A deep window used to saturate the budget and render one flat colour."""
    polynomial = _polynomial()
    coarse = solve(FractalConfig(size=128, iterations=12), polynomial)
    # A pixel that never converged is on the Julia set: the hardest case for a
    # deep zoom, and exactly where the old code went flat.
    stuck = np.argwhere(coarse.iterations > 12)
    assert stuck.size > 0
    pick = stuck[np.argmin(np.hypot(stuck[:, 1] - 64, stuck[:, 0] - 64))]
    centre = complex(-1 + 2 * pick[1] / 128, -1 + 2 * pick[0] / 128)

    for exponent in (6, 10):
        width = 2.0 * 10.0**-exponent
        domain = Domain(centre.real - width / 2, centre.real + width / 2, centre.imag - width / 2, centre.imag + width / 2)
        result = solve(FractalConfig(size=96, iterations=30, domain=domain), polynomial)
        unique = np.unique(result.iterations).size
        assert unique > 5, f"zoom 1e{exponent} collapsed to a flat colour ({unique} distinct value(s))"
        assert result.saturation < 0.5


def test_budget_grows_when_the_default_saturates():
    """With a tiny budget the frame saturates; auto_iterations must recover it."""
    polynomial = _polynomial()
    width = 2e-6
    centre = complex(-0.00390625, -0.00390625)
    domain = Domain(centre.real - width / 2, centre.real + width / 2, centre.imag - width / 2, centre.imag + width / 2)

    starved = solve(FractalConfig(size=64, iterations=30, auto_iterations=False, domain=domain), polynomial)
    assert starved.saturation > 0.5
    assert np.unique(starved.iterations).size == 1

    recovered = solve(FractalConfig(size=64, iterations=30, domain=domain), polynomial)
    assert recovered.config.iterations > 30
    assert recovered.saturation < 0.1
    assert np.unique(recovered.iterations).size > 5


def test_zoom_below_float64_resolution_is_reported():
    from fractalgen.config import ConfigError

    with pytest.raises(ConfigError, match="float64 resolution"):
        Domain(1.0 - 1e-17, 1.0, -1e-17, 1e-17)

    width = 1e-16
    domain = Domain(1.0 - width, 1.0, -width / 2, width / 2)
    assert not domain.can_resolve
    with pytest.raises(ConfigError, match="float64 resolution"):
        solve(FractalConfig(size=32, domain=domain), _polynomial())


def test_pixel_grid_bounds():
    grid = pixel_grid(10, FractalConfig().domain)
    assert np.real(grid).min() == pytest.approx(-1.0)
    assert np.imag(grid).max() < 1.0


def test_continuous_iteration_count_is_monotone_in_the_residual():
    """nu = n + log2(ln(u)/ln(eps)): u == eps gives exactly n, u < eps gives n..n+1."""
    from fractalgen.solver import _smooth_iteration

    tolerance = 1e-5
    exact = _smooth_iteration(7, np.array([tolerance]), tolerance, order=2)
    smaller = _smooth_iteration(7, np.array([1e-6]), tolerance, order=2)
    tiny = _smooth_iteration(7, np.array([1e-300]), tolerance, order=2)
    assert exact[0] == pytest.approx(7.0)
    assert 7.0 <= smaller[0] <= 8.0
    assert 7.0 <= tiny[0] <= 8.0
    assert smaller[0] < tiny[0]


def test_shading_height_is_smooth():
    """The default height must not be the integer counter (that embosses terraces)."""
    from fractalgen.config import ShadingConfig

    result = solve(FractalConfig(size=64, iterations=20), _polynomial())
    smooth = result.height_field("iteration", "auto")
    assert np.unique(np.round(smooth, 6)).size > np.unique(result.iterations).size
    assert ShadingConfig().height == "auto"
    assert np.unique(np.round(result.height_field("iteration", "iteration-steps"), 6)).size == np.unique(
        result.iterations
    ).size


def test_root_colouring_is_piecewise_constant():
    result = solve(FractalConfig(size=64, iterations=20), _polynomial())
    index = result.root_index()
    assert index.max() <= len(result.polynomial.roots)
    assert index.min() >= 0
