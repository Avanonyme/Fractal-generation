import numpy as np
import pytest

from fractalgen import Polynomial
from fractalgen.config import ConfigError, PolynomialConfig
from fractalgen.solver import choose_polynomial, pixel_grid
from fractalgen.solver import FractalConfig


def test_from_roots_matches_numpy():
    roots = [1, -1, 1j, -1j]
    polynomial = Polynomial.from_roots(list(roots))
    reference = np.poly(roots)
    assert np.allclose(polynomial.coefficients[::-1], reference)


def test_evaluation_matches_numpy_polyval():
    rng = np.random.default_rng(3)
    coefficients = rng.normal(size=6) + 1j * rng.normal(size=6)
    coefficients[-1] = 1 + 0j
    polynomial = Polynomial.from_coefficients(list(coefficients))
    z = rng.normal(size=8) + 1j * rng.normal(size=8)
    assert np.allclose(polynomial(z), np.polyval(coefficients[::-1], z))


def test_derivatives_are_consistent():
    polynomial = Polynomial.from_roots([1, -2, 3j])
    z = np.array([0.3 + 0.1j, -1.2 + 0.4j])
    epsilon = 1e-7
    numeric = (polynomial(z + epsilon) - polynomial(z - epsilon)) / (2 * epsilon)
    assert np.allclose(polynomial.derivative(z), numeric, rtol=1e-5, atol=1e-6)
    numeric_second = (polynomial.derivative(z + epsilon) - polynomial.derivative(z - epsilon)) / (2 * epsilon)
    assert np.allclose(polynomial.second_derivative(z), numeric_second, rtol=1e-4, atol=1e-4)


def test_roots_are_recovered():
    roots = [1 + 0j, -1 + 0j, 2j - 0.5]
    polynomial = Polynomial.from_roots(list(roots))
    found = np.sort_complex(polynomial.roots)
    assert np.allclose(found, np.sort_complex(roots))


def test_rejects_degenerate_polynomials():
    with pytest.raises(ConfigError):
        Polynomial.from_coefficients([1.0])
    with pytest.raises(ConfigError):
        Polynomial.from_coefficients([1.0, 0.0])


def test_random_polynomial_respects_degree_bound():
    rng = np.random.default_rng(0)
    for _ in range(20):
        polynomial = Polynomial.random(6, rng)
        assert 4 <= polynomial.coefficients.size <= 6


def test_family_shapes():
    polynomial = Polynomial.from_roots([1, -1, 1j])
    rng = np.random.default_rng(1)
    family = polynomial.family(np.linspace(-1, 1, 5), rng)
    assert family.shape == (5, polynomial.coefficients.size)
    broadcast = polynomial.family(np.linspace(-1, 1, 5), rng, perturb=False)
    assert np.allclose(broadcast[0], polynomial.coefficients)


def test_taylor_mode_builds_usable_polynomial():
    polynomial = Polynomial.from_taylor("sin", 7, 2.0)
    assert polynomial.degree >= 3
    assert abs(polynomial(0.0)) < 1e-9


def test_from_config_modes():
    rng = np.random.default_rng(0)
    roots = Polynomial.from_config(PolynomialConfig(mode="roots", roots=(1, -1)), rng)
    assert np.allclose(np.sort_complex(roots.roots), np.array([-1, 1]))
    coefs = Polynomial.from_config(PolynomialConfig(mode="coefs", coefs=(1, 0, 1)), rng)
    assert np.allclose(coefs.roots, np.array([1j, -1j]))


def test_pixel_grid_orientation():
    grid = pixel_grid(4, FractalConfig().domain)
    assert grid.shape == (4, 4)
    assert np.imag(grid)[0, 0] == np.imag(grid)[0, -1]
    assert np.real(grid)[0, 0] < np.real(grid)[0, -1]


def test_choice_is_deterministic_for_a_seed():
    config = FractalConfig(size=64, polynomial=PolynomialConfig(mode="random", max_degree=5, seed=11))
    first = choose_polynomial(config)
    second = choose_polynomial(config)
    assert np.allclose(first.coefficients, second.coefficients)
