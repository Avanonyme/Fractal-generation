"""Polynomial algebra used by the root finders.

A :class:`Polynomial` stores ascending complex coefficients and evaluates
itself with Horner's rule, which is both faster than the original
``sum(coefs[k] * z**k ...)`` formulation and numerically better behaved.

All evaluation helpers are vectorised over arbitrary numpy arrays, so they can
be used directly with the chunked solver or on their own::

    from fractalgen import Polynomial

    poly = Polynomial.from_roots([1, -1, 1j, -1j])
    poly(0.5 + 0.25j), poly.derivative(1j), poly.roots
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import ConfigError, PolynomialConfig

__all__ = ["Polynomial"]

def _series(function: str, degree: int) -> np.ndarray:
    """Taylor coefficients (ascending) of a named analytic function around 0."""
    import math

    if function == "tanh":
        return _quotient_series(_sinh(degree), _cosh(degree))
    coefficients = np.zeros(degree + 1, dtype=np.complex128)
    for power in range(degree + 1):
        if function == "sin":
            value = 0.0 if power % 2 == 0 else (-1) ** ((power - 1) // 2) / math.factorial(power)
        elif function == "cos":
            value = 0.0 if power % 2 else (-1) ** (power // 2) / math.factorial(power)
        elif function == "exp":
            value = 1.0 / math.factorial(power)
        elif function == "arctan":
            value = 0.0 if power % 2 == 0 else (-1) ** ((power - 1) // 2) / power
        elif function == "sqrt":
            value = _binomial_half(power)
        else:
            raise ConfigError(f"polynomial.function: unknown function {function!r}")
        coefficients[power] = value
    return coefficients


def _sinh(degree: int) -> list[float]:
    import math

    return [0.0 if power % 2 == 0 else 1.0 / math.factorial(power) for power in range(degree + 1)]


def _cosh(degree: int) -> list[float]:
    import math

    return [1.0 / math.factorial(power) if power % 2 == 0 else 0.0 for power in range(degree + 1)]


def _quotient_series(numerator: list[float], denominator: list[float]) -> np.ndarray:
    """Power-series division, valid because ``denominator[0] == 1``."""
    length = len(numerator)
    quotient = [0.0] * length
    for index in range(length):
        remainder = numerator[index] - sum(
            denominator[step] * quotient[index - step] for step in range(1, index + 1)
        )
        quotient[index] = remainder / denominator[0]
    return np.asarray(quotient, dtype=np.complex128)


def _binomial_half(power: int) -> float:
    """Coefficient of ``z**power`` in ``sqrt(1 + z)``."""
    import math

    value = 1.0
    for step in range(power):
        value *= 0.5 - step
    return value / math.factorial(power)


def _horner(coefficients: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Evaluate ``sum(coefficients[k] * z**k)`` from the highest power down."""
    result = np.zeros(z.shape, dtype=np.complex128)
    for coefficient in coefficients[::-1]:
        result = result * z + coefficient
    return result


@dataclass(frozen=True)
class Polynomial:
    """An ascending-coefficient complex polynomial."""

    coefficients: np.ndarray
    roots_cache: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.complex128).ravel()
        if coefficients.size < 2:
            raise ConfigError("polynomial: needs at least 2 coefficients")
        if coefficients[-1] == 0:
            raise ConfigError("polynomial: leading coefficient must be non-zero")
        object.__setattr__(self, "coefficients", coefficients)

    # ------------------------------------------------------------------ #
    # evaluation
    # ------------------------------------------------------------------ #
    @property
    def degree(self) -> int:
        return self.coefficients.size - 1

    def __call__(self, z: np.ndarray | complex) -> np.ndarray:
        return _horner(self.coefficients, np.asarray(z, dtype=np.complex128))

    def derivative_coefficients(self) -> np.ndarray:
        powers = np.arange(1, self.coefficients.size)
        return self.coefficients[1:] * powers

    def second_derivative_coefficients(self) -> np.ndarray:
        powers = np.arange(2, self.coefficients.size)
        return self.coefficients[2:] * powers * (powers - 1)

    def derivative(self, z: np.ndarray | complex) -> np.ndarray:
        return _horner(self.derivative_coefficients(), np.asarray(z, dtype=np.complex128))

    def second_derivative(self, z: np.ndarray | complex) -> np.ndarray:
        return _horner(self.second_derivative_coefficients(), np.asarray(z, dtype=np.complex128))

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_coefficients(cls, coefficients: Sequence[complex]) -> "Polynomial":
        return cls(np.asarray(coefficients, dtype=np.complex128))

    @classmethod
    def from_roots(cls, roots: Sequence[complex]) -> "Polynomial":
        """Build a monic polynomial from its roots (ascending coefficients)."""
        roots_array = np.asarray(roots, dtype=np.complex128).ravel()
        if roots_array.size == 0:
            raise ConfigError("polynomial: needs at least one root")
        coefficients = np.poly(roots_array)  # descending, leading coefficient 1
        return cls(np.ascontiguousarray(coefficients[::-1]))

    @classmethod
    def from_taylor(cls, function: str, degree: int, interval: float) -> "Polynomial":
        """Taylor polynomial of a named analytic function around 0.

        ``interval`` scales the coefficients, so a wider interval produces a
        polynomial with the same roots but a coarser approximation.
        """
        coefficients = _series(function, degree)
        if interval and interval != 1.0:
            coefficients = coefficients * np.power(interval, -np.arange(coefficients.size))
        # Drop negligible high-order terms so the leading coefficient is usable.
        scale = np.max(np.abs(coefficients)) or 1.0
        keep = np.flatnonzero(np.abs(coefficients) > 1e-12 * scale)
        coefficients = coefficients[: keep[-1] + 1] if keep.size else coefficients
        return cls(coefficients)

    @classmethod
    def random(cls, max_degree: int, rng: np.random.Generator, magnitude: float = 10.0) -> "Polynomial":
        """Random polynomial with 4..``max_degree`` coefficients."""
        term_count = operator.index(rng.integers(4, max(max_degree, 4) + 1))
        coefficients = rng.uniform(-magnitude, magnitude, size=term_count) + 1j * rng.uniform(
            -magnitude, magnitude, size=term_count
        )
        if coefficients[-1] == 0:
            coefficients[-1] = magnitude
        return cls(coefficients)

    @classmethod
    def from_config(cls, config: PolynomialConfig, rng: np.random.Generator) -> "Polynomial":
        if config.mode == "random":
            return cls.random(config.max_degree, rng)
        if config.mode == "roots":
            return cls.from_roots(config.roots)
        if config.mode == "coefs":
            return cls.from_coefficients(config.coefs)
        return cls.from_taylor(config.function, config.degree or 10, config.interval)

    # ------------------------------------------------------------------ #
    # derived polynomials
    # ------------------------------------------------------------------ #
    @property
    def roots(self) -> np.ndarray:
        """All roots, computed with the companion-matrix eigenvectors."""
        roots = self.roots_cache
        if roots is None:
            roots = np.sort_complex(np.roots(self.coefficients[::-1]))
            object.__setattr__(self, "roots_cache", roots)
        return roots

    def critical_points(self) -> np.ndarray:
        """Roots of the first derivative (used as Nova starting points)."""
        if self.degree < 2:
            return np.zeros(1, dtype=np.complex128)
        return np.roots(self.derivative_coefficients()[::-1])

    def family(self, c: np.ndarray, rng: np.random.Generator, *, perturb: bool = True) -> np.ndarray:
        """Per-pixel coefficient matrix for the ``nova`` variant.

        ``c`` is a flat array of pixel values; the result has shape
        ``(c.size, len(coefficients))``. With ``perturb`` each coefficient is
        randomly either kept or shifted by ``c`` plus a small random offset
        (the original RFA "c-mapping"). The transformation is fully vectorised
        - the previous implementation looped in Python over every pixel.
        """
        c = np.asarray(c, dtype=np.complex128).ravel()
        base = self.coefficients
        if not perturb:
            return np.broadcast_to(base, (c.size, base.size)).copy()

        term_count = base.size
        mask = rng.random(term_count) >= 0.7
        if not mask.any():
            mask[0] = True
            mask[-1] = True
        signs = rng.choice([1.0, -1.0], size=term_count)
        offsets = rng.random(term_count)

        shifted = c[:, None] + (signs * offsets)[None, :]
        return np.where(mask[None, :], shifted * base[None, :], base[None, :])

    # ------------------------------------------------------------------ #
    # convenience
    # ------------------------------------------------------------------ #
    def to_config(self) -> PolynomialConfig:
        return PolynomialConfig(mode="coefs", coefs=tuple(self.coefficients.tolist()))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Polynomial(degree={self.degree})"
