"""Numeric coercion with configuration-grade error messages.

Both the standard library converters and numpy scalars raise on bad input; the
helpers below turn those failures into :class:`~fractalgen.config.ConfigError`
messages that name the offending field, so a typo in a JSON config reports
``fractal.size: expected an integer, got 'big'`` instead of a bare ``ValueError``.
"""

from __future__ import annotations

import numbers
import operator

import numpy as np

from .config import ConfigError

__all__ = ["as_int", "as_float", "as_complex"]


def as_int(value, where: str = "value") -> int:
    """Coerce to ``int``, rejecting booleans and non-integral values."""
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected an integer, got {value!r}")
    try:
        return operator.index(value)
    except TypeError as exc:
        return _int_from_real(value, where, exc)


def _int_from_real(value, where: str, cause: Exception) -> int:
    """Convert a real number that has no ``__index__`` (kept out of the handler)."""
    number = as_float(value, where)
    if number.is_integer():
        return round(number)
    raise ConfigError(f"{where}: expected an integer, got {value!r}") from cause


def as_float(value, where: str = "value") -> float:
    """Coerce to ``float``, rejecting booleans and non-numeric values."""
    if isinstance(value, bool):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    if isinstance(value, (str, bytes)) or not isinstance(value, (numbers.Real, np.floating, np.integer)):
        raise ConfigError(f"{where}: expected a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded above
        raise ConfigError(f"{where}: expected a number, got {value!r}") from exc


def as_complex(value, where: str = "value") -> complex:
    """Coerce ``1+2j``, a real number or a ``[real, imag]`` pair."""
    if isinstance(value, str):
        try:
            return complex(value)
        except ValueError as exc:
            raise ConfigError(f"{where}: invalid complex literal {value!r}") from exc
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return complex(as_float(value, where))
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 2:
        return complex(as_float(value[0], f"{where}[0]"), as_float(value[1], f"{where}[1]"))
    raise ConfigError(f"{where}: expected 1+2j, 1.5 or [real, imag]")
