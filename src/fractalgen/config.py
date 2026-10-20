"""Configuration model for fractal generation.

Everything that used to live in an unvalidated ``parameters`` dict is described
here by frozen dataclasses with explicit defaults, unit-aware names and strict
validation. A JSON document is only a serialisation of this model: keys are
checked, unknown keys are reported instead of silently ignored, and every
dataclass round-trips back to JSON via :func:`to_dict`.

Public entry points::

    load_config(path)              -> Config
    Config.from_dict(mapping)      -> Config
    config.to_dict()               -> dict (JSON-serialisable)
"""

from __future__ import annotations

import json
import math
import numbers
import operator
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

__all__ = [
    "ConfigError",
    "Domain",
    "PolynomialConfig",
    "OrbitTrapConfig",
    "MapConfig",
    "FractalConfig",
    "ShadingConfig",
    "ColourConfig",
    "ImageConfig",
    "AnimationConfig",
    "VideoConfig",
    "RuntimeConfig",
    "Config",
    "load_config",
    "save_config",
]


# --------------------------------------------------------------------------- #
# validation helpers
# --------------------------------------------------------------------------- #
class ConfigError(ValueError):
    """Raised when a configuration document is not usable."""


def _require_mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected an object, got {type(value).__name__}")
    return value


def _check_keys(data: Mapping[str, Any], allowed: Iterable[str], path: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if not unknown:
        return
    guessed = ", ".join(f"{key!r} -> {_closest(key, allowed)!r}" for key in unknown)
    raise ConfigError(f"{path}: unknown key(s) {guessed}")


def _closest(key: str, allowed: Iterable[str]) -> str | None:
    import difflib

    matches = difflib.get_close_matches(key, list(allowed), n=1, cutoff=0.55)
    return matches[0] if matches else None


def _as_int(value: Any, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ConfigError(f"{path}: expected an integer, got {value!r}")
    try:
        result = operator.index(value)
    except TypeError as exc:  # pragma: no cover - guarded by the check above
        raise ConfigError(f"{path}: expected an integer, got {value!r}") from exc
    if minimum is not None and result < minimum:
        raise ConfigError(f"{path}: must be >= {minimum}, got {result}")
    return result


def _as_float(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ConfigError(f"{path}: expected a number, got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - guarded by the check above
        raise ConfigError(f"{path}: expected a number, got {value!r}") from exc


def _as_bool(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{path}: expected true/false, got {value!r}")
    return value


def _as_str(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ConfigError(f"{path}: expected a string, got {value!r}")
    return value


def _as_choice(value: Any, choices: Sequence[str], path: str) -> str:
    text = _as_str(value, path).lower().replace("_", "-")
    if text not in choices:
        raise ConfigError(f"{path}: expected one of {list(choices)}, got {value!r}")
    return text


def _as_path(value: Any, path: str) -> Path | None:
    if value is None:
        return None
    return Path(_as_str(value, path)).expanduser()


def _complex_pairs(value: Any, path: str) -> list[complex]:
    """Accept ``1+2j``, ``[1, 2]`` (real, imag) or ``[1, 2]``-per-root lists."""
    if isinstance(value, str):
        return [complex(value)]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{path}: expected a number, 'a+bj' or a list")
    if len(value) == 2 and all(isinstance(item, numbers.Real) for item in value):
        return [_single_complex(value, path)]
    return [_single_complex(item, f"{path}[{i}]") for i, item in enumerate(value)]


def _single_complex(value: Any, path: str) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, str):
        try:
            return complex(value)
        except ValueError as exc:
            raise ConfigError(f"{path}: invalid complex literal {value!r}") from exc
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        return complex(_as_float(value, path))
    if isinstance(value, Sequence) and len(value) == 2:
        return complex(_as_float(value[0], f"{path}[0]"), _as_float(value[1], f"{path}[1]"))
    raise ConfigError(f"{path}: expected 1+2j, 1.5 or [real, imag]")


# --------------------------------------------------------------------------- #
# value objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class Domain:
    """Axis-aligned region of the complex plane handed to the solver."""

    x_min: float = -1.0
    x_max: float = 1.0
    y_min: float = -1.0
    y_max: float = 1.0

    def __post_init__(self) -> None:
        _check_span(self.x_min, self.x_max, "x")
        _check_span(self.y_min, self.y_max, "y")

    @property
    def center(self) -> complex:
        return complex((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @classmethod
    def from_dict(cls, value: Any, path: str = "fractal.domain") -> "Domain":
        if isinstance(value, Domain):
            return value
        if isinstance(value, Mapping):
            _check_keys(value, ["x", "y", "x_min", "x_max", "y_min", "y_max"], path)
            if "x" in value or "y" in value:
                x = value.get("x", [-1.0, 1.0])
                y = value.get("y", [-1.0, 1.0])
                return cls(*_span(x, f"{path}.x"), *_span(y, f"{path}.y"))
            return cls(
                _as_float(value["x_min"], f"{path}.x_min"),
                _as_float(value["x_max"], f"{path}.x_max"),
                _as_float(value["y_min"], f"{path}.y_min"),
                _as_float(value["y_max"], f"{path}.y_max"),
            )
        if isinstance(value, Sequence) and len(value) == 2:
            return cls(*_span(value[0], f"{path}[0]"), *_span(value[1], f"{path}[1]"))
        raise ConfigError(f"{path}: expected [[x0, x1], [y0, y1]] or {{'x': [...], 'y': [...]}}")

    @property
    def resolution(self) -> float:
        """Smallest representable step (in pixels) for this window.

        float64 cannot distinguish coordinates closer than ~1 ulp of the largest
        coordinate, so a window narrower than ``resolution * 16`` collapses into
        repeated pixels. That is the hard limit on how deep a zoom can go.
        """
        magnitude = max(1.0, abs(self.x_min), abs(self.x_max), abs(self.y_min), abs(self.y_max))
        return math.ulp(magnitude)

    @property
    def can_resolve(self) -> bool:
        return min(self.width, self.height) > 16 * self.resolution

    @property
    def max_zoom(self) -> float:
        """Zoom factor (relative to a 2-unit window) beyond which pixels repeat."""
        return max(1.0, 2.0 / (16 * self.resolution))

    def zoomed(self, factor: float, center: complex | None = None) -> "Domain":
        """Return a domain scaled by ``factor`` around ``center``."""
        center = self.center if center is None else center
        half_w = self.width * factor / 2
        half_h = self.height * factor / 2
        return Domain(center.real - half_w, center.real + half_w, center.imag - half_h, center.imag + half_h)

    def as_array(self) -> np.ndarray:
        return np.array([[self.x_min, self.x_max], [self.y_min, self.y_max]], dtype=float)

    def to_dict(self) -> dict[str, list[float]]:
        return {"x": [self.x_min, self.x_max], "y": [self.y_min, self.y_max]}


def _check_span(low: float, high: float, axis: str) -> None:
    if low < high:
        return
    ulp = math.ulp(max(1.0, abs(low), abs(high)))
    if high - low < 16 * ulp:
        # A collapsed window is a precision failure, not a typo: say so, with the
        # zoom limit the user has hit.
        raise ConfigError(
            f"fractal.domain: {axis} window [{low!r}, {high!r}] is below float64 resolution "
            f"(~{ulp:.1e}); the maximum zoom for this centre is about {2.0 / (16 * ulp):,.0f}x"
        )
    raise ConfigError(f"fractal.domain: {axis} min must be < max ({low} !< {high})")


def _span(value: Any, path: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or len(value) != 2:
        raise ConfigError(f"{path}: expected [min, max]")
    low, high = _as_float(value[0], f"{path}[0]"), _as_float(value[1], f"{path}[1]")
    if not low < high:
        raise ConfigError(f"{path}: min must be < max ({low} !< {high})")
    return low, high


@dataclass(frozen=True, slots=True)
class PolynomialConfig:
    """Which polynomial the root finder iterates on.

    ``mode`` decides the source of the polynomial:

    * ``random``  – random coefficients, resampled until the image is neither
      empty nor saturated (see ``min_area``/``max_area``/``attempts``).
    * ``roots``   – build from explicit roots (easiest to control).
    * ``coefs``   – raw ascending coefficients.
    * ``taylor``  – Taylor approximation of a named analytic function around 0.
    """

    mode: str = "random"
    max_degree: int = 8
    degree: int | None = None
    roots: tuple[complex, ...] = ()
    coefs: tuple[complex, ...] = ()
    function: str = "sin"
    interval: float = 1.0
    seed: int | None = None
    min_area: float = 0.10
    max_area: float = 0.60
    attempts: int = 25

    MODES = ("random", "roots", "coefs", "taylor")
    FUNCTIONS = ("sin", "cos", "tanh", "exp", "arctan", "sqrt")

    def __post_init__(self) -> None:
        if self.mode not in self.MODES:
            raise ConfigError(f"fractal.polynomial.mode: expected one of {list(self.MODES)}, got {self.mode!r}")
        if self.mode in ("roots", "coefs") and len(getattr(self, self.mode)) < 2:
            raise ConfigError(f"fractal.polynomial.{self.mode}: needs at least 2 entries")
        if self.mode == "taylor" and self.function not in self.FUNCTIONS:
            raise ConfigError(
                f"fractal.polynomial.function: expected one of {list(self.FUNCTIONS)}, got {self.function!r}"
            )
        if self.degree is not None and not 1 <= self.degree <= 30:
            raise ConfigError("fractal.polynomial.degree: must be within 1..30")
        if not 2 <= self.max_degree <= 30:
            raise ConfigError("fractal.polynomial.max_degree: must be within 2..30")
        if not 0.0 <= self.min_area < self.max_area <= 1.0:
            raise ConfigError("fractal.polynomial: require 0 <= min_area < max_area <= 1")

    @classmethod
    def from_dict(cls, value: Any, path: str = "fractal.polynomial") -> "PolynomialConfig":
        if isinstance(value, PolynomialConfig):
            return value
        data = _require_mapping(value, path)
        _check_keys(
            data,
            [
                "mode",
                "max_degree",
                "degree",
                "roots",
                "coefs",
                "function",
                "interval",
                "seed",
                "min_area",
                "max_area",
                "attempts",
            ],
            path,
        )
        default = cls()
        mode = _as_choice(data.get("mode", default.mode), cls.MODES, f"{path}.mode")
        roots: tuple[complex, ...] = ()
        coefs: tuple[complex, ...] = ()
        if "roots" in data:
            roots = _parse_complex_list(data["roots"], f"{path}.roots")
        if "coefs" in data:
            coefs = _parse_complex_list(data["coefs"], f"{path}.coefs")
        if mode == "roots" and not roots:
            raise ConfigError(f"{path}.roots: required when mode = 'roots'")
        if mode == "coefs" and not coefs:
            raise ConfigError(f"{path}.coefs: required when mode = 'coefs'")
        return cls(
            mode=mode,
            max_degree=_as_int(data.get("max_degree", default.max_degree), f"{path}.max_degree", minimum=2),
            degree=None if data.get("degree") is None else _as_int(data["degree"], f"{path}.degree", minimum=1),
            roots=roots,
            coefs=coefs,
            function=_as_choice(data.get("function", default.function), cls.FUNCTIONS, f"{path}.function"),
            interval=_as_float(data.get("interval", default.interval), f"{path}.interval"),
            seed=None if data.get("seed") is None else _as_int(data["seed"], f"{path}.seed", minimum=0),
            min_area=_as_float(data.get("min_area", default.min_area), f"{path}.min_area"),
            max_area=_as_float(data.get("max_area", default.max_area), f"{path}.max_area"),
            attempts=_as_int(data.get("attempts", default.attempts), f"{path}.attempts", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"mode": self.mode, "seed": self.seed}
        if self.mode == "random":
            data.update(max_degree=self.max_degree, min_area=self.min_area, max_area=self.max_area, attempts=self.attempts)
        if self.mode == "roots":
            data["roots"] = [[item.real, item.imag] for item in self.roots]
        if self.mode == "coefs":
            data["coefs"] = [[item.real, item.imag] for item in self.coefs]
        if self.mode == "taylor":
            data.update(function=self.function, degree=self.degree or 10, interval=self.interval)
        return data


def _parse_complex_list(value: Any, path: str) -> tuple[complex, ...]:
    if isinstance(value, str):
        return (_single_complex(value, path),)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{path}: expected a list of roots/coefficients")
    return tuple(_single_complex(item, f"{path}[{i}]") for i, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class OrbitTrapConfig:
    """Optional orbit trap: distance field of a bitmap injected during iteration.

    ``image`` is a path (relative paths resolve against the config file's
    directory) to a black-and-white image; the shape is extracted from it and a
    Euclidean distance map is precomputed once. ``distance_mode`` selects the
    ``get_distance`` variant (0-7, see :mod:`fractalgen.orbit`).
    """

    image: Path | None = None
    size: int = 800
    distance_mode: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.distance_mode <= 7:
            raise ConfigError("fractal.orbit_trap.distance_mode: must be within 0..7")
        if self.size < 16:
            raise ConfigError("fractal.orbit_trap.size: must be >= 16")

    @property
    def enabled(self) -> bool:
        return self.image is not None

    @classmethod
    def from_dict(cls, value: Any, path: str = "fractal.orbit_trap") -> "OrbitTrapConfig":
        if isinstance(value, OrbitTrapConfig):
            return value
        data = _require_mapping(value, path)
        _check_keys(data, ["image", "size", "distance_mode"], path)
        default = cls()
        return cls(
            image=_as_path(data.get("image"), f"{path}.image"),
            size=_as_int(data.get("size", default.size), f"{path}.size", minimum=16),
            distance_mode=_as_int(data.get("distance_mode", default.distance_mode), f"{path}.distance_mode"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"image": None if self.image is None else str(self.image), "size": self.size, "distance_mode": self.distance_mode}


@dataclass(frozen=True, slots=True)
class MapConfig:
    """Per-pixel modulation of the iteration by a raster image.

    The image is sampled at each pixel's *starting position* and scales the
    damping of the root finder, so the local dynamics (and therefore the basin
    pattern) follow the picture. This is the "map modifier" that makes a source
    image readable inside the fractal; an orbit trap alone echoes the shape but
    does not reproduce it, because a Newton orbit collapses onto a root long
    before it can sample much of the trap.

    ``strength`` is the half-range of the damping multiplier: the damping used at
    a pixel of luminance ``v`` is ``damping * (1 + strength * (v - 0.5))``, so
    ``0`` disables the modulation and ``2`` roughly doubles/halves it.
    """

    image: Path | None = None
    strength: float = 0.0
    size: int = 96
    #: ``damping`` scales the Newton damping per pixel; it is the mode that makes
    #: a picture actually readable inside the fractal (the basin pattern follows
    #: the image). ``warp`` displaces the starting point along the image gradient:
    #: linear and gentle, it nudges the pattern without flattening anything, so a
    #: photo only faintly shows through.
    mode: str = "damping"
    #: 0..1: how strongly the frame's brightness follows the image itself. The
    #: dynamical coupling above reshapes the basins, but a Newton orbit only ever
    #: visits a small part of the plane, so the picture is faint there; blending
    #: the raster into the frame (in image space) is what makes it readable.
    blend: float = 0.0

    MODES = ("warp", "damping")

    def __post_init__(self) -> None:
        if self.mode not in self.MODES:
            raise ConfigError(f"fractal.map.mode: expected one of {list(self.MODES)}, got {self.mode!r}")
        if self.strength < 0:
            raise ConfigError("fractal.map.strength: must be >= 0")
        if not 0.0 <= self.blend <= 1.0:
            raise ConfigError("fractal.map.blend: must be within 0..1")
        if self.size < 8:
            raise ConfigError("fractal.map.size: must be >= 8")

    @property
    def enabled(self) -> bool:
        """True when the image does anything: dynamical map or image-space blend."""
        return self.image is not None and (self.strength > 0 or self.blend > 0)

    @classmethod
    def from_dict(cls, value: Any, path: str = "fractal.map") -> "MapConfig":
        if isinstance(value, MapConfig):
            return value
        data = _require_mapping(value, path)
        _check_keys(data, ["image", "strength", "size", "mode", "blend"], path)
        default = cls()
        return cls(
            image=_as_path(data.get("image"), f"{path}.image"),
            strength=_as_float(data.get("strength", default.strength), f"{path}.strength"),
            size=_as_int(data.get("size", default.size), f"{path}.size", minimum=8),
            mode=_as_choice(data.get("mode", default.mode), cls.MODES, f"{path}.mode"),
            blend=_as_float(data.get("blend", default.blend), f"{path}.blend"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image": None if self.image is None else str(self.image),
            "strength": self.strength,
            "size": self.size,
            "mode": self.mode,
            "blend": self.blend,
        }


@dataclass(frozen=True, slots=True)
class FractalConfig:
    """Numerical description of the fractal itself (no rendering concerns)."""

    size: int = 800
    domain: Domain = field(default_factory=Domain)
    method: str = "newton"
    variant: str = "standard"
    iterations: int = 30
    #: Raise the iteration budget automatically when too many pixels exhaust it.
    #: Without this a deep zoom renders as one flat colour, because every pixel
    #: hits the cap and gets the "did not converge" value.
    auto_iterations: bool = True
    #: Ceiling for the automatic budget (``None`` = ``8 * iterations``).
    max_iterations: int | None = None
    tolerance: float = 1e-5
    damping: complex = complex(1.0, 0.0)
    round_digits: int = 4
    polynomial: PolynomialConfig = field(default_factory=PolynomialConfig)
    orbit_trap: OrbitTrapConfig = field(default_factory=OrbitTrapConfig)
    map: MapConfig = field(default_factory=MapConfig)

    METHODS = ("newton", "halley")
    VARIANTS = ("standard", "nova")

    def __post_init__(self) -> None:
        if self.method not in self.METHODS:
            raise ConfigError(f"fractal.method: expected one of {list(self.METHODS)}, got {self.method!r}")
        if self.variant not in self.VARIANTS:
            raise ConfigError(f"fractal.variant: expected one of {list(self.VARIANTS)}, got {self.variant!r}")
        if self.size < 16:
            raise ConfigError("fractal.size: must be >= 16")
        if self.iterations < 1:
            raise ConfigError("fractal.iterations: must be >= 1")
        if self.max_iterations is not None and self.max_iterations < self.iterations:
            raise ConfigError("fractal.max_iterations: must be >= fractal.iterations")
        if not self.tolerance > 0:
            raise ConfigError("fractal.tolerance: must be > 0")
        if not 0 <= self.round_digits <= 12:
            raise ConfigError("fractal.round_digits: must be within 0..12")

    @classmethod
    def from_dict(cls, value: Any, path: str = "fractal") -> "FractalConfig":
        if isinstance(value, FractalConfig):
            return value
        data = _require_mapping(value, path)
        _check_keys(
            data,
            [
                "size",
                "domain",
                "method",
                "variant",
                "iterations",
                "auto_iterations",
                "max_iterations",
                "tolerance",
                "damping",
                "round_digits",
                "polynomial",
                "orbit_trap",
                "map",
            ],
            path,
        )
        default = cls()
        return cls(
            size=_as_int(data.get("size", default.size), f"{path}.size", minimum=16),
            domain=Domain.from_dict(data.get("domain", default.domain), f"{path}.domain"),
            method=_as_choice(data.get("method", default.method), cls.METHODS, f"{path}.method"),
            variant=_as_choice(data.get("variant", default.variant), cls.VARIANTS, f"{path}.variant"),
            iterations=_as_int(data.get("iterations", default.iterations), f"{path}.iterations", minimum=1),
            auto_iterations=_as_bool(data.get("auto_iterations", default.auto_iterations), f"{path}.auto_iterations"),
            max_iterations=(
                None
                if data.get("max_iterations") is None
                else _as_int(data["max_iterations"], f"{path}.max_iterations", minimum=1)
            ),
            tolerance=_as_float(data.get("tolerance", default.tolerance), f"{path}.tolerance"),
            damping=_single_complex(data.get("damping", default.damping), f"{path}.damping"),
            round_digits=_as_int(data.get("round_digits", default.round_digits), f"{path}.round_digits"),
            polynomial=PolynomialConfig.from_dict(data.get("polynomial", {}), f"{path}.polynomial"),
            orbit_trap=OrbitTrapConfig.from_dict(data.get("orbit_trap", {}), f"{path}.orbit_trap"),
            map=MapConfig.from_dict(data.get("map", {}), f"{path}.map"),
        )

    def derive(self, **changes: Any) -> "FractalConfig":
        """Copy with field overrides (used by zoom animations)."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "size": self.size,
            "domain": self.domain.to_dict(),
            "method": self.method,
            "variant": self.variant,
            "iterations": self.iterations,
            "auto_iterations": self.auto_iterations,
            "max_iterations": self.max_iterations,
            "tolerance": self.tolerance,
            "damping": [self.damping.real, self.damping.imag],
            "round_digits": self.round_digits,
            "polynomial": self.polynomial.to_dict(),
            "orbit_trap": self.orbit_trap.to_dict(),
            "map": self.map.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ShadingConfig:
    """Relief shading used on the fractal height field."""

    type: str | None = "matplotlib"
    #: Height field to shade: auto | distance | roots | iteration | iteration-steps
    #: ``auto`` prefers the orbit-trap distance, then a continuous iteration
    #: count (integer counts emboss level boundaries into visible terraces).
    height: str = "auto"
    #: Gaussian blur applied to the height field, in pixels (0 = off).
    smooth: float = 1.0
    azimuth: float = 315.0
    elevation: float = 20.0
    vertical_exaggeration: float = 2.0
    fraction: float = 1.2
    #: How strongly the relief darkens the colour (0 = flat, 1 = full shadow).
    strength: float = 0.8
    # Blinn-Phong weights: ambient, diffuse, specular, shininess.
    ambient: float = 0.0
    diffuse: float = 0.5
    specular: float = 1.2
    shininess: float = 1.0

    TYPES = ("matplotlib", "blinn-phong", "fossil", "none")
    HEIGHTS = ("auto", "distance", "roots", "iteration", "iteration-steps")

    def __post_init__(self) -> None:
        if self.height not in self.HEIGHTS:
            raise ConfigError(f"shading.height: expected one of {list(self.HEIGHTS)}, got {self.height!r}")
        if self.smooth < 0:
            raise ConfigError("shading.smooth: must be >= 0")
        if not 0.0 <= self.strength <= 1.0:
            raise ConfigError("shading.strength: must be within 0..1")

    @classmethod
    def from_dict(cls, value: Any, path: str = "shading") -> "ShadingConfig":
        if isinstance(value, ShadingConfig):
            return value
        if value is None:
            return cls(type=None)
        data = _require_mapping(value, path)
        _check_keys(
            data,
            [
                "type",
                "height",
                "smooth",
                "azimuth",
                "elevation",
                "vertical_exaggeration",
                "fraction",
                "strength",
                "ambient",
                "diffuse",
                "specular",
                "shininess",
                "lights",
            ],
            path,
        )
        default = cls()
        if "lights" in data:
            # Legacy 4-tuple (az, el, vert_exag, fraction) or 7-tuple (blinn-phong).
            lights = data["lights"]
            if not isinstance(lights, Sequence) or len(lights) not in (4, 7):
                raise ConfigError(f"{path}.lights: expected 4 or 7 numbers")
            numbers_ = [_as_float(item, f"{path}.lights[{i}]") for i, item in enumerate(lights)]
            if len(numbers_) == 4:
                return cls(
                    type=_shading_type(data.get("type", default.type), path),
                    height=_as_choice(data.get("height", default.height), cls.HEIGHTS, f"{path}.height"),
                    smooth=_as_float(data.get("smooth", default.smooth), f"{path}.smooth"),
                    azimuth=numbers_[0],
                    elevation=numbers_[1],
                    vertical_exaggeration=numbers_[2],
                    fraction=numbers_[3],
                )
            return cls(
                type=_shading_type(data.get("type", default.type), path),
                height=_as_choice(data.get("height", default.height), cls.HEIGHTS, f"{path}.height"),
                smooth=_as_float(data.get("smooth", default.smooth), f"{path}.smooth"),
                azimuth=numbers_[0],
                elevation=numbers_[1],
                specular=numbers_[2],
                ambient=numbers_[3],
                diffuse=numbers_[4],
                shininess=numbers_[5],
                fraction=numbers_[6],
            )
        height = _as_choice(data.get("height", default.height), cls.HEIGHTS, f"{path}.height")
        return cls(
            type=_shading_type(data.get("type", default.type), path),
            height=height,
            smooth=_as_float(data.get("smooth", default.smooth), f"{path}.smooth"),
            azimuth=_as_float(data.get("azimuth", default.azimuth), f"{path}.azimuth"),
            elevation=_as_float(data.get("elevation", default.elevation), f"{path}.elevation"),
            vertical_exaggeration=_as_float(
                data.get("vertical_exaggeration", default.vertical_exaggeration), f"{path}.vertical_exaggeration"
            ),
            fraction=_as_float(data.get("fraction", default.fraction), f"{path}.fraction"),
            strength=_as_float(data.get("strength", default.strength), f"{path}.strength"),
            ambient=_as_float(data.get("ambient", default.ambient), f"{path}.ambient"),
            diffuse=_as_float(data.get("diffuse", default.diffuse), f"{path}.diffuse"),
            specular=_as_float(data.get("specular", default.specular), f"{path}.specular"),
            shininess=_as_float(data.get("shininess", default.shininess), f"{path}.shininess"),
        )

    @property
    def enabled(self) -> bool:
        return self.type not in (None, "none")

    def to_dict(self) -> dict[str, Any]:
        if not self.enabled:
            return {"type": "none"}
        return {
            "type": self.type,
            "height": self.height,
            "smooth": self.smooth,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "vertical_exaggeration": self.vertical_exaggeration,
            "fraction": self.fraction,
            "strength": self.strength,
            "ambient": self.ambient,
            "diffuse": self.diffuse,
            "specular": self.specular,
            "shininess": self.shininess,
        }


def _shading_type(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _as_choice(value, ShadingConfig.TYPES, f"{path}.type")


@dataclass(frozen=True, slots=True)
class ColourConfig:
    """Where the colormap comes from.

    ``mode``:

    * ``matplotlib`` – built-in matplotlib colormap by ``name``.
    * ``seaborn``    – seaborn palette by ``name`` (needs the ``palette`` extra).
    * ``colors``     – explicit list of hex colours / RGB triples.
    * ``accents``    – extract a palette from ``image`` then derive accents.
    """

    mode: str = "matplotlib"
    name: str = "viridis"
    colors: tuple[str, ...] = ()
    image: Path | None = None
    accent: str = "complementary"
    add_black: bool = False
    n_colors: int = 256
    seed: int | None = None

    MODES = ("matplotlib", "seaborn", "colors", "accents")
    ACCENTS = ("complementary", "triadic", "analogous", "split-complementary", "tetradic", "shades")

    @classmethod
    def from_dict(cls, value: Any, path: str = "colour") -> "ColourConfig":
        if isinstance(value, ColourConfig):
            return value
        data = _require_mapping(value, path)
        _check_keys(data, ["mode", "name", "colors", "image", "accent", "add_black", "n_colors", "seed"], path)
        default = cls()
        mode = _as_choice(data.get("mode", default.mode), cls.MODES, f"{path}.mode")
        colors = data.get("colors", ())
        if mode == "colors" and not colors:
            raise ConfigError(f"{path}.colors: required when mode = 'colors'")
        if colors and not isinstance(colors, Sequence):
            raise ConfigError(f"{path}.colors: expected a list of hex strings or [r, g, b] triples")
        return cls(
            mode=mode,
            name=_as_str(data.get("name", default.name), f"{path}.name"),
            colors=tuple(_colour_entry(item, f"{path}.colors[{i}]") for i, item in enumerate(colors)),
            image=_as_path(data.get("image"), f"{path}.image"),
            accent=_as_choice(data.get("accent", default.accent), cls.ACCENTS, f"{path}.accent"),
            add_black=_as_bool(data.get("add_black", default.add_black), f"{path}.add_black"),
            n_colors=_as_int(data.get("n_colors", default.n_colors), f"{path}.n_colors", minimum=2),
            seed=None if data.get("seed") is None else _as_int(data["seed"], f"{path}.seed", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"mode": self.mode, "add_black": self.add_black, "n_colors": self.n_colors}
        if self.mode in ("matplotlib", "seaborn"):
            data["name"] = self.name
        if self.mode == "colors":
            data["colors"] = list(self.colors)
        if self.mode == "accents":
            data["image"] = None if self.image is None else str(self.image)
            data["accent"] = self.accent
        return data


def _colour_entry(value: Any, path: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and len(value) in (3, 4):
        channels = [_as_float(item, f"{path}[{i}]") for i, item in enumerate(value[:3])]
        if any(not 0 <= channel <= 255 for channel in channels):
            raise ConfigError(f"{path}: RGB channels must be within 0..255")
        return "#%02X%02X%02X" % tuple(round(channel) for channel in channels)
    raise ConfigError(f"{path}: expected '#RRGGBB' or [r, g, b]")


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """One post-render filter (see :mod:`fractalgen.postprocess`).

    Parameters are kept as a plain mapping and validated by the filter itself,
    so a typo is reported with the filter's own list of accepted keys.
    """

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    NAMES = ("deband", "unsharp")

    def __post_init__(self) -> None:
        if self.name not in self.NAMES:
            raise ConfigError(f"postprocess.name: expected one of {list(self.NAMES)}, got {self.name!r}")

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "PostprocessConfig":
        if isinstance(value, PostprocessConfig):
            return value
        if isinstance(value, str):
            return cls(_as_choice(value, cls.NAMES, path))
        data = _require_mapping(value, path)
        if "name" not in data:
            raise ConfigError(f"{path}: missing 'name'")
        name = _as_choice(data["name"], cls.NAMES, f"{path}.name")
        return cls(name=name, params={key: item for key, item in data.items() if key != "name"})

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **dict(self.params)}


@dataclass(frozen=True, slots=True)
class ImageConfig:
    """What a single rendered frame contains."""

    #: ``iteration`` keeps the classic banded look (the rings are removed
    #: afterwards by the ``deband`` post filter); ``smooth`` colours by basin
    #: identity and has no rings by construction.
    return_type: str = "iteration"
    #: Gaussian blur (pixels) applied to the colour field before colouring.
    smooth: float = 0.0
    shading: ShadingConfig = field(default_factory=ShadingConfig)
    #: Post-render filters, applied in order to the finished frame.
    postprocess: tuple[PostprocessConfig, ...] = (
        PostprocessConfig("deband"),
        PostprocessConfig("unsharp"),
    )
    background: Path | None = None
    alpha: int = 255

    RETURN_TYPES = (
        "smooth",
        "iteration",
        "iteration-steps",
        "root",
        "distance",
        "boundary",
        "roots",
        "image",
    )

    def __post_init__(self) -> None:
        if not 0 <= self.alpha <= 255:
            raise ConfigError("image.alpha: must be within 0..255")
        if self.smooth < 0:
            raise ConfigError("image.smooth: must be >= 0")

    @classmethod
    def from_dict(cls, value: Any, path: str = "image") -> "ImageConfig":
        if isinstance(value, ImageConfig):
            return value
        data = _require_mapping(value, path)
        _check_keys(
            data, ["return_type", "return", "smooth", "shading", "postprocess", "background", "alpha"], path
        )
        default = cls()
        return_type = data.get("return_type", data.get("return", default.return_type))
        return cls(
            return_type=_as_choice(return_type, cls.RETURN_TYPES, f"{path}.return_type"),
            smooth=_as_float(data.get("smooth", default.smooth), f"{path}.smooth"),
            postprocess=_parse_postprocess(data.get("postprocess", default.postprocess), f"{path}.postprocess"),
            shading=ShadingConfig.from_dict(data.get("shading", {}), f"{path}.shading"),
            background=_as_path(data.get("background"), f"{path}.background"),
            alpha=_as_int(data.get("alpha", default.alpha), f"{path}.alpha"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "return_type": self.return_type,
            "smooth": self.smooth,
            "shading": self.shading.to_dict(),
            "postprocess": [step.to_dict() for step in self.postprocess],
            "background": None if self.background is None else str(self.background),
            "alpha": self.alpha,
        }


@dataclass(frozen=True, slots=True)
class AnimationConfig:
    """One animation stage and its stage-specific parameters."""

    name: str
    params: Mapping[str, Any] = field(default_factory=dict)

    NAMES = (
        "pulse",
        "flicker",
        "grain",
        "explosion",
        "alpha",
        "zoom",
        "translate",
        "shading",
    )

    def __post_init__(self) -> None:
        if self.name not in self.NAMES:
            raise ConfigError(f"video.animations[].name: expected one of {list(self.NAMES)}, got {self.name!r}")

    @classmethod
    def from_dict(cls, value: Any, path: str) -> "AnimationConfig":
        if isinstance(value, str):
            return cls(_as_choice(value, cls.NAMES, path))
        data = _require_mapping(value, path)
        if "name" not in data:
            raise ConfigError(f"{path}: missing 'name'")
        name = _as_choice(data["name"], cls.NAMES, f"{path}.name")
        params = {key: item for key, item in data.items() if key != "name"}
        return cls(name=name, params=params)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, **dict(self.params)}


def _parse_postprocess(value: Any, path: str) -> tuple[PostprocessConfig, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, Mapping)):
        value = [value]
    if not isinstance(value, Sequence):
        raise ConfigError(f"{path}: expected a list of filters")
    return tuple(PostprocessConfig.from_dict(item, f"{path}[{index}]") for index, item in enumerate(value))


@dataclass(frozen=True, slots=True)
class VideoConfig:
    fps: int = 20
    duration: float = 5.0
    frames: int | None = None
    animations: tuple[AnimationConfig, ...] = (AnimationConfig("alpha"),)
    scale_keyframes: int = 500
    keep_frames: bool = False
    codec: str = "gif"

    def __post_init__(self) -> None:
        if self.fps < 1:
            raise ConfigError("video.fps: must be >= 1")
        if self.frames is None and self.duration <= 0:
            raise ConfigError("video.duration: must be > 0 when video.frames is not set")
        if self.frames is not None and self.frames < 1:
            raise ConfigError("video.frames: must be >= 1")

    @property
    def frame_count(self) -> int:
        if self.frames is not None:
            return self.frames
        return max(1, round(self.fps * self.duration))

    @property
    def animation_names(self) -> tuple[str, ...]:
        return tuple(animation.name for animation in self.animations)

    @classmethod
    def from_dict(cls, value: Any, path: str = "video") -> "VideoConfig":
        if isinstance(value, VideoConfig):
            return value
        if value is None:
            return cls()
        data = _require_mapping(value, path)
        _check_keys(data, ["fps", "duration", "frames", "animations", "scale_keyframes", "keep_frames", "codec"], path)
        default = cls()
        raw_animations = data.get("animations", ["alpha"])
        if isinstance(raw_animations, str):
            raw_animations = [raw_animations]
        if not isinstance(raw_animations, Sequence):
            raise ConfigError(f"{path}.animations: expected a list")
        return cls(
            fps=_as_int(data.get("fps", default.fps), f"{path}.fps", minimum=1),
            duration=_as_float(data.get("duration", default.duration), f"{path}.duration"),
            frames=None if data.get("frames") is None else _as_int(data["frames"], f"{path}.frames", minimum=1),
            animations=tuple(
                AnimationConfig.from_dict(item, f"{path}.animations[{i}]") for i, item in enumerate(raw_animations)
            ),
            scale_keyframes=_as_int(data.get("scale_keyframes", default.scale_keyframes), f"{path}.scale_keyframes", minimum=16),
            keep_frames=_as_bool(data.get("keep_frames", default.keep_frames), f"{path}.keep_frames"),
            codec=_as_choice(data.get("codec", default.codec), ("gif", "mp4"), f"{path}.codec"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "duration": self.duration,
            "frames": self.frames,
            "animations": [animation.to_dict() for animation in self.animations],
            "scale_keyframes": self.scale_keyframes,
            "keep_frames": self.keep_frames,
            "codec": self.codec,
        }


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    workers: int = 0
    verbose: bool = False
    temp_dir: Path | None = None
    seed: int | None = None

    @property
    def worker_count(self) -> int:
        if self.workers > 0:
            return self.workers
        import os

        return max(1, os.cpu_count() or 1)

    @classmethod
    def from_dict(cls, value: Any, path: str = "runtime") -> "RuntimeConfig":
        if isinstance(value, RuntimeConfig):
            return value
        if value is None:
            return cls()
        data = _require_mapping(value, path)
        _check_keys(data, ["workers", "verbose", "temp_dir", "seed"], path)
        default = cls()
        return cls(
            workers=_as_int(data.get("workers", default.workers), f"{path}.workers", minimum=0),
            verbose=_as_bool(data.get("verbose", default.verbose), f"{path}.verbose"),
            temp_dir=_as_path(data.get("temp_dir"), f"{path}.temp_dir"),
            seed=None if data.get("seed") is None else _as_int(data["seed"], f"{path}.seed", minimum=0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "workers": self.workers,
            "verbose": self.verbose,
            "temp_dir": None if self.temp_dir is None else str(self.temp_dir),
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class Config:
    """Full render configuration, one JSON document == one :class:`Config`."""

    fractal: FractalConfig = field(default_factory=FractalConfig)
    colour: ColourConfig = field(default_factory=ColourConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    output: Path | None = None
    form: str = "image"
    base_dir: Path = field(default=Path("."), compare=False)

    FORMS = ("image", "video")

    @classmethod
    def from_dict(cls, value: Any, path: str = "config", base_dir: Path | None = None) -> "Config":
        if isinstance(value, Config):
            return value
        data = _require_mapping(value, path)
        _check_keys(data, ["fractal", "colour", "image", "video", "runtime", "output", "form"], path)
        output = data.get("output")
        if isinstance(output, Mapping):
            form = _as_choice(output.get("form", "image"), cls.FORMS, f"{path}.output.form")
            target = _as_path(output.get("path"), f"{path}.output.path")
        else:
            form = _as_choice(data.get("form", "image"), cls.FORMS, f"{path}.form")
            target = _as_path(output, f"{path}.output")
        return cls(
            fractal=FractalConfig.from_dict(data.get("fractal", {}), f"{path}.fractal"),
            colour=ColourConfig.from_dict(data.get("colour", {}), f"{path}.colour"),
            image=ImageConfig.from_dict(data.get("image", {}), f"{path}.image"),
            video=VideoConfig.from_dict(data.get("video", {}), f"{path}.video"),
            runtime=RuntimeConfig.from_dict(data.get("runtime", {}), f"{path}.runtime"),
            output=target,
            form=form,
            base_dir=Path(base_dir or ".").resolve(),
        )

    def resolve_path(self, path: Path | None) -> Path | None:
        """Resolve an input path against the config file's directory.

        Assets (backgrounds, orbit traps, palette images) live next to the config
        that references them, so they are resolved relative to it.
        """
        if path is None:
            return None
        return path if path.is_absolute() else (self.base_dir / path)

    def resolve_output(self, path: Path | None = None) -> Path | None:
        """Resolve an output path against the current working directory.

        Outputs are what the user asked for on the command line, so a relative
        ``out/frame.png`` lands next to where they ran the command, not next to
        the config file.
        """
        target = self.output if path is None else path
        if target is None:
            return None
        return target if target.is_absolute() else (Path.cwd() / target)

    def to_dict(self) -> dict[str, Any]:
        return {
            "form": self.form,
            "output": None if self.output is None else str(self.output),
            "fractal": self.fractal.to_dict(),
            "colour": self.colour.to_dict(),
            "image": self.image.to_dict(),
            "video": self.video.to_dict(),
            "runtime": self.runtime.to_dict(),
        }


def load_config(path: str | Path) -> Config:
    """Read and validate a JSON configuration file."""
    path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc
    return Config.from_dict(raw, base_dir=path.parent)


def save_config(config: Config, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    return path


def to_jsonable(value: Any) -> Any:
    """Best-effort conversion of dataclass/numpy values to JSON-friendly data."""
    if is_dataclass(value):
        return {item.name: to_jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Mapping):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value
