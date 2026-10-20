"""Relief shading of a height field.

Two families are supported:

* ``matplotlib``/``fossil`` - :class:`matplotlib.colors.LightSource` shading,
  which returns a coloured RGBA image directly.
* ``blinn-phong`` - Lambert + Blinn-Phong highlight computed from surface
  normals, returning a scalar brightness field that still needs a colormap.

Normals always come from the gradient of the height field (the orbit-trap
distance field when present, the iteration count otherwise). That keeps
``blinn-phong`` usable with or without an orbit trap - previously it silently
produced a flat image whenever no trap was configured.
"""

from __future__ import annotations

import numpy as np

from .config import ShadingConfig

__all__ = ["relief", "shade", "smooth_field", "normals_from_field", "blinn_phong"]


def smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a field, NaN-safe (non-converged pixels are filled first)."""
    from scipy import ndimage

    array = np.asarray(field, dtype=np.float64)
    if np.isfinite(array).all():
        return ndimage.gaussian_filter(array, sigma, mode="nearest")
    fill = np.nanmax(np.where(np.isfinite(array), array, np.nan))
    return ndimage.gaussian_filter(np.nan_to_num(array, nan=fill), sigma, mode="nearest")


def normals_from_field(field: np.ndarray, scale: float | None = None) -> np.ndarray:
    """Surface normals (``dx + 1j * dy``) of a height field.

    The gradient of a fractal height field has a huge dynamic range: most of the
    frame is flat while the filaments are near-vertical. Scaling by a high
    percentile (instead of using raw slopes) keeps Blinn-Phong shading from
    turning the detail into noise.
    """
    array = np.asarray(field, dtype=np.float64)
    gradient_y, gradient_x = np.gradient(array)
    magnitude = np.abs(gradient_x + 1j * gradient_y)
    if scale is None:
        reference = np.percentile(magnitude[magnitude > 0], 95) if np.any(magnitude > 0) else 1.0
        scale = float(reference) or 1.0
    return np.asarray((gradient_x + 1j * gradient_y) / scale, dtype=np.complex128)


def blinn_phong(normals: np.ndarray, config: ShadingConfig) -> np.ndarray:
    """Blinn-Phong brightness for surface ``normals`` and a directional light."""
    normal = np.asarray(normals, dtype=np.complex128)
    denominator = np.abs(normal)
    with np.errstate(all="ignore"):
        normal = np.where(denominator > 0, normal / denominator, 0.0)

    azimuth = np.deg2rad(config.azimuth)
    elevation = np.deg2rad(config.elevation)
    slope_x, slope_y = np.real(normal), np.imag(normal)

    diffuse = (
        slope_x * np.cos(azimuth) * np.cos(elevation)
        + slope_y * np.sin(azimuth) * np.cos(elevation)
        + np.sin(elevation)
    )
    diffuse = diffuse / (1 + np.sin(elevation))

    half = (np.pi / 2 + elevation) / 2
    specular = (
        slope_x * np.cos(azimuth) * np.sin(half)
        + slope_y * np.sin(azimuth) * np.sin(half)
        + np.cos(half)
    )
    specular = (specular / (1 + np.cos(half))) ** config.shininess

    brightness = config.ambient + config.diffuse * diffuse + config.specular * specular
    return brightness * config.fraction + (1 - config.fraction) / 2


def relief(field: np.ndarray, config: ShadingConfig) -> np.ndarray:
    """Relief intensity in ``[0, 1]`` for a height field.

    This deliberately returns *lighting only*. The old implementation let
    ``LightSource.shade`` colour the image with ``cmap(height)``, which turned
    every iso-line of the height field into a visible contour: iteration-count
    terraces gave concentric rings all over the basins. Colour and lighting are
    now independent - the colour comes from the chosen render field, the relief
    only modulates its brightness.
    """
    if not config.enabled:
        return np.ones(np.shape(field), dtype=np.float64)

    height = np.asarray(field, dtype=np.float64)
    if config.smooth > 0:
        height = smooth_field(height, config.smooth)

    if config.type == "blinn-phong":
        brightness = blinn_phong(normals_from_field(height), config)
        low, high = np.percentile(brightness, [2, 98])
        return np.clip((brightness - low) / max(high - low, 1e-9), 0.0, 1.0)

    from matplotlib.colors import LightSource

    light = LightSource(azdeg=config.azimuth, altdeg=config.elevation)
    exaggeration = config.vertical_exaggeration
    if config.type == "fossil":
        # "fossil" keeps the plateau look: flatten the low half of the field so
        # only the fractal body catches the light.
        threshold = np.percentile(height, 50)
        height = np.where(height > threshold, height, threshold + (height - threshold) * 0.15)
        exaggeration *= 1.6
    return light.hillshade(height, vert_exag=exaggeration, fraction=config.fraction)


def shade(field: np.ndarray, config: ShadingConfig, cmap=None, fixed_range: bool = False) -> np.ndarray:
    """Backwards-compatible alias returning the relief intensity."""
    del cmap, fixed_range
    return relief(field, config)
