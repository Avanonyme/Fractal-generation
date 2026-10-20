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

__all__ = ["shade", "normals_from_field", "blinn_phong"]


def normals_from_field(field: np.ndarray) -> np.ndarray:
    """Gradient of ``field`` as a complex array (``dx + 1j * dy``)."""
    gradient_y, gradient_x = np.gradient(np.asarray(field, dtype=np.float64))
    return np.asarray(gradient_x + 1j * gradient_y, dtype=np.complex128)


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


def shade(field: np.ndarray, config: ShadingConfig, cmap, fixed_range: bool = False) -> np.ndarray:
    """Return a shaded image.

    The result is either a 2-D brightness field (``blinn-phong``, needs a
    colormap afterwards) or an ``(h, w, 4)`` RGBA array (LightSource shading).

    ``fixed_range`` means the caller already scaled ``field`` into ``[0, 1]``
    against a global range; the light source then keeps that scale instead of
    re-stretching to each frame's own min/max (animations pass it so brightness
    does not drift).
    """
    if not config.enabled:
        return np.asarray(field, dtype=np.float64)

    if config.type == "blinn-phong":
        return blinn_phong(normals_from_field(field), config)

    from matplotlib.colors import LightSource, Normalize

    blends = {"soft": "soft", "hsv": "hsv", "overlay": "overlay"}
    blend_mode = blends.get(config.blend_mode, "soft")
    light = LightSource(azdeg=config.azimuth, altdeg=config.elevation)
    height = np.asarray(field, dtype=np.float64)
    if config.type == "fossil":
        # "fossil" is the original name for shading only the fractal body.
        height = np.where(height > np.percentile(height, 50), height, height * 0.25)
    fixed = Normalize(vmin=0.0, vmax=1.0, clip=True) if fixed_range else None
    return light.shade(
        height,
        cmap=cmap,
        vert_exag=config.vertical_exaggeration,
        fraction=config.fraction,
        blend_mode=blend_mode,  # type: ignore[arg-type]
        norm=fixed,
    )
