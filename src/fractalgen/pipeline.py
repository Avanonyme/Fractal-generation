"""Image and video pipelines.

Two public entry points, both driven by a :class:`~fractalgen.config.Config`::

    from fractalgen.pipeline import render_image, render_video

    render_image(config)   # -> RenderedImage(rgb array + optional file path)
    render_video(config)   # -> Path to the written animation

Everything below them is reusable: :func:`build_frame` turns a
:class:`~fractalgen.solver.FractalImage` into pixels, and the zoom/shading frame
generators only need a config plus a solved fractal.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import postprocess, render
from ._numeric import as_float, as_int
from .animations import EFFECTS, AnimationContext, resolve_effect_params
from .colour import build_colormap
from .config import AnimationConfig, Config
from .frames import FrameStore
from .shading import relief, smooth_field
from .solver import FractalImage, solve

__all__ = [
    "RenderedImage",
    "build_frame",
    "build_video_frames",
    "render_image",
    "render_video",
    "RE_SOLVE_ANIMATIONS",
]

#: Animations that need the solver again instead of transforming frames.
RE_SOLVE_ANIMATIONS = ("zoom", "translate")


@dataclass
class RenderedImage:
    """A rendered frame plus the fields it came from."""

    rgb: np.ndarray
    fractal: FractalImage
    path: Path | None = None

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no output path given")
        self.path = render.save(self.rgb, target)
        return self.path


# --------------------------------------------------------------------------- #
# single frame
# --------------------------------------------------------------------------- #
def build_frame(
    config: Config,
    fractal: FractalImage,
    cmap,
    *,
    azimuth: float | None = None,
    background: np.ndarray | None = None,
    value_range: tuple[float, float] | None = None,
    height_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Colourise one solved fractal according to ``config.image``.

    ``value_range``/``height_range`` pin the normalisation to a fixed interval,
    which the zoom animation uses so brightness does not drift between frames.
    """
    image_config = config.image
    field = fractal.scalar_field(image_config.return_type)
    if image_config.smooth > 0 and image_config.return_type in ("iteration", "roots"):
        field = smooth_field(field, image_config.smooth)
    rgb = render.colorize(field, cmap, value_range)

    shading_config = image_config.shading
    if shading_config.enabled:
        if azimuth is not None:
            shading_config = replace(shading_config, azimuth=azimuth)
        height = fractal.height_field(image_config.return_type, shading_config.height)
        if height_range is not None:
            height = render.normalise_with(height, height_range)
        rgb = render.modulate(rgb, relief(height, shading_config), shading_config.strength)

    if image_config.postprocess:
        rgb = postprocess.apply_all(image_config.postprocess, rgb)

    rgb = _blend_map_image(config, rgb)

    if background is not None:
        rgb = render.compose_over(rgb, background, image_config.alpha)
    return rgb


def _blend_map_image(config: Config, rgb: np.ndarray) -> np.ndarray:
    """Modulate the frame's brightness by the map image (image-space blend).

    A dynamical map (``fractal.map.mode``) reshapes the basins, but Newton orbits
    only visit a small part of the plane, so the picture stays faint. Multiplying
    the finished frame by the raster keeps every fractal filament while making the
    source image readable - the "picture made of fractal" look.
    """
    map_config = config.fractal.map
    if not map_config.enabled or map_config.blend <= 0 or map_config.image is None:
        return rgb

    from PIL import Image

    from .solver import _map_raster

    raster = _map_raster(str(map_config.image), map_config.size)
    size = np.asarray(rgb).shape[0]
    if raster.shape[0] != size:
        resized = Image.fromarray((np.clip(raster, 0, 1) * 255).astype(np.uint8)).resize(
            (size, size), Image.Resampling.BICUBIC
        )
        raster = np.asarray(resized, dtype=np.float64) / 255.0
    weight = (1.0 - map_config.blend) + map_config.blend * (0.25 + 0.75 * raster)
    return render.to_uint8(np.clip(np.asarray(rgb, dtype=np.float64) * weight[..., None], 0, 255))


def render_image(config: Config, polynomial=None) -> RenderedImage:
    """Solve and render a single image."""
    cmap = build_colormap(config.colour, config.base_dir)
    fractal = solve(config.fractal, polynomial, workers=config.runtime.worker_count)
    rendered = RenderedImage(rgb=build_frame(config, fractal, cmap, background=_background(config)), fractal=fractal)
    target = config.resolve_output()
    if target is not None:
        rendered.save(target)
    return rendered


def _background(config: Config, size: int | None = None) -> np.ndarray | None:
    path = config.resolve_path(config.image.background)
    if path is None:
        return None
    background = render.load(path)
    target = size if size is not None else config.fractal.size
    return background if background.shape[0] == target else render.resize(background, target)


# --------------------------------------------------------------------------- #
# video
# --------------------------------------------------------------------------- #
def build_video_frames(config: Config, cmap=None) -> FrameStore:
    """Produce the animation frames described by ``config.video``."""
    cmap = cmap or build_colormap(config.colour, config.base_dir)
    video = config.video
    works_on_frames = any(name not in RE_SOLVE_ANIMATIONS + ("shading",) for name in video.animation_names)

    result: FractalImage | None = None
    if works_on_frames or not video.animations:
        result = solve(config.fractal, None, workers=config.runtime.worker_count)

    store = FrameStore(_frame_cache_dir(config))
    for animation in video.animations:
        if animation.name in RE_SOLVE_ANIMATIONS:
            store, result = _reframe(config, cmap, animation)
            continue
        if animation.name == "shading":
            if result is None:
                result = solve(config.fractal, None, workers=config.runtime.worker_count)
            store = _shading_frames(config, result, cmap, animation)
            continue
        if result is None:  # pragma: no cover - guarded by works_on_frames
            result = solve(config.fractal, None, workers=config.runtime.worker_count)
        if len(store) == 0:
            store.append(build_frame(config, result, cmap, background=_background(config)))
        # Every frame effect needs the full timeline: a single rendered image is
        # held for `video.frame_count` frames before the chain runs.
        store = _ensure_frame_count(store, video.frame_count)
        context = AnimationContext(
            config=config,
            image=result,
            cmap=cmap,
            frames=len(store),
            fps=video.fps,
        )
        store = EFFECTS[animation.name](store, animation.params, context)

    if result is None:  # no animations configured: a single held frame
        result = solve(config.fractal, None, workers=config.runtime.worker_count)
        store.append(build_frame(config, result, cmap, background=_background(config)))
    return store


def render_video(config: Config, cmap=None) -> Path:
    """Render the configured animation and return the written file."""
    cmap = cmap or build_colormap(config.colour, config.base_dir)
    store = build_video_frames(config, cmap)
    target = config.resolve_output()
    if target is None:
        raise ValueError("video.output: no output path configured")
    codec = "mp4" if target.suffix.lower() in (".mp4", ".webm") else config.video.codec
    written = store.save(target, fps=config.video.fps, codec=codec)
    if not config.video.keep_frames:
        _clear_cache(config)
    return written


def _frame_cache_dir(config: Config) -> Path | None:
    """Disk-backed frames only when a temp dir is configured."""
    base = config.resolve_path(config.runtime.temp_dir)
    return None if base is None else base / "frames"


def _clear_cache(config: Config) -> None:
    base = config.resolve_path(config.runtime.temp_dir)
    if base is None:
        return
    cache = base / "frames"
    if not cache.is_dir():
        return
    for path in cache.glob("*.npy"):
        path.unlink(missing_ok=True)


def _ensure_frame_count(store: FrameStore, count: int) -> FrameStore:
    """Trim or pad (by repeating the last frame) to exactly ``count`` frames."""
    output = FrameStore(getattr(store, "_cache_dir", None))
    for index in range(min(count, len(store))):
        output.append(np.asarray(store[index]))
    if len(output) == 0:
        return output
    last = np.asarray(store[len(store) - 1])
    while len(output) < count:
        output.append(last)
    return output


# --------------------------------------------------------------------------- #
# re-solving animations
# --------------------------------------------------------------------------- #
def _shading_frames(config: Config, result: FractalImage, cmap, animation: AnimationConfig) -> FrameStore:
    """Day/night pass over a static fractal: only the light azimuth changes."""
    count = config.video.frame_count
    options = resolve_effect_params("shading", animation.params)
    rotations = as_float(options["rotations"], "shading.rotations")
    start = config.image.shading.azimuth
    background = _background(config)
    store = FrameStore(_frame_cache_dir(config))
    for azimuth in np.linspace(start, start + 360 * rotations, count):
        store.append(build_frame(config, result, cmap, azimuth=as_float(azimuth, "shading.azimuth"), background=background))
    return store


def _deepest_point(result: FractalImage, fallback: complex) -> complex:
    """Boundary pixel closest to the frame centre (the zoom target)."""
    boundary = result.boundary
    if not boundary.any():
        return fallback
    from scipy import ndimage

    distance = np.asarray(ndimage.distance_transform_edt(np.logical_not(boundary)), dtype=np.float64).ravel()
    flat = np.flatnonzero(boundary.ravel())
    best = flat[np.argmin(distance[flat])]
    row, column = divmod(best, boundary.shape[1])
    domain = result.config.domain
    size = result.config.size
    return complex(
        domain.x_min + column / max(1, size) * domain.width,
        domain.y_min + row / max(1, size) * domain.height,
    )


def _reframe(config: Config, cmap, animation: AnimationConfig) -> tuple[FrameStore, FractalImage]:
    """Zoom / translate animation: re-solve the plane while the window moves."""
    video = config.video
    count = video.frame_count
    params = animation.params
    workers = config.runtime.worker_count
    store = FrameStore(_frame_cache_dir(config))

    options = resolve_effect_params(animation.name, params)

    if animation.name == "translate":
        start = _complex(options["init_damping"])
        end = _complex(options["end_damping"])
        base = solve(config.fractal, None, workers=workers)
        background = _background(config)
        frame_result = base
        for damping in np.linspace(start, end, count):
            frame_result = solve(config.fractal.derive(damping=complex(damping)), base.polynomial, workers=workers)
            store.append(build_frame(config, frame_result, cmap, background=background))
        return store, frame_result

    factor = as_float(options["zoom_factor"], "zoom.zoom_factor")
    oversample = as_int(options["oversample"], "zoom.oversample")
    per_keyframe = max(1, round(video.fps))
    keyframes = max(1, math.ceil(count / per_keyframe))

    domain = config.fractal.domain
    target = domain.center
    polynomial = None
    frame_result: FractalImage | None = None
    _warned = [False]
    height_range: tuple[float, float] | None = None
    value_range: tuple[float, float] | None = None

    previous_ranges: tuple[tuple[float, float], tuple[float, float]] | None = None

    for key in range(keyframes):
        if key > 0 and frame_result is not None:
            target = _deepest_point(frame_result, target)
            candidate = domain.zoomed(factor, target)
            if candidate.can_resolve:
                domain = candidate
            elif not _warned[0]:
                # float64 cannot separate pixels any further: hold the last
                # resolvable window instead of rendering a blocky/blank frame.
                print(
                    f"zoom: stopping at {candidate.max_zoom:,.0f}x, the window is below float64 resolution",
                    file=sys.stderr,
                )
                _warned[0] = True

        render_size = min(4096, config.fractal.size * oversample)
        big = solve(config.fractal.derive(size=render_size, domain=domain), polynomial, workers=workers)
        polynomial = big.polynomial
        frame_result = big

        height = big.height_field(config.image.return_type)
        field = big.scalar_field(config.image.return_type)
        current = (
            (float(np.min(height)), float(np.max(height))),
            (float(np.min(field)), float(np.max(field))),
        )
        previous = previous_ranges or current

        for offset in range(per_keyframe):
            if len(store) >= count:
                break
            fraction = factor ** (-(offset + 1) / per_keyframe)
            weight = (offset + 1) / per_keyframe
            height_range = _lerp_range(previous[0], current[0], weight)
            value_range = _lerp_range(previous[1], current[1], weight)
            wide = build_frame(
                config,
                big,
                cmap,
                background=_background(config, render_size),
                value_range=value_range,
                height_range=height_range,
            )
            store.append(_center_crop_resize(wide, fraction, config.fractal.size))

        previous_ranges = current

    if frame_result is None:  # pragma: no cover - defensive
        frame_result = solve(config.fractal, polynomial, workers=workers)
    return store, frame_result


def _lerp_range(
    start: tuple[float, float], end: tuple[float, float], weight: float
) -> tuple[float, float]:
    """Interpolate a normalisation range so exposure drifts instead of pumping."""
    return (
        start[0] + (end[0] - start[0]) * weight,
        start[1] + (end[1] - start[1]) * weight,
    )


def _complex(value) -> complex:
    if isinstance(value, complex):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return complex(as_float(value[0], "damping[0]"), as_float(value[1], "damping[1]"))
    raise ValueError(f"expected a complex number or [real, imag], got {value!r}")


def _center_crop_resize(image: np.ndarray, fraction: float, size: int) -> np.ndarray:
    """Crop the central ``fraction`` of ``image`` and rescale it to ``size``."""
    array = np.asarray(image)
    height, width = array.shape[:2]
    half_height = max(1, round(height * fraction / 2))
    half_width = max(1, round(width * fraction / 2))
    middle_y, middle_x = height // 2, width // 2
    cropped = array[
        middle_y - half_height : middle_y + half_height,
        middle_x - half_width : middle_x + half_width,
    ]
    return render.resize(cropped, size)
