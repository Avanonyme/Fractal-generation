"""Animation effects.

Every effect takes a :class:`~fractalgen.frames.FrameStore` and returns a new
store, so effects can be chained in any order and each one is independently
testable::

    frames = FrameStore.from_arrays([frame])
    frames = flicker(frames, {"amplitude": 0.5}, context)
    frames = alpha(frames, {}, context)

The old implementation had each effect duplicated for "list of arrays" and
"list of file paths"; the duplicated branches and their divergent behaviour are
gone. Effects that need the fractal boundary or the colormap get them from
:class:`AnimationContext`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import ndimage

from ._numeric import as_float, as_int
from .config import Config, ConfigError, ImageConfig
from .frames import FrameStore
from .render import colorize, compose_over, load, normalise, resize, to_uint8
from .solver import FractalImage

__all__ = ["AnimationContext", "EFFECTS", "flicker", "grain", "pulse", "explosion", "alpha", "resolve_effect_params"]


@dataclass
class AnimationContext:
    """Everything an effect may need beyond its own parameters."""

    config: Config
    image: FractalImage
    cmap: Any
    frames: int
    fps: int
    sprites_dir: Path | None = None

    @property
    def boundary(self) -> np.ndarray:
        return self.image.boundary

    @property
    def background(self) -> np.ndarray | None:
        path = self.config.resolve_path(self.config.image.background)
        return None if path is None else load(path)


def _resolve(params: Mapping[str, Any], defaults: Mapping[str, Any], effect: str) -> dict[str, Any]:
    """Merge user parameters over the effect defaults, rejecting typos."""
    unknown = sorted(set(params) - set(defaults))
    if unknown:
        allowed = ", ".join(sorted(defaults))
        raise ConfigError(f"{effect}: unknown parameter(s) {unknown}; allowed: {allowed}")
    merged = dict(defaults)
    merged.update(params)
    return merged


ALPHA_DEFAULTS: dict[str, Any] = {
    #: Frames used to fade the fractal in (0 = automatic).
    "fade_in": 0,
    #: Frames used to fade it back out (0 = automatic).
    "fade_out": 0,
    #: Alpha clamp inside / outside the fractal body.
    "min_alpha": 60,
    "max_alpha": 230,
    "seed": None,
}


# --------------------------------------------------------------------------- #
# alpha reveal
# --------------------------------------------------------------------------- #
def alpha(frames: FrameStore, params: Mapping[str, Any], context: AnimationContext) -> FrameStore:
    """Fade the fractal in over the background image (``image.background``)."""
    options = _resolve(params, ALPHA_DEFAULTS, "alpha")
    background = context.background
    size = np.asarray(frames[0]).shape[0]
    if background is None:
        background = np.zeros((size, size, 3), dtype=np.uint8)
    elif background.shape[0] != size:
        background = resize(background, size)

    image_config: ImageConfig = context.config.image
    total = len(frames)
    fade_in = as_int(options["fade_in"], "alpha.fade_in") or max(1, total // 12)
    fade_out = as_int(options["fade_out"], "alpha.fade_out") or max(1, total // 24)
    low = as_int(options["min_alpha"], "alpha.min_alpha")
    high = as_int(options["max_alpha"], "alpha.max_alpha")
    boundary = _resize_mask(context.boundary, size)

    output = FrameStore(frames._cache_dir if hasattr(frames, "_cache_dir") else None)
    for index, frame in enumerate(frames):
        if image_config.return_type == "distance":
            level = min(255.0, 255.0 * index / fade_in) if index < fade_in else 255.0
            weight = np.full((size, size), level)
        elif index < fade_in:
            weight = np.where(boundary, 255.0 * index / fade_in, 0.0)
        elif index > total - fade_out:
            weight = np.full((size, size), 255.0 - 255.0 * (index - (total - fade_out)) / fade_out)
        else:
            weight = np.full((size, size), 255.0)

        weight = np.where(boundary, np.clip(weight, low, high), np.clip(weight, low + 40, high)) * (
            image_config.alpha / 255.0
        )
        output.append(compose_over(frame, background, weight))
    return output


def _resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    if mask.shape[0] == size:
        return mask.astype(bool)
    from PIL import Image

    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(resized) > 127


# --------------------------------------------------------------------------- #
# pulsing wave
# --------------------------------------------------------------------------- #
PULSE_DEFAULTS: dict[str, Any] = {
    #: Wave damping; negative values make it decay with distance.
    "beta": None,
    #: Extra pixels of head-room so the wave is not clipped at the border.
    "decal": 0,
    #: Angular frequency of the wave.
    "oscillation_frequency": float(np.pi),
    #: Wave amplitude.
    "oscillation_amplitude": 1.0,
    #: Pixels travelled per frame (None = derived from the frame size).
    "speed": None,
    "seed": None,
}


def pulse(frames: FrameStore, params: Mapping[str, Any], context: AnimationContext) -> FrameStore:
    """Add an expanding radial wave to the fractal body."""
    options = _resolve(params, PULSE_DEFAULTS, "pulse")
    beta = options["beta"]
    decal = as_float(options["decal"], "pulse.decal")
    omega = as_float(options["oscillation_frequency"], "pulse.oscillation_frequency")
    amplitude = as_float(options["oscillation_amplitude"], "pulse.oscillation_amplitude")
    speed = options["speed"]

    size = np.asarray(frames[0]).shape[0]
    boundary = _resize_mask(context.boundary, size).astype(np.float64)
    if beta is None:
        beta = -25 / size
    if speed is None:
        speed = (size + decal) / max(1, len(frames))

    coordinates = np.arange(size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(grid_x - size // 2, grid_y - size // 2)

    output = FrameStore(getattr(frames, "_cache_dir", None))
    for step, frame in enumerate(frames):
        front = speed * step
        wave = amplitude * np.exp(-beta * (radius - front)) * np.sin(omega * (radius - front))
        wave = np.where(radius <= front, wave, 0.0) * boundary * 255.0
        base = normalise(np.asarray(frame).mean(axis=2) if np.asarray(frame).ndim == 3 else np.asarray(frame))
        combined = np.clip(base * 255.0 + wave, 0, 255)
        output.append(colorize(combined, context.cmap))
    return output


# --------------------------------------------------------------------------- #
# flicker: soft twinkling highlights
# --------------------------------------------------------------------------- #
FLICKER_DEFAULTS: dict[str, Any] = {
    #: Peak brightness added to a twinkling pixel, as a fraction of its head-room.
    "amplitude": 0.9,
    #: Share of the frame covered by the sparkle field (0.01 = 1 %).
    "percentage": 0.008,
    #: Number of independent twinkling regions.
    "clusters": 8,
    #: Twinkle cycles over the whole animation (per cluster, scaled randomly).
    "speed": 2.0,
    #: Sparkle blob size relative to the frame (Gaussian sigma).
    "blob_size": 0.005,
    #: Pulse sharpening: 1 = slow swell, 3 = short glints, >4 = strobe.
    "sharpen": 2.5,
    #: Grow the fractal mask by this many pixels so highlights hug the filaments.
    "edge_width": 3,
    #: Restrict twinkling to the fractal body.
    "on_fractal": False,
    #: 0 = uniform twinkling, 1 = mid-tones only (the brightest pixels cannot
    #: brighten any further, and black basins swallow the glow).
    "luminance_weight": 0.8,
    #: "additive" adds light (default); "multiply" keeps the historical look.
    "mode": "additive",
    "seed": None,
}


def _soft_blobs(size: int, coverage: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Sparse, rounded mask covering roughly ``coverage`` of the frame.

    Thresholding *smooth* noise (instead of picking isolated pixels) is what
    turns the effect from TV static into visible glints.
    """
    noise = ndimage.gaussian_filter(rng.standard_normal((size, size)), sigma=max(0.6, sigma), mode="reflect")
    threshold = np.quantile(noise, 1.0 - min(max(coverage, 1e-6), 0.9))
    return noise >= threshold


def _fractal_mask(boundary: np.ndarray, size: int, edge_width: int) -> np.ndarray:
    mask = _resize_mask(boundary, size)
    if edge_width > 0:
        mask = ndimage.binary_dilation(mask, iterations=edge_width)
    return mask


def flicker(frames: FrameStore, params: Mapping[str, Any], context: AnimationContext) -> FrameStore:
    """Twinkling highlights: independent soft clusters pulsing at their own rate.

    Differences from the previous implementation, all visible:

    * the twinkle field is made of **soft blobs** rather than single pixels, so it
      reads as glints instead of strobing static;
    * each cluster twinkles with its own period and phase, so the frame never
      flashes as a whole;
    * brightness is **added** and the result is clipped - the old code
      re-normalised every frame, which shifted the exposure of the whole image
      as soon as one sparkle peaked;
    * highlights are weighted by local luminance, so they land on the bright
      filaments instead of the flat basins;
    * ``edge_width`` dilates the fractal mask, so ``on_fractal`` is no longer a
      one-pixel stencil.
    """
    options = _resolve(params, FLICKER_DEFAULTS, "flicker")
    amplitude = as_float(options["amplitude"], "flicker.amplitude")
    coverage = as_float(options["percentage"], "flicker.percentage")
    clusters = max(1, as_int(options["clusters"], "flicker.clusters"))
    speed = as_float(options["speed"], "flicker.speed")
    blob_size = as_float(options["blob_size"], "flicker.blob_size")
    edge_width = as_int(options["edge_width"], "flicker.edge_width")
    luminance_weight = as_float(options["luminance_weight"], "flicker.luminance_weight")
    sharpen = max(1.0, as_float(options["sharpen"], "flicker.sharpen"))
    mode = str(options["mode"])
    if mode not in ("additive", "multiply"):
        raise ConfigError(f"flicker.mode: expected 'additive' or 'multiply', got {mode!r}")

    size = np.asarray(frames[0]).shape[0]
    rng = np.random.default_rng(options["seed"] if options["seed"] is not None else context.config.runtime.seed)
    sigma = max(0.6, blob_size * size)

    masks = np.stack([_soft_blobs(size, coverage, sigma, rng) for _ in range(clusters)]).astype(np.float64)
    if options["on_fractal"]:
        body = _fractal_mask(context.boundary, size, edge_width).astype(np.float64)
        masks *= body[None, :, :]

    periods = speed * rng.uniform(0.6, 1.6, clusters)
    phases = rng.uniform(0, 2 * np.pi, clusters)
    total_frames = max(1, len(frames))

    output = FrameStore(getattr(frames, "_cache_dir", None))
    for index, frame in enumerate(frames):
        pixels = np.asarray(frame, dtype=np.float64)
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)

        time = index / total_frames
        twinkle = np.zeros((size, size))
        for cluster in range(clusters):
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * periods[cluster] * time + phases[cluster])
            twinkle += masks[cluster] * pulse**sharpen
        twinkle /= clusters

        if luminance_weight > 0:
            # Peaky at mid-grey: that is where added light is actually visible.
            luminance = normalise(pixels.mean(axis=-1))
            focus = 4 * luminance * (1 - luminance)
            twinkle *= (1 - luminance_weight) + luminance_weight * focus

        if mode == "additive":
            # Scale by the remaining head-room so highlights brighten smoothly
            # instead of clipping to white and disappearing.
            head_room = 255.0 - pixels
            frame_out = pixels + amplitude * head_room * twinkle[..., None]
        else:
            frame_out = pixels * (1 + amplitude * twinkle[..., None])
        output.append(to_uint8(np.clip(frame_out, 0, 255)))
    return output


# --------------------------------------------------------------------------- #
# grain: animated film grain
# --------------------------------------------------------------------------- #
GRAIN_DEFAULTS: dict[str, Any] = {
    #: Luminance modulation strength; 0.15 means roughly +/- 15 %.
    "amount": 0.15,
    #: Grain scale relative to the frame (0.003 = 3 px at 1000 px, scale-free).
    "grain_size": 0.003,
    #: Number of noise octaves; more octaves = richer texture.
    "octaves": 2,
    #: How many noise fields the animation boils through.
    "keyframes": 8,
    #: Fraction of a keyframe consumed per second of animation.
    "speed": 0.9,
    #: Corner darkening, 0..1.
    "vignette": 0.0,
    "seed": None,
}


def _noise_field(size: int, sigma: float, octaves: int, rng: np.random.Generator) -> np.ndarray:
    """Multi-octave smooth noise in ``[0, 1]``."""
    total = np.zeros((size, size))
    weight = 1.0
    for octave in range(max(1, octaves)):
        total += weight * ndimage.gaussian_filter(
            rng.standard_normal((size, size)), sigma=max(0.5, sigma * 2**octave), mode="reflect"
        )
        weight *= 0.5
    # Centre on 0.5 with ~0.22 standard deviations: smoothing compresses the
    # range, so a plain min-max normalise leaves the grain almost invisible, and
    # a hard 0..1 stretch (the first attempt) produced harsh black/white clumps
    # instead of film grain.
    deviation = float(total.std()) or 1.0
    field = 0.5 + 0.22 * (total - total.mean()) / deviation
    return np.clip(field, 0.0, 1.0)


def _evolving_fields(
    size: int, sigma: float, octaves: int, count: int, rng: np.random.Generator
) -> np.ndarray:
    """Noise fields that drift smoothly from one to the next.

    Each field is half of the previous one plus half of fresh noise, so linear
    interpolation between neighbours is both seamless and constant-speed (a
    smoothstep between independent fields would pulse at every keyframe).
    """
    fields = np.empty((count, size, size))
    current = _noise_field(size, sigma, octaves, rng)
    fields[0] = current
    for index in range(1, count):
        current = 0.5 * current + 0.5 * _noise_field(size, sigma, octaves, rng)
        fields[index] = current
    return fields


def grain(frames: FrameStore, params: Mapping[str, Any], context: AnimationContext) -> FrameStore:
    """Animated film grain.

    Replaces the old "binary random dots re-rolled every frame" mask, which read
    as television static and pumped the overall brightness because each frame was
    re-normalised. A handful of smooth noise fields are pre-computed and the
    animation interpolates between them with a smoothstep, so the texture
    *boils* continuously; the modulation is symmetric around 1, so the mean
    luminance never drifts.
    """
    options = _resolve(params, GRAIN_DEFAULTS, "grain")
    amount = as_float(options["amount"], "grain.amount")
    grain_size = as_float(options["grain_size"], "grain.grain_size")
    octaves = max(1, as_int(options["octaves"], "grain.octaves"))
    keyframe_count = max(2, as_int(options["keyframes"], "grain.keyframes"))
    speed = as_float(options["speed"], "grain.speed")
    vignette = as_float(options["vignette"], "grain.vignette")

    size = np.asarray(frames[0]).shape[0]
    rng = np.random.default_rng(options["seed"] if options["seed"] is not None else context.config.runtime.seed)
    sigma = max(0.8, grain_size * size)
    fields = _evolving_fields(size, sigma, octaves, keyframe_count, rng)

    shade = np.ones((size, size))
    if vignette > 0:
        coordinates = np.linspace(-1, 1, size)
        grid_x, grid_y = np.meshgrid(coordinates, coordinates)
        radius = np.clip(np.hypot(grid_x, grid_y) / np.sqrt(2), 0, 1)
        shade = 1 - vignette * radius**2

    total_frames = max(1, len(frames))
    output = FrameStore(getattr(frames, "_cache_dir", None))
    for index, frame in enumerate(frames):
        position = (index / total_frames) * speed * keyframe_count
        lower = int(np.floor(position)) % keyframe_count
        upper = (lower + 1) % keyframe_count
        blend = position - np.floor(position)
        noise = fields[lower] * (1 - blend) + fields[upper] * blend

        pixels = np.asarray(frame, dtype=np.float64)
        if pixels.ndim == 2:
            pixels = np.stack([pixels] * 3, axis=-1)
        modulation = (1 + amount * (2 * noise - 1)) * shade
        output.append(to_uint8(np.clip(pixels * modulation[..., None], 0, 255)))
    return output


# --------------------------------------------------------------------------- #
# explosion / burst
# --------------------------------------------------------------------------- #
def _sprite_frames(sprites_dir: Path | None) -> list[np.ndarray] | None:
    if sprites_dir is None or not sprites_dir.is_dir():
        return None
    paths = sorted(path for path in sprites_dir.iterdir() if path.suffix == ".png")
    return [load(path) for path in paths] or None


EXPLOSION_DEFAULTS: dict[str, Any] = {
    #: Logarithmic base controlling how fast the burst grows.
    "speed": 45.0,
    #: Fraction of frames kept at full size between growth and shrink.
    "hold_ratio": 0.2,
    "seed": None,
}


def explosion(frames: FrameStore, params: Mapping[str, Any], context: AnimationContext) -> FrameStore:
    """Grow-then-shrink burst of the fractal over the background.

    The original relied on sprite sheets (``NFT_cache/sprites``) that were never
    committed. When ``sprites_dir`` is absent a procedural radial burst is used,
    so the effect works out of the box.
    """
    options = _resolve(params, EXPLOSION_DEFAULTS, "explosion")
    speed = as_float(options["speed"], "explosion.speed")
    hold_ratio = as_float(options["hold_ratio"], "explosion.hold_ratio")

    size = np.asarray(frames[0]).shape[0]
    steps = np.unique(np.clip(np.logspace(0, np.log(size), num=50, base=speed).astype(int), 1, size))
    growth = list(steps)
    hold = max(0, len(frames) - 2 * len(growth) - 20)
    schedule = growth + [size] * hold + growth[::-1]
    schedule = (schedule + [size] * len(frames))[: len(frames)]

    background = context.background
    if background is None:
        background = np.zeros((size, size, 3), dtype=np.uint8)
    sprites = _sprite_frames(context.sprites_dir)
    rng = np.random.default_rng(options["seed"] if options["seed"] is not None else context.config.runtime.seed)

    output = FrameStore(getattr(frames, "_cache_dir", None))
    coordinates = np.arange(size)
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(grid_x - size / 2, grid_y - size / 2) / (size / 2)

    for index, current in enumerate(schedule):
        current = max(1, as_int(current, "explosion.size"))
        source = np.asarray(frames[min(index, len(frames) - 1)])
        scaled = resize(source, current)
        if sprites is not None:
            sprite = resize(sprites[min(index, len(sprites) - 1)], size)
            canvas = compose_over(scaled, sprite)
        else:
            ring = np.clip(1.0 - np.abs(radius - current / size), 0, 1) ** 2
            canvas = compose_over(scaled, background, np.clip(ring * 255, 0, 255))
        sparkle = rng.random((size, size, 1)) > 0.985
        sparkle_period = max(1, round(1 / max(hold_ratio, 1e-6)))
        if not hold_ratio or index % sparkle_period == 0:
            canvas = np.clip(np.asarray(canvas) + sparkle * 40, 0, 255)
        output.append(canvas.astype(np.uint8))
    return output


#: Parameters of the effects that re-solve the fractal (handled by the pipeline).
RESOLVE_DEFAULTS: dict[str, dict[str, Any]] = {
    "zoom": {"zoom_factor": 1.15, "oversample": 2},
    "translate": {"init_damping": [0.4, -0.5], "end_damping": [1.35, 0.85]},
    "shading": {"rotations": 1.0},
}

EFFECTS = {
    "alpha": alpha,
    "pulse": pulse,
    "flicker": flicker,
    "grain": grain,
    "explosion": explosion,
}


def resolve_effect_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ``params`` against the effect's known keys and fill defaults."""
    if name in EFFECTS:
        defaults = {
            "alpha": ALPHA_DEFAULTS,
            "pulse": PULSE_DEFAULTS,
            "flicker": FLICKER_DEFAULTS,
            "grain": GRAIN_DEFAULTS,
            "explosion": EXPLOSION_DEFAULTS,
        }[name]
    elif name in RESOLVE_DEFAULTS:
        defaults = RESOLVE_DEFAULTS[name]
    else:
        raise ConfigError(f"unknown animation {name!r}")
    return _resolve(params, defaults, name)
