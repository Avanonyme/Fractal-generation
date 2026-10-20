"""Local web UI for the fractal generator.

    fractalgen ui            # serves http://127.0.0.1:8123
    python -m fractalgen.ui  # same thing

The page is a single HTML document (inline CSS/JS, no CDN, works offline) that
renders a form from ``GET /api/schema`` and posts it to ``GET /api/render``. The
schema is generated from the config dataclasses by walking ``Config().to_dict()``,
so a new config field shows up in the UI automatically; only presentation
(choices, ranges, help text) comes from the tables below.

The server is standard library only (``http.server``) and binds to the loopback
interface by default. Renders are serialised behind one lock - two clicks cannot
start two solvers - and happen in a worker thread with a timeout so a slow
render cannot wedge the request loop.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import tempfile
import threading
import time
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping

from .animations import (
    ALPHA_DEFAULTS,
    EXPLOSION_DEFAULTS,
    FLICKER_DEFAULTS,
    GRAIN_DEFAULTS,
    PULSE_DEFAULTS,
    RESOLVE_DEFAULTS,
)
from .config import (
    OrbitTrapConfig,
    AnimationConfig,
    ColourConfig,
    Config,
    ConfigError,
    FractalConfig,
    ImageConfig,
    MapConfig,
    PolynomialConfig,
    PostprocessConfig,
    ShadingConfig,
)
from . import optional
from .image_import import analyse_image, write_assets
from .pipeline import render_image, render_video
from .postprocess import PARAMS as POSTPROCESS_PARAMS

__all__ = ["serve", "create_server", "build_schema", "DEFAULT_PORT"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8123
MAX_BODY_BYTES = 8 * 1024 * 1024
RENDER_TIMEOUT_SECONDS = 180.0
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
#: Characters that are safe in a file name once spaces and punctuation are folded.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


class RequestTooLarge(ValueError):
    """Raised when a request body exceeds :data:`MAX_BODY_BYTES`."""

#: Where renders and uploaded images are written. One directory per process.
_OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="fractalgen-ui-"))
_RENDER_LOCK = threading.Lock()
_RENDER_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fractalgen-render")


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Field:
    """One editable leaf of the configuration."""

    path: str
    label: str
    kind: str
    default: Any = None
    choices: tuple[Any, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    help: str = ""
    optional: bool = False
    managed: bool = False
    item: Mapping[str, Any] | None = None
    group: str = "general"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path": self.path,
            "label": self.label,
            "kind": self.kind,
            "section": self.group,
        }
        if self.default is not None or self.kind not in ("list", "path"):
            payload["default"] = self.default
        if self.choices:
            labels = _CHOICE_LABELS.get(self.path, {})
            payload["choices"] = [
                [choice[0], labels.get(str(choice[0]), choice[1])]
                if isinstance(choice, tuple) and len(choice) == 2
                else [choice, labels.get(str(choice), str(choice).replace("_", " "))]
                for choice in self.choices
            ]
        for name in ("minimum", "maximum", "step"):
            value = getattr(self, name)
            if value is not None:
                payload[name] = value
        if self.help:
            payload["help"] = self.help
        if self.optional:
            payload["optional"] = True
        if self.managed:
            payload["managed"] = True
        if self.item is not None:
            payload["item"] = self.item
        return payload


def _catalogue() -> tuple[tuple[str, str], ...]:
    """Every colormap name the renderer can actually resolve."""
    from matplotlib import colormaps

    names = set(colormaps)
    try:  # seaborn palettes are only valid for colour.mode='seaborn'
        import seaborn as sns

        names.update(sns.color_palette(None).__class__.__module__ and [])
        names.update(
            {
                "rocket",
                "mako",
                "flare",
                "crest",
                "icefire",
                "vlag",
                "Spectral",
                "RdYlGn",
                "deep",
                "muted",
                "pastel",
                "bright",
                "dark",
                "colorblind",
            }
        )
    except ImportError:
        pass
    return tuple((name, name) for name in sorted(names))


#: enum value -> display label, so the UI is readable ("extract from image").
_CHOICE_LABELS: dict[str, dict[str, str]] = {
    "colour.mode": {
        "matplotlib": "Built-in colormap",
        "seaborn": "Seaborn palette",
        "colors": "Custom colours",
        "accents": "Extract from an image + accents",
    },
    "image.return_type": {
        "smooth": "Smooth (basin hue, no rings)",
        "iteration": "Iteration bands (rings, debanded)",
        "iteration-steps": "Iteration steps (raw bands)",
        "root": "Root identity (flat basins)",
        "roots": "Distance to root (smooth gradient)",
        "distance": "Orbit-trap distance",
        "boundary": "Fractal boundary mask",
        "image": "Source image colours + fractal structure",
        "trap": "Orbit-trap colouring (image painted along the orbits)",
    },
    "image.shading.type": {
        "matplotlib": "Hillshade (LightSource)",
        "fossil": "Fossil (plateau light)",
        "blinn-phong": "Blinn-Phong (specular)",
        "none": "No shading",
    },
    "image.shading.height": {
        "auto": "Auto (trap, else distance to root)",
        "roots": "Distance to root",
        "iteration": "log(1 + iteration count)",
        "iteration-steps": "Raw iteration count",
        "distance": "Orbit-trap distance",
    },
    "fractal.method": {"newton": "Newton (quadratic)", "halley": "Halley (cubic)"},
    "fractal.variant": {"standard": "Standard (start at pixel)", "nova": "Nova (per-pixel polynomial)"},
    "fractal.polynomial.mode": {
        "random": "Random polynomial",
        "roots": "Explicit roots",
        "coefs": "Explicit coefficients",
        "taylor": "Taylor series of a function",
    },
    "fractal.map.mode": {"damping": "Damping (strong)", "warp": "Warp (subtle)"},
    "fractal.orbit_trap.colour_mode": {
        "closest": "Closest approach (image clearly visible)",
        "average": "Average along the orbit (washed out)",
        "last": "Last orbit point",
    },
    "video.codec": {"gif": "GIF", "mp4": "MP4 (needs imageio-ffmpeg)"},
    "colour.accent": {
        "complementary": "Complementary",
        "triadic": "Triadic",
        "analogous": "Analogous",
        "split-complementary": "Split complementary",
        "tetradic": "Tetradic",
        "shades": "Shades",
    },
}

#: path -> enum choices, taken from the config classes so they cannot drift.
_CHOICES: dict[str, tuple[Any, ...]] = {
    "form": ("image", "video"),
    "fractal.method": FractalConfig.METHODS,
    "fractal.variant": FractalConfig.VARIANTS,
    "fractal.polynomial.mode": PolynomialConfig.MODES,
    "fractal.polynomial.function": PolynomialConfig.FUNCTIONS,
    "fractal.orbit_trap.distance_mode": tuple(range(8)),
    "fractal.orbit_trap.colour_mode": OrbitTrapConfig.COLOUR_MODES,
    "fractal.map.mode": MapConfig.MODES,
    "colour.mode": ColourConfig.MODES,
    "colour.accent": ColourConfig.ACCENTS,
    "colour.name": _catalogue(),
    "image.return_type": ImageConfig.RETURN_TYPES,
    "image.shading.type": ShadingConfig.TYPES,
    "image.shading.height": ShadingConfig.HEIGHTS,
    "video.codec": ("gif", "mp4"),
    "video.animations[].name": AnimationConfig.NAMES,
    "image.postprocess[].name": PostprocessConfig.NAMES,
}

#: path -> (min, max, step)
_RANGES: dict[str, tuple[float, float, float]] = {
    "fractal.size": (16, 2400, 2),
    "fractal.iterations": (1, 600, 1),
    "fractal.max_iterations": (1, 8192, 1),
    "fractal.tolerance": (1e-12, 1e-2, 1e-7),
    "fractal.round_digits": (0, 12, 1),
    "fractal.orbit_trap.size": (16, 1024, 8),
    "fractal.orbit_trap.distance_mode": (0, 7, 1),
    "fractal.map.strength": (0.0, 5.0, 0.05),
    "fractal.orbit_trap.scale": (0.25, 16.0, 0.25),
    "fractal.map.size": (8, 512, 8),
    "fractal.polynomial.max_degree": (2, 30, 1),
    "fractal.polynomial.degree": (1, 30, 1),
    "fractal.polynomial.min_area": (0.0, 1.0, 0.01),
    "fractal.polynomial.max_area": (0.0, 1.0, 0.01),
    "fractal.polynomial.attempts": (1, 200, 1),
    "fractal.polynomial.interval": (0.01, 10.0, 0.01),
    "colour.n_colors": (2, 1024, 1),
    "image.smooth": (0.0, 12.0, 0.1),
    "image.trap_mix": (0.0, 1.0, 0.01),
    "image.alpha": (0, 255, 1),
    "image.shading.smooth": (0.0, 12.0, 0.1),
    "image.shading.azimuth": (0, 360, 1),
    "image.shading.elevation": (0, 90, 1),
    "image.shading.vertical_exaggeration": (0.0, 12.0, 0.1),
    "image.shading.fraction": (0.0, 4.0, 0.05),
    "image.shading.strength": (0.0, 1.0, 0.01),
    "image.shading.ambient": (0.0, 2.0, 0.01),
    "image.shading.diffuse": (0.0, 4.0, 0.01),
    "image.shading.specular": (0.0, 8.0, 0.01),
    "image.shading.shininess": (0.1, 32.0, 0.1),
    "video.fps": (1, 60, 1),
    "video.duration": (0.1, 120.0, 0.1),
    "video.frames": (1, 2000, 1),
    "video.scale_keyframes": (16, 4096, 8),
    "runtime.workers": (1, 32, 1),
}

#: path -> short help text shown next to the input.
_HELP: dict[str, str] = {
    "form": "Render a single image or an animation.",
    "fractal.size": "Square resolution in pixels.",
    "fractal.domain.x": "Visible real-axis interval [min, max].",
    "fractal.domain.y": "Visible imaginary-axis interval [min, max].",
    "fractal.method": "Root finder: Newton (quadratic) or Halley (cubic).",
    "fractal.variant": "standard starts at the pixel; nova perturbs the polynomial per pixel.",
    "fractal.iterations": "Base iteration budget before the automatic growth kicks in.",
    "fractal.auto_iterations": "Grow the budget when too many pixels saturate (needed for deep zooms).",
    "fractal.max_iterations": "Ceiling for the automatic budget; empty = 8x iterations.",
    "fractal.tolerance": "Convergence threshold on the Newton step.",
    "fractal.damping": "Complex damping [real, imag] applied to every step.",
    "fractal.round_digits": "Root quantisation used for root-identity colouring.",
    "fractal.polynomial.mode": "random, explicit roots, coefficients, or a Taylor series.",
    "fractal.orbit_trap.scale": "How many times the trap picture spans the frame: 1 = one huge copy, 2-3 = small tiles.",
    "fractal.orbit_trap.offset": "Shift of the trap picture, in plane units.",
    "fractal.orbit_trap.wrap": "Tile the trap picture instead of smearing at its edges.",
    "fractal.orbit_trap.colour_mode": "Where the trap colour is sampled: closest approach, average along the orbit, or the last orbit point.",
    "fractal.orbit_trap.size": "Trap bitmap resolution; higher keeps small details of the picture.",
    "image.trap_mix": "How much the trap picture replaces the fractal colours (lower fades the orbit streaks).",
    "image.return_type": "Which field is coloured: iteration bands, smooth basins, root identity, source image, or orbit trap.",
    "fractal.polynomial.seed": "Seed for the random polynomial (empty = different every run).",
    "fractal.polynomial.max_degree": "Random polynomials use 4..this many coefficients.",
    "fractal.polynomial.min_area": "Reject random polynomials that fill less of the frame.",
    "fractal.polynomial.max_area": "Reject random polynomials that fill more of the frame.",
    "fractal.polynomial.roots": "Explicit roots, e.g. ['1+0j', '0+1j'] - used when mode = roots.",
    "fractal.polynomial.coefs": "Ascending coefficients - used when mode = coefs.",
    "fractal.polynomial.function": "Function approximated when mode = taylor.",
    "fractal.orbit_trap.image": "Bitmap whose distance field is injected into the iteration.",
    "fractal.orbit_trap.distance_mode": "0-7: shape warps applied to the trap distance (1 = plain distance).",
    "fractal.map.image": "Raster sampled per pixel as the map modifier.",
    "fractal.map.strength": "How strongly the map bends the basins (0 = off, 1.4 = readable image).",
    "fractal.map.mode": "damping follows the picture; warp only nudges the pattern.",
    "colour.mode": "matplotlib name, seaborn palette, explicit colours, or a palette from an image.",
    "colour.name": "Colormap or palette name (depends on the mode).",
    "colour.colors": "Explicit hex colours; interpolated into a colormap.",
    "colour.image": "Image to extract the palette from (mode = accents).",
    "colour.accent": "Accent scheme added to the extracted palette.",
    "colour.add_black": "Prepend black to the colormap.",
    "colour.n_colors": "Number of samples in the colormap.",
    "image.return_type": "Which field is colourised; smooth avoids contour rings.",
    "image.smooth": "Blur applied to the colour field before colourizing (px).",
    "image.shading.type": "Relief shading: matplotlib hillshade, blinn-phong, fossil, or none.",
    "image.shading.height": "Height field used for the relief.",
    "image.shading.strength": "How much the relief darkens the colour (0 = flat).",
    "image.shading.azimuth": "Light azimuth in degrees.",
    "image.shading.elevation": "Light elevation in degrees.",
    "image.shading.vertical_exaggeration": "Relief exaggeration.",
    "image.postprocess": "Post-render filters, applied in order.",
    "image.background": "Composited behind the frame.",
    "image.alpha": "Foreground opacity when compositing over a background.",
    "video.fps": "Frames per second.",
    "video.duration": "Seconds; ignored when frames is set.",
    "video.frames": "Exact frame count (empty = fps * duration).",
    "video.animations": "Effect chain, run in order.",
    "video.codec": "gif (Pillow) or mp4 (needs the video extra).",
    "runtime.workers": "Solver processes inside the render thread; 1 is safest.",
    "runtime.seed": "Global seed for effects that randomise (empty = random).",
    "runtime.verbose": "Print solver progress to the server console.",
}


_SECTIONS: tuple[tuple[str, str], ...] = (
    ("general", "Output"),
    ("fractal", "Fractal"),
    ("colour", "Colour"),
    ("image", "Image"),
    ("postprocess", "Post-processing"),
    ("video", "Animation"),
    ("runtime", "Runtime"),
)

#: Fields that are not present in ``Config().to_dict()`` but are still editable.
_EXTRA_FIELDS: tuple[Field, ...] = (
    Field(
        path="colour.colors",
        label="Colours",
        kind="colors",
        default=["#000000", "#ffffff"],
        optional=True,
        help=_HELP["colour.colors"],
        group="colour",
    ),
    Field(path="colour.image", label="Palette image", kind="path", optional=True, help=_HELP["colour.image"], group="colour"),
    Field(
        path="fractal.polynomial.roots",
        label="Roots",
        kind="complex-list",
        default=["1+0j", "-1+0j", "0+1j", "0-1j"],
        optional=True,
        help=_HELP["fractal.polynomial.roots"],
        group="fractal",
    ),
    Field(
        path="fractal.polynomial.coefs",
        label="Coefficients",
        kind="complex-list",
        default=["0+0j", "1+0j"],
        optional=True,
        help=_HELP["fractal.polynomial.coefs"],
        group="fractal",
    ),
    Field(
        path="fractal.orbit_trap.image",
        label="Trap bitmap",
        kind="path",
        optional=True,
        help=_HELP["fractal.orbit_trap.image"],
        group="fractal",
    ),
    Field(
        path="fractal.map.image",
        label="Map raster",
        kind="path",
        optional=True,
        help=_HELP["fractal.map.image"],
        group="fractal",
    ),
    Field(path="image.background", label="Background", kind="path", optional=True, help=_HELP["image.background"], group="image"),
)


def _iter_leaves(value: Any, prefix: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            yield from _iter_leaves(item, child)
        return
    if isinstance(value, list) and value and isinstance(value[0], Mapping):
        yield prefix, value
        return
    yield prefix, value


def _kind_for(path: str, value: Any) -> str:
    if path in _CHOICES:
        return "enum" if path != "fractal.orbit_trap.distance_mode" else "int"
    if path in ("fractal.damping", "fractal.orbit_trap.offset"):
        return "complex"
    if path.endswith("domain.x") or path.endswith("domain.y"):
        return "span"
    if path in ("image.postprocess", "video.animations"):
        return "list"
    if path.startswith("image.shading") and path.endswith(".type"):
        return "enum"
    if value is None:
        return "path"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return "string"


_SECTION_TITLE_BY_ID = {
    "general": "General",
    "fractal": "Fractal",
    "colour": "Colour",
    "image": "Image",
    "postprocess": "Post-process",
    "video": "Video",
    "runtime": "Runtime",
}

_FIELD_TITLES = {
    "fractal.polynomial": "Polynomial",
    "fractal.orbit_trap": "Orbit trap",
    "fractal.map": "Map",
    "image.shading": "Shading",
    "fractal.domain": "Domain",
}


def _label_for(path: str) -> str:
    """Human label that is unique: ``Section · parent · leaf``.

    Two fields used to render as "smooth" (image and shading); prefixing the
    parent path keeps every label distinct.
    """
    parts = path.replace("[]", "").split(".")
    group = "postprocess" if path.startswith("image.postprocess") else parts[0]
    prefix = _SECTION_TITLE_BY_ID.get(group, group.title())
    parents = []
    for index in range(1, len(parts) - 1):
        segment = parts[index]
        parents.append(_FIELD_TITLES.get(".".join(parts[: index + 1]), segment.replace("_", " ").title()))
    leaf = parts[-1].replace("_", " ")
    parts_text = " · ".join([prefix, *parents, leaf])
    return parts_text.replace("[].", " ").strip()


def _item_schema(path: str) -> Mapping[str, Any] | None:
    if path == "video.animations":
        parameters = {
            "alpha": ALPHA_DEFAULTS,
            "pulse": PULSE_DEFAULTS,
            "flicker": FLICKER_DEFAULTS,
            "grain": GRAIN_DEFAULTS,
            "explosion": EXPLOSION_DEFAULTS,
            **RESOLVE_DEFAULTS,
        }
        return {
            "nameChoices": list(AnimationConfig.NAMES),
            "params": {
                name: [
                    {
                        "name": key,
                        "kind": "bool" if isinstance(default, bool) else ("int" if isinstance(default, int) else ("float" if isinstance(default, float) else "string")),
                        "default": default,
                    }
                    for key, default in defaults.items()
                    if key != "seed"
                ]
                for name, defaults in parameters.items()
            },
        }
    if path == "image.postprocess":
        return {
            "nameChoices": list(PostprocessConfig.NAMES),
            "params": {
                name: [
                    {
                        "name": key,
                        "kind": "bool" if isinstance(default, bool) else ("int" if isinstance(default, int) else "float"),
                        "default": default,
                    }
                    for key, default in defaults.items()
                ]
                for name, defaults in POSTPROCESS_PARAMS.items()
            },
        }
    return None


def _default_config() -> dict[str, Any]:
    """Defaults tuned for the UI: one solver process, no output path."""
    data = Config().to_dict()
    data.pop("output", None)
    data["form"] = "image"
    data["runtime"]["workers"] = 1
    return data


def build_schema() -> dict[str, Any]:
    """Describe every editable field, derived from the config dataclasses."""
    defaults = _default_config()
    fields: list[Field] = []
    seen: set[str] = set()

    for path, value in _iter_leaves(defaults):
        if path == "output":
            continue
        group = path.split(".")[0]
        if path.startswith("image.postprocess"):
            group = "postprocess"
        entry = Field(
            path=path,
            label=_label_for(path),
            kind=_kind_for(path, value),
            default=value,
            choices=_CHOICES.get(path, ()),
            help=_HELP.get(path, ""),
            item=_item_schema(path),
            group="general" if path in ("form",) else group,
        )
        if path in _RANGES:
            minimum, maximum, step = _RANGES[path]
            entry = Field(
                **{
                    **entry.__dict__,
                    "minimum": minimum,
                    "maximum": maximum,
                    "step": step,
                }
            )
        if path in ("image.shading.type",):
            entry = Field(**{**entry.__dict__, "optional": True})
        fields.append(entry)
        seen.add(path)

    for extra in _EXTRA_FIELDS:
        if extra.path not in seen:
            fields.append(extra)
            seen.add(extra.path)

    return {
        "sections": [
            {"id": identifier, "title": title, "forms": _SECTION_FORMS.get(identifier, ["image", "video"])}
            for identifier, title in _SECTIONS
        ],
        "fields": [entry.to_dict() for entry in fields],
        "defaults": defaults,
        "animations": list(AnimationConfig.NAMES),
    }


def _section_title(identifier: str) -> str:
    for key, title in _SECTIONS:
        if key == identifier:
            return title
    return identifier.replace("_", " ").title()


_SECTION_TITLES = dict(_SECTIONS)

#: which render form a section applies to (the UI hides the rest).
_SECTION_FORMS: dict[str, list[str]] = {"video": ["video"], "postprocess": ["image", "video"]}


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
def _slug_name(name: str, fallback: str = "upload") -> str:
    """Turn an arbitrary uploaded file name into a safe one instead of rejecting it.

    ``_safe_name`` is the strict guard used when *reading* an output path; an
    upload only needs a predictable name, so "pink house.jpg" becomes
    "pink_house.jpg" rather than an error.
    """
    raw = str(name or "").replace("\\", "/")
    base = Path(raw).name  # drop any directory part
    if ".." in base:
        base = base.replace("..", "")
    stem, dot, suffix = base.rpartition(".")
    if not dot:
        stem, suffix = base, ""
    cleaned_stem = _UNSAFE_CHARS.sub("_", stem).strip("._-") or fallback
    cleaned_suffix = _UNSAFE_CHARS.sub("", suffix).lower()[:8]
    candidate = f"{cleaned_stem[:96]}.{cleaned_suffix}" if cleaned_suffix else cleaned_stem[:96]
    return candidate


def _safe_name(name: str) -> str:
    """Reject anything that is not a plain file name inside the output directory."""
    raw = str(name)
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(f"invalid file name: {name!r}")
    candidate = Path(raw).name
    if not _SAFE_NAME.match(candidate):
        raise ValueError(f"invalid file name: {name!r}")
    root = _OUTPUT_DIR.resolve()
    if root not in (root / candidate).resolve().parents:
        raise ValueError(f"invalid file name: {name!r}")
    return candidate


def _resolve_output(name: str) -> Path:
    """Map a requested file name to a path inside the output directory.

    Two independent checks: the name must be a whitelisted bare file name, and the
    resolved path must still be a direct child of the output root (which defeats
    symlinks and any separator trick that slipped through).
    """
    safe = _safe_name(name)
    root = _OUTPUT_DIR.resolve()
    path = (root / safe).resolve()
    if path.parent != root:
        raise ValueError(f"invalid file name: {name!r}")
    return path


def _localise_urls(value: Any) -> Any:
    """Map ``http://127.0.0.1:PORT/outputs/<name>`` strings back to local paths.

    The page shows assets through URLs, so a config built in the browser can end
    up with a URL where the renderer expects a path. Only URLs that point at this
    server's output directory are rewritten.
    """
    if isinstance(value, Mapping):
        return {key: _localise_urls(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_localise_urls(item) for item in value]
    if isinstance(value, str) and "://" in value and "/outputs/" in value:
        name = value.split("/outputs/", 1)[1].split("?", 1)[0].split("/")[0]
        candidate = _OUTPUT_DIR / name
        if name and candidate.is_file():
            return str(candidate)
    return value


def _render(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate, render and return the served URL."""
    form = str(payload.get("form", "image"))
    if form not in ("image", "video"):
        raise ConfigError(f"form: expected 'image' or 'video', got {form!r}")

    raw = payload.get("config")
    if not isinstance(raw, Mapping):
        raise ConfigError("config: expected a configuration object")

    data = _localise_urls(json.loads(json.dumps(raw)))  # deep copy of plain JSON
    data["form"] = form

    codec = "gif"
    if form == "video":
        requested = str(data.get("video", {}).get("codec", "gif"))
        # imageio needs imageio-ffmpeg for MP4; fall back instead of failing the
        # request with imageio's cryptic "TiffWriter.write() got fps".
        codec = requested if requested == "gif" or optional.available("mp4") else "gif"
        data.setdefault("video", {})["codec"] = codec
    suffix = ".gif" if form == "video" and codec == "gif" else (".mp4" if form == "video" else ".png")
    target = _OUTPUT_DIR / f"fractal-{int(time.time() * 1000)}{suffix}"
    data["output"] = str(target)

    config = Config.from_dict(data, base_dir=_OUTPUT_DIR)

    started = time.monotonic()
    if form == "video":
        render_video(config)
    else:
        render_image(config)
    elapsed = time.monotonic() - started
    if not target.exists():
        raise ConfigError("render produced no file")

    return {
        "ok": True,
        "url": f"/outputs/{target.name}",
        "elapsed": round(elapsed, 3),
        "form": form,
        "codec": codec,
        "filename": target.name,
    }


def render_request(payload: Mapping[str, Any], timeout: float = RENDER_TIMEOUT_SECONDS) -> dict[str, Any]:
    """Run one render under the global lock, in a worker thread, with a timeout."""
    if not _RENDER_LOCK.acquire(timeout=timeout):
        raise TimeoutError("another render is still running")
    try:
        future = _RENDER_POOL.submit(_render, payload)
        try:
            return future.result(timeout=timeout)
        except TimeoutError:  # concurrent.futures.TimeoutError is builtins.TimeoutError
            raise
    finally:
        _RENDER_LOCK.release()


# --------------------------------------------------------------------------- #
# image import
# --------------------------------------------------------------------------- #
def import_image(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Store an uploaded image, analyse it, and return palette + suggested config."""
    encoded = payload.get("data")
    if not isinstance(encoded, str) or not encoded:
        raise ConfigError("data: expected a base64-encoded image")
    name = _slug_name(str(payload.get("filename", "upload.png")))

    try:
        blob = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ConfigError(f"data: not valid base64 ({error})") from error
    if len(blob) > MAX_BODY_BYTES:
        raise ConfigError("data: image is larger than 8 MB")

    uploads = _OUTPUT_DIR / "uploads"
    uploads.mkdir(parents=True, exist_ok=True)
    source = uploads / name
    source.write_bytes(blob)

    try:
        analysis = analyse_image(
            source,
            colours=int(payload.get("colours", 6)),
            raster_size=int(payload.get("raster_size", 96)),
        )
    except ConfigError as error:
        if "cannot identify image file" in str(error):
            raise ConfigError(
                f"{name}: unsupported image format - convert it to JPEG or PNG "
                "(HEIC/HEIF needs 'pip install pillow-heif'); details: {error}"
            ) from error
        raise
    stem = f"{Path(name).stem}-assets"
    raster_path, palette_path = write_assets(analysis, _OUTPUT_DIR, stem=stem)

    return {
        "ok": True,
        "colours": list(analysis.colours),
        "source_path": str(source),
        "source_url": f"/outputs/uploads/{_safe_name(source.name)}",
        "raster_path": str(raster_path),
        "palette_path": str(palette_path),
        "raster_url": f"/outputs/{_safe_name(raster_path.name)}",
        "palette_url": f"/outputs/{_safe_name(palette_path.name)}",
        "size": list(analysis.source_size),
        "suggested": {
            "colour": {"mode": "colors", "colors": list(analysis.colours)},
            "fractal": {
                "map": {"image": str(raster_path), "strength": 1.4, "size": 96, "mode": "damping"},
                "orbit_trap": {"image": str(raster_path), "size": 96, "distance_mode": 1},
            },
        },
    }


def list_outputs(limit: int = 40) -> list[dict[str, Any]]:
    """Recent renders in the output directory, newest first.

    The UI shows them in an "outputs" picker so a render is not lost when the
    preview is replaced (or when the page is reloaded).
    """
    if not _OUTPUT_DIR.is_dir():
        return []
    entries: list[tuple[float, Path]] = []
    for path in _OUTPUT_DIR.iterdir():
        if not path.is_file() or not _SAFE_NAME.match(path.name):
            continue
        if path.suffix.lower() not in (".png", ".gif", ".mp4", ".webm", ".json"):
            continue
        try:
            entries.append((path.stat().st_mtime, path))
        except OSError:  # pragma: no cover - file vanished mid-scan
            continue
    entries.sort(reverse=True)
    listed = []
    for modified, path in entries[:limit]:
        listed.append(
            {
                "name": path.name,
                "url": f"/outputs/{path.name}",
                "bytes": path.stat().st_size,
                "modified": int(modified),
                "kind": "video" if path.suffix.lower() in (".gif", ".mp4", ".webm") else "image",
            }
        )
    return listed


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class _Handler(BaseHTTPRequestHandler):
    server_version = "fractalgen-ui"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 - stdlib signature
        if self.server.verbose:  # type: ignore[attr-defined]
            print(f"[ui] {format % args}")

    # -- helpers ---------------------------------------------------------- #
    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: Mapping[str, Any]) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"ok": False, "error": message})

    def _body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            raise ConfigError("empty request body")
        if length > MAX_BODY_BYTES:
            raise RequestTooLarge("request body is larger than 8 MB")
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ConfigError(f"invalid JSON: {error.msg}") from error
        if not isinstance(payload, Mapping):
            raise ConfigError("body: expected a JSON object")
        return dict(payload)

    def _output(self, name: str) -> None:
        try:
            path = _resolve_output(name)
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"no such output: {path.name}")
            return
        content_type = {
            ".png": "image/png",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".json": "application/json",
        }.get(path.suffix.lower(), "application/octet-stream")
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    def _upload(self, name: str) -> None:
        """Serve an uploaded source image (used as the trap's colour source)."""
        try:
            candidate = _safe_name(name)
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        path = _OUTPUT_DIR / "uploads" / candidate
        if not path.is_file():
            self._error(HTTPStatus.NOT_FOUND, f"no such upload: {candidate}")
            return
        suffix = path.suffix.lower()
        content_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
        }.get(suffix, "application/octet-stream")
        self._send(HTTPStatus.OK, path.read_bytes(), content_type)

    # -- routes ----------------------------------------------------------- #
    def do_GET(self) -> None:  # noqa: N802 - stdlib signature
        route = self.path.split("?", 1)[0]
        try:
            if route in ("/", "/index.html"):
                self._send(HTTPStatus.OK, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif route == "/api/schema":
                self._json(HTTPStatus.OK, build_schema())
            elif route == "/api/outputs":
                self._json(HTTPStatus.OK, {"outputs": list_outputs()})
            elif route == "/api/health":
                self._json(HTTPStatus.OK, {"ok": True, "outputs": str(_OUTPUT_DIR)})
            elif route.startswith("/outputs/uploads/"):
                self._upload(route[len("/outputs/uploads/") :])
            elif route.startswith("/outputs/"):
                self._output(route[len("/outputs/") :])
            else:
                self._error(HTTPStatus.NOT_FOUND, f"no route for {route}")
        except (ConfigError, ValueError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))

    def do_POST(self) -> None:  # noqa: N802 - stdlib signature
        route = self.path.split("?", 1)[0]
        try:
            payload = self._body()
        except RequestTooLarge as error:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, str(error))
            return
        except ValueError as error:
            # ConfigError and the safe-name guards both subclass ValueError.
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return

        try:
            if route == "/api/render":
                self._json(HTTPStatus.OK, render_request(payload))
            elif route == "/api/from-image":
                self._json(HTTPStatus.OK, import_image(payload))
            else:
                self._error(HTTPStatus.NOT_FOUND, f"no route for {route}")
        except ValueError as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
        except TimeoutError:
            self._error(HTTPStatus.GATEWAY_TIMEOUT, "render timed out; try a smaller size")
        except Exception as error:  # noqa: BLE001 - keep the server alive, report the reason
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(error).__name__}: {error}")


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], verbose: bool = False) -> None:
        super().__init__(address, _Handler)
        self.verbose = verbose


def create_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, verbose: bool = False) -> _Server:
    """Create (but do not start) the HTTP server; ``port=0`` picks a free port."""
    return _Server((host, port), verbose=verbose)


def serve(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, open_browser: bool = False, verbose: bool = True) -> None:
    """Serve the UI until interrupted."""
    server = create_server(host, port, verbose=verbose)
    address, actual_port = server.server_address[0], server.server_address[1]
    url = f"http://{address}:{actual_port}/"
    if verbose:
        print(f"fractalgen ui on {url}  (outputs: {_OUTPUT_DIR})")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        if verbose:
            print("\nstopped")
    finally:
        server.server_close()


# --------------------------------------------------------------------------- #
# page
# --------------------------------------------------------------------------- #
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fractalgen</title>
<style>
  :root { color-scheme: dark; --bg:#101318; --panel:#171b22; --line:#2a313c; --text:#e6e9ef; --muted:#93a0b4; --accent:#7cc4ff; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; background:var(--bg); color:var(--text); }
  header { display:flex; align-items:baseline; gap:12px; padding:12px 18px; border-bottom:1px solid var(--line); background:var(--panel); position:sticky; top:0; z-index:5; }
  header h1 { font-size:15px; margin:0; letter-spacing:.02em; }
  header .meta { color:var(--muted); font-size:12px; }
  main { display:grid; grid-template-columns: minmax(320px, 460px) 1fr; min-height: calc(100vh - 49px); }
  #controls { border-right:1px solid var(--line); padding:12px 14px 80px; overflow:auto; max-height: calc(100vh - 49px); }
  #stage { display:flex; flex-direction:column; gap:10px; padding:14px; }
  details { border:1px solid var(--line); border-radius:8px; margin-bottom:8px; background:var(--panel); }
  details > summary { cursor:pointer; padding:9px 11px; font-weight:600; font-size:13px; letter-spacing:.01em; }
  .fields { padding:4px 11px 12px; display:grid; gap:9px; }
  .field { display:grid; grid-template-columns: 1fr 132px; gap:8px; align-items:center; }
  .field label { font-size:12.5px; color:#cdd5e1; }
  .field .hint { grid-column:1/-1; color:var(--muted); font-size:11px; margin-top:-4px; }
  input[type=number], input[type=text], select { width:100%; background:#0d1015; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:5px 7px; font:inherit; font-size:12.5px; }
  input[type=range] { width:100%; }
  .row { display:flex; gap:6px; align-items:center; }
  .row input[type=number] { width:64px; }
  button { background:#22303f; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:6px 10px; font:inherit; font-size:12.5px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  button.primary { background:#1d4f7a; border-color:#2f6f9f; font-weight:600; }
  button.primary:disabled { opacity:.55; cursor:progress; }
  button.icon { padding:4px 7px; }
  #toolbar { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  #status { color:var(--muted); font-size:12px; min-height:16px; }
  #preview { flex:1; display:grid; place-items:center; border:1px solid var(--line); border-radius:10px; background:#0b0e12; min-height:340px; overflow:hidden; }
  #preview img { max-width:100%; max-height: calc(100vh - 190px); display:block; }
  #preview { grid-auto-rows: min-content; align-content: center; gap: 10px; }
  #preview .download { justify-self: center; color: var(--accent, #6ee7ff); font-size: 12.5px; text-decoration: none; border: 1px solid var(--line); padding: 6px 12px; border-radius: 8px; }
  #preview .download:hover { background: #131820; }
  #swatches { display:flex; height:26px; border-radius:6px; overflow:hidden; border:1px solid var(--line); }
  #swatches span { flex:1; }
  .listRow { border:1px dashed var(--line); border-radius:6px; padding:7px; display:grid; gap:6px; }
  .listRow header { all:unset; display:flex; gap:6px; align-items:center; }
  .chips { display:flex; gap:4px; flex-wrap:wrap; }
  .chip { border:1px solid var(--line); border-radius:999px; padding:2px 8px; font-size:11.5px; color:var(--muted); cursor:pointer; }
  .chip.on { border-color:var(--accent); color:var(--text); }
  #log { color:var(--muted); font-size:11.5px; white-space:pre-wrap; max-height:120px; overflow:auto; }
  a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>fractalgen</h1>
  <span class="meta">local only · 127.0.0.1</span>
</header>
<main>
  <section id="controls"></section>
  <section id="stage">
    <div id="toolbar">
      <button id="generate" class="primary">Generate</button>
      <button id="randomise" title="new random polynomial">New polynomial</button>
      <button id="download" title="save the current config">Download config</button>
      <label style="color:var(--muted);font-size:12px">form
        <select id="form" style="width:auto;margin-left:4px"><option value="image">image</option><option value="video">video</option></select>
      </label>
      <label style="color:var(--muted);font-size:12px">outputs
        <select id="outputs" style="width:auto;margin-left:4px;max-width:220px"><option value="">(none yet)</option></select>
      </label>
      <button id="open-output" title="show the selected output">Open</button>
      <span id="status"></span>
    </div>
    <div id="swatches"></div>
    <div id="preview"><span style="color:var(--muted)">no render yet</span></div>
    <div id="log"></div>
  </section>
</main>
<script>
const state = { schema: null, config: null, fields: new Map(), lists: new Map() };

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, kids = []) => {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'text') node.textContent = v;
    else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const kid of [].concat(kids)) if (kid) node.appendChild(kid);
  return node;
};

const get = (path) => path.split('.').reduce((node, key) => (node == null ? undefined : node[key]), state.config);
const set = (path, value) => {
  const keys = path.split('.');
  let node = state.config;
  for (const key of keys.slice(0, -1)) {
    if (node[key] == null || typeof node[key] !== 'object') node[key] = {};
    node = node[key];
  }
  if (value === undefined || value === null) delete node[keys.at(-1)];
  else node[keys.at(-1)] = value;
};

function log(message) {
  const line = el('div', { text: message });
  $('#log').prepend(line);
  while ($('#log').childElementCount > 40) $('#log').lastElementChild.remove();
}

// ---------------------------------------------------------------- controls
function numberInput(field, value, onChange) {
  const input = el('input', { type: 'number', value: value ?? '', step: field.step ?? 'any' });
  if (field.minimum !== undefined) input.min = field.minimum;
  if (field.maximum !== undefined) input.max = field.maximum;
  input.addEventListener('change', () => {
    const parsed = input.value === '' ? null : Number(input.value);
    onChange(parsed);
  });
  return input;
}

function control(field) {
  const value = get(field.path);
  const onChange = (next) => { set(field.path, next); renderSwatches(); };
  const wrap = el('div', { class: 'field' }, [el('label', { text: field.label })]);
  let input;

  if (field.kind === 'bool') {
    input = el('input', { type: 'checkbox' });
    input.checked = Boolean(value);
    input.addEventListener('change', () => onChange(input.checked));
  } else if (field.kind === 'enum') {
    input = el('select', {}, field.choices.map((choice) => {
      const [choiceValue, choiceLabel] = Array.isArray(choice) ? choice : [choice, choice];
      return el('option', { value: String(choiceValue), text: String(choiceLabel) });
    }));
    input.value = String(value ?? (Array.isArray(field.choices[0]) ? field.choices[0][0] : field.choices[0]));
    input.addEventListener('change', () => onChange(input.value));
  } else if (field.kind === 'complex') {
    const pair = Array.isArray(value) ? value : [1, 0];
    const re = numberInput(field, pair[0], (v) => onChange([v ?? 0, pair[1] ?? 0]));
    const im = numberInput(field, pair[1], (v) => onChange([pair[0] ?? 0, v ?? 0]));
    input = el('div', { class: 'row' }, [re, im]);
  } else if (field.kind === 'span') {
    const pair = Array.isArray(value) ? value : [-1, 1];
    const lo = numberInput(field, pair[0], (v) => onChange([v ?? -1, pair[1] ?? 1]));
    const hi = numberInput(field, pair[1], (v) => onChange([pair[0] ?? -1, v ?? 1]));
    input = el('div', { class: 'row' }, [lo, hi]);
  } else if (field.kind === 'colors') {
    input = el('input', { type: 'text', value: (value || []).join(', '), placeholder: '#112233, #445566' });
    input.addEventListener('change', () => {
      const colours = input.value.split(',').map((c) => c.trim()).filter(Boolean);
      onChange(colours.length ? colours : undefined);
    });
  } else if (field.kind === 'complex-list') {
    input = el('input', { type: 'text', value: (value || []).join(', '), placeholder: '1+0j, 0+1j' });
    input.addEventListener('change', () => {
      const items = input.value.split(',').map((c) => c.trim()).filter(Boolean);
      onChange(items.length ? items : undefined);
    });
  } else if (field.kind === 'list') {
    input = listControl(field);
  } else if (field.kind === 'int' || field.kind === 'float') {
    const slider = field.minimum !== undefined && field.maximum !== undefined && field.kind === 'float'
      ? el('input', { type: 'range', min: field.minimum, max: field.maximum, step: field.step ?? 0.01, value: value ?? field.minimum })
      : null;
    input = numberInput(field, value, onChange);
    if (slider) {
      slider.addEventListener('input', () => {
        input.value = slider.value;
        onChange(Number(slider.value));
      });
      input = el('div', { class: 'row' }, [slider, input]);
    }
  } else {
    input = el('input', { type: 'text', value: value ?? '', placeholder: field.optional ? 'not set' : '' });
    input.addEventListener('change', () => onChange(input.value === '' && field.optional ? undefined : input.value));
  }

  wrap.appendChild(input);
  if (field.help) wrap.appendChild(el('div', { class: 'hint', text: field.help }));
  return wrap;
}

function listControl(field) {
  const items = get(field.path) || [];
  const container = el('div', { style: 'display:grid;gap:6px;grid-column:1/-1' });
  const rebuild = () => {
    container.replaceChildren();
    items.forEach((item, index) => container.appendChild(listRow(field, items, index, rebuild)));
    const add = el('button', { class: 'icon', text: '+ add' });
    add.addEventListener('click', () => {
      items.push({ name: field.item.nameChoices[0] });
      set(field.path, items);
      rebuild();
    });
    container.appendChild(add);
  };
  rebuild();
  return container;
}

function listRow(field, items, index, rebuild) {
  const item = items[index];
  const nameChoices = field.item.nameChoices;
  const select = el('select', {}, nameChoices.map((name) => el('option', { value: name, text: name })));
  select.value = item.name;
  select.addEventListener('change', () => {
    items[index] = { name: select.value };
    set(field.path, items);
    rebuild();
  });
  const remove = el('button', { class: 'icon', text: '×', title: 'remove' });
  remove.addEventListener('click', () => {
    items.splice(index, 1);
    set(field.path, items.length ? items : undefined);
    rebuild();
  });
  const params = (field.item.params[item.name] || []);
  const rows = params.map((param) => {
    const current = item[param.name] !== undefined ? item[param.name] : param.default;
    const input = el('input', {
      type: param.kind === 'bool' ? 'checkbox' : (param.kind === 'string' ? 'text' : 'number'),
      step: 'any',
    });
    if (param.kind === 'bool') input.checked = Boolean(current);
    else input.value = current ?? '';
    input.addEventListener('change', () => {
      if (param.kind === 'bool') item[param.name] = input.checked;
      else if (param.kind === 'string') item[param.name] = input.value;
      else item[param.name] = input.value === '' ? undefined : Number(input.value);
      if (item[param.name] === undefined) delete item[param.name];
      set(field.path, items);
    });
    return el('label', { class: 'field', style: 'grid-template-columns:1fr 92px' }, [el('span', { text: param.name, style: 'color:var(--muted);font-size:11.5px' }), input]);
  });
  return el('div', { class: 'listRow' }, [el('div', { class: 'row' }, [select, remove]), ...rows]);
}

function buildControls() {
  const bySection = new Map();
  for (const field of state.schema.fields) {
    if (field.managed) continue;
    if (!bySection.has(field.section)) bySection.set(field.section, []);
    bySection.get(field.section).push(field);
  }
  const root = $('#controls');
  root.replaceChildren();
  // form selector drives the request; keep it in the toolbar, not the form
  const order = ['general', 'fractal', 'colour', 'image', 'postprocess', 'video', 'runtime'];
  const form = $('#form').value;
  for (const section of order) {
    const fields = bySection.get(section);
    if (!fields || !fields.length) continue;
    const meta = state.schema.sections.find((s) => s.id === section) || {};
    // Image renders have no timeline: hide video/animation controls entirely.
    if (Array.isArray(meta.forms) && !meta.forms.includes(form)) continue;
    const title = meta.title || section;
    const details = el('details', { open: section !== 'runtime' && section !== 'video' }, [el('summary', { text: title })]);
    const body = el('div', { class: 'fields' });
    for (const field of fields) {
      if (field.path === 'form') continue;
      body.appendChild(control(field));
    }
    details.appendChild(body);
    root.appendChild(details);
  }
}

// ---------------------------------------------------------------- preview
function renderSwatches() {
  const holder = $('#swatches');
  const colours = get('colour.colors');
  const mode = get('colour.mode');
  holder.replaceChildren();
  if (mode === 'colors' && Array.isArray(colours)) {
    for (const colour of colours) holder.appendChild(el('span', { style: `background:${colour}` }));
    return;
  }
  const name = get('colour.name') || 'viridis';
  holder.appendChild(el('span', { style: 'background:linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)', title: name }));
}

function showPreview(url, form, filename) {
  const holder = $('#preview');
  holder.replaceChildren();
  holder.appendChild(el('img', { src: url + '?t=' + Date.now(), alt: form === 'video' ? 'animation' : 'render' }));
  // Download the exact bytes the server produced (GIF animates when opened).
  const name = filename || decodeURIComponent(url.split('/').pop().split('?')[0]);
  holder.appendChild(el('a', {
    href: url,
    download: name,
    class: 'download',
    text: `Download ${name}`,
  }));
}

async function refreshOutputs(selectName) {
  const select = $('#outputs');
  try {
    const payload = await (await fetch('/api/outputs')).json();
    const outputs = payload.outputs || [];
    select.replaceChildren();
    if (!outputs.length) {
      select.appendChild(el('option', { value: '', text: '(none yet)' }));
      return;
    }
    for (const entry of outputs) {
      const label = `${entry.name} · ${(entry.bytes / 1024).toFixed(0)} kB`;
      select.appendChild(el('option', { value: entry.url, text: label, 'data-form': entry.kind }));
    }
    if (selectName) {
      const match = [...select.options].find((option) => option.value.endsWith(selectName));
      if (match) select.value = match.value;
    }
  } catch (error) {
    log(`could not list outputs: ${error.message || error}`);
  }
}

async function generate() {
  const button = $('#generate');
  button.disabled = true;
  $('#status').textContent = 'rendering…';
  const form = $('#form').value;
  const config = JSON.parse(JSON.stringify(state.config));
  for (const key of Object.keys(config)) if (config[key] === null) delete config[key];
  const started = performance.now();
  try {
    const response = await fetch('/api/render', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ form, config }),
    });
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || 'render failed');
    showPreview(payload.url, form, payload.filename);
    refreshOutputs(payload.filename);
    const seconds = payload.elapsed ?? (performance.now() - started) / 1000;
    $('#status').textContent = `done in ${Number(seconds).toFixed(2)}s`;
    log(`rendered ${payload.url} in ${Number(seconds).toFixed(2)}s`);
  } catch (error) {
    $('#status').textContent = 'error';
    log(String(error.message || error));
  } finally {
    button.disabled = false;
  }
}

// ---------------------------------------------------------------- image import
async function uploadImage(file) {
  $('#status').textContent = 'analysing image…';
  const buffer = await file.arrayBuffer();
  let binary = '';
  const bytes = new Uint8Array(buffer);
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  try {
    const response = await fetch('/api/from-image', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ filename: file.name, data: btoa(binary), colours: 6, raster_size: 96 }),
    });
    const payload = await response.json();
    if (!payload.ok) throw new Error(payload.error || 'analysis failed');
    set('colour.mode', 'colors');
    set('colour.colors', payload.colours);
    const raster = payload.raster_path || payload.raster_url;
    set('fractal.map.image', raster);
    set('fractal.map.strength', 1.4);
    // Trap colouring needs the picture's colours, not the luminance raster.
    set('fractal.orbit_trap.image', payload.source_path || raster);
    set('fractal.orbit_trap.scale', 2);
    set('fractal.orbit_trap.wrap', true);
    set('fractal.orbit_trap.colour_mode', 'closest');
    set('fractal.orbit_trap.size', 128);
    buildControls();
    renderSwatches();
    $('#status').textContent = `palette: ${payload.colours.length} colours`;
    log(`image ${file.name}: ${payload.colours.join(' ')}`);
  } catch (error) {
    $('#status').textContent = 'error';
    log(String(error.message || error));
  }
}

// ---------------------------------------------------------------- boot
async function boot() {
  const schema = await (await fetch('/api/schema')).json();
  state.schema = schema;
  state.config = schema.defaults;
  $('#form').value = state.config.form || 'image';
  $('#form').addEventListener('change', () => { set('form', $('#form').value); buildControls(); });
  $('#open-output').addEventListener('click', () => {
    const url = $('#outputs').value;
    if (!url) return;
    const option = $('#outputs').selectedOptions[0];
    showPreview(url, option ? option.dataset.form : 'image', decodeURIComponent(url.split('/').pop()));
  });
  refreshOutputs();

  const upload = el('input', { type: 'file', accept: 'image/*', style: 'display:none' });
  upload.addEventListener('change', () => { if (upload.files[0]) uploadImage(upload.files[0]); });
  const uploadButton = el('button', { text: 'Import image…' });
  uploadButton.addEventListener('click', () => upload.click());
  $('#toolbar').insertBefore(uploadButton, $('#status'));
  document.body.appendChild(upload);

  $('#generate').addEventListener('click', generate);
  $('#randomise').addEventListener('click', () => {
    set('fractal.polynomial.seed', Math.floor(Math.random() * 1_000_000));
    buildControls();
  });
  $('#download').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify(state.config, null, 2)], { type: 'application/json' });
    const link = el('a', { href: URL.createObjectURL(blob), download: 'fractalgen-config.json' });
    document.body.appendChild(link);
    link.click();
    link.remove();
  });

  buildControls();
  renderSwatches();
  if (state.schema.defaults.form) $('#form').value = state.schema.defaults.form;
}

boot().catch((error) => log(`boot failed: ${error.message || error}`));
</script>
</body>
</html>
"""


if __name__ == "__main__":  # pragma: no cover - manual entry point
    serve(open_browser=False)
