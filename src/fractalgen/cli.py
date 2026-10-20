"""Command line interface for the fractal generator.

    fractalgen render config.json          # render image or video
    fractalgen image --fractal.size 1024   # one-shot render with overrides
    fractalgen video --output out.gif --video.animations pulse flicker
    fractalgen schema                      # JSON schema of the config model
    fractalgen example                     # print a commented example config

Overrides use the dotted form ``--section.key value`` and are typed against the
field they target, so ``--fractal.size 900`` is an integer and
``--colour.colors '#112233' '#445566'`` is a list.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from dataclasses import MISSING, fields, is_dataclass
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints

from .config import Config, ConfigError, load_config
from .pipeline import render_image, render_video

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fractalgen",
        description="Root-finding fractal renderer (images and animations).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (("render", "render a config file"), ("image", "render one image"), ("video", "render an animation")):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("config", nargs="?", help="JSON config file")
        command.add_argument("-o", "--output", help="output path (overrides config)")
        command.add_argument("-w", "--workers", type=int, help="worker processes (0 = all cores)")
        command.add_argument("-v", "--verbose", action="store_true", help="print progress")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
        command.add_argument("--fullscreen", action="store_true", help=argparse.SUPPRESS)

    from_image = subparsers.add_parser(
        "from-image", help="derive a palette + map modifier from an image, then render"
    )
    from_image.add_argument("image", help="source image (jpg/png/...)")
    from_image.add_argument("-o", "--output", help="output path (also where the raster assets go)")
    from_image.add_argument("--base", help="base JSON config to start from")
    from_image.add_argument("--colours", type=int, default=6, help="number of palette colours")
    from_image.add_argument("--raster-size", type=int, default=96, help="raster resolution")
    from_image.add_argument("--strength", type=float, default=1.4, help="map-modifier strength")
    from_image.add_argument("--blend", type=float, default=0.8, help="image-space blend (0..1)")
    from_image.add_argument("--raster-only", action="store_true", help="write the assets, skip the render")
    from_image.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    from_image.add_argument("-w", "--workers", type=int, help="worker processes (0 = all cores)")
    from_image.add_argument("-v", "--verbose", action="store_true")

    schema = subparsers.add_parser("schema", help="print the configuration JSON schema")
    schema.add_argument("--indent", type=int, default=2)

    example = subparsers.add_parser("example", help="print a commented example configuration")
    example.add_argument("--form", choices=["image", "video"], default="image")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)

    try:
        if arguments.command == "schema":
            print(json.dumps(config_schema(), indent=arguments.indent, sort_keys=False))
            return 0
        if arguments.command == "example":
            print(json.dumps(example_config(arguments.form), indent=2))
            return 0
        return _render(arguments)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return 2
    except FileNotFoundError as error:
        print(f"missing file: {error}", file=sys.stderr)
        return 2


def _render(arguments: argparse.Namespace) -> int:
    if arguments.command == "from-image":
        return _render_from_image(arguments)

    config = _load(arguments)
    if config.form == "video":
        if config.output is None:
            raise ConfigError("video.output: no output path configured")
        path = render_video(config)
        print(f"wrote {path}")
        return 0

    rendered = render_image(config)
    if rendered.path is not None:
        print(f"wrote {rendered.path}")
    else:
        print(f"rendered {rendered.rgb.shape[1]}x{rendered.rgb.shape[0]} frame (no output path)")
    return 0


def _render_from_image(arguments: argparse.Namespace) -> int:
    """Build a config from an image (palette + raster map), then render it."""
    from .image_import import analyse_image, config_from_image, write_assets

    base = load_config(arguments.base) if arguments.base else Config.from_dict(example_config("image"))
    target = Path(arguments.output).expanduser().resolve() if arguments.output else base.resolve_output()
    if target is None:
        target = Path.cwd() / "out" / "fractal.png"

    analysis = analyse_image(arguments.image, colours=arguments.colours, raster_size=arguments.raster_size)
    name = Path(arguments.image).stem
    raster_path, palette_path = write_assets(analysis, target.parent, stem=name)
    print(f"wrote {raster_path}")
    print(f"wrote {palette_path}")

    config = config_from_image(
        base,
        arguments.image,
        colours=arguments.colours,
        raster_size=arguments.raster_size,
        strength=arguments.strength,
        blend=arguments.blend,
        output=target,
        stem=name,
    )
    for item in arguments.set or []:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        config = _apply_override(config, key, _parse_override(raw))
    if arguments.workers is not None:
        config = _apply_override(config, "runtime.workers", arguments.workers)
    if arguments.verbose:
        config = _apply_override(config, "runtime.verbose", True)

    if arguments.raster_only:
        return 0

    rendered = render_image(config)
    if rendered.path is not None:
        print(f"wrote {rendered.path}")
    return 0


def _load(arguments: argparse.Namespace) -> Config:
    if arguments.config:
        config = load_config(arguments.config)
    else:
        form = "video" if arguments.command == "video" else "image"
        config = Config.from_dict(example_config(form), base_dir=Path.cwd())

    overrides: dict[str, Any] = {}
    for item in arguments.set or []:
        if "=" not in item:
            raise ConfigError(f"--set expects KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key] = _parse_override(raw)

    if arguments.output:
        overrides["output"] = arguments.output
    if arguments.workers is not None:
        overrides["runtime.workers"] = arguments.workers
    if arguments.verbose:
        overrides["runtime.verbose"] = True
    if arguments.command == "video":
        overrides.setdefault("form", "video")

    for key, value in overrides.items():
        config = _apply_override(config, key, value)
    return config


def _parse_override(raw: str) -> Any:
    """Interpret an override value: JSON if possible, else a plain string."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw


def _apply_override(config: Config, dotted: str, value: Any) -> Config:
    """Apply ``section.key=value`` onto the immutable config tree."""
    parts = dotted.split(".")
    data = config.to_dict()
    node: dict[str, Any] = data
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            raise ConfigError(f"--set {dotted}: no such section {part!r}")
        node = node[part]
    if parts[-1] not in node:
        raise ConfigError(f"--set {dotted}: unknown field {parts[-1]!r}")
    node[parts[-1]] = value
    return Config.from_dict(data, base_dir=config.base_dir)


# --------------------------------------------------------------------------- #
# introspection helpers (used by ``fractalgen schema`` and ``fractalgen example``)
# --------------------------------------------------------------------------- #
_SIMPLE_TYPES = {
    int: "integer",
    float: "number",
    bool: "boolean",
    str: "string",
    complex: "complex",
    Path: "path",
}


def config_schema() -> dict[str, Any]:
    """A JSON-schema-like description of the configuration model."""
    return _describe(Config)


def _describe(model: type) -> dict[str, Any]:
    hints = get_type_hints(model)
    properties: dict[str, Any] = {}
    for item in fields(model):
        default = None if item.default is MISSING else item.default
        properties[item.name] = _describe_field(hints.get(item.name, item.type), default)
    return {"title": model.__name__, "type": "object", "properties": properties}


def _describe_field(annotation: Any, default: Any) -> dict[str, Any]:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return _describe(annotation)

    origin = get_origin(annotation)
    arguments = get_args(annotation)

    if origin in (Union, types.UnionType):
        optional = type(None) in arguments
        described = next(
            (_describe_field(argument, default) for argument in arguments if argument is not type(None)), {"type": "any"}
        )
        if optional:
            described = {**described, "nullable": True}
        return described
    if origin in (tuple, list):
        item = _describe_field(arguments[0], None) if arguments else {"type": "any"}
        return {"type": "array", "items": item}
    if annotation in _SIMPLE_TYPES:
        entry: dict[str, Any] = {"type": _SIMPLE_TYPES[annotation]}
        if isinstance(default, (int, float, bool, str)):
            entry["default"] = default
        return entry
    return {"type": str(annotation).replace("typing.", "")}


def example_config(form: str = "image") -> dict[str, Any]:
    """A minimal, fully explicit configuration for the given form."""
    data = Config.from_dict({}).to_dict()
    data["form"] = form
    data["output"] = "out.gif" if form == "video" else "out.png"
    data["fractal"] = {
        "size": 600,
        "method": "newton",
        "variant": "standard",
        "iterations": 30,
        "tolerance": 1e-05,
        "damping": [1.0, 0.0],
        "polynomial": {"mode": "random", "max_degree": 6, "seed": 7},
        "orbit_trap": {"image": None, "distance_mode": 1},
    }
    data["colour"] = {"mode": "matplotlib", "name": "viridis"}
    data["image"] = {"return_type": "iteration", "shading": {"type": "matplotlib"}}
    if form == "video":
        data["video"] = {"fps": 20, "duration": 3, "animations": [{"name": "alpha"}], "codec": "gif"}
    else:
        data["video"] = {"fps": 20, "duration": 3, "animations": []}
    return data


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
