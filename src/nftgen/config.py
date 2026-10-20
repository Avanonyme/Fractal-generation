"""Configuration for an NFT collection.

The document mirrors :class:`fractalgen.config.Config` for everything that
concerns a single fractal, and adds collection-level fields::

    {
      "collection": {"name": "Fractals", "description": "...", "size": 10, "seed": 42},
      "output": {"directory": "nft", "metadata": true},
      "composition": {"background": "background.png", "scale": 0.55, "rotation": [-12, 12]},
      "fractal": {...}, "colour": {...}, "image": {...}
    }
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from fractalgen._numeric import as_float, as_int
from fractalgen.config import ColourConfig, Config, ConfigError, FractalConfig, ImageConfig

__all__ = ["CollectionConfig", "CompositionConfig", "NftConfig", "load_nft_config"]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"{path}: expected an object, got {type(value).__name__}")
    return value


def _check(data: Mapping[str, Any], allowed: Sequence[str], path: str) -> None:
    unknown = sorted(set(data) - set(allowed))
    if unknown:
        raise ConfigError(f"{path}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def _span(value: Any, default: tuple[float, float], path: str) -> tuple[float, float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        fixed = as_float(value, path)
        return fixed, fixed
    if isinstance(value, Sequence) and len(value) == 2:
        return as_float(value[0], f"{path}[0]"), as_float(value[1], f"{path}[1]")
    raise ConfigError(f"{path}: expected a number or [min, max]")


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    name: str = "Fractal"
    description: str = "Procedurally generated root-finding fractal."
    size: int = 10
    start_index: int = 1
    seed: int | None = None

    @classmethod
    def from_dict(cls, value: Any, path: str = "collection") -> "CollectionConfig":
        data = _mapping(value, path)
        _check(data, ["name", "description", "size", "start_index", "seed"], path)
        default = cls()
        size = as_int(data.get("size", default.size), f"{path}.size")
        if size < 1:
            raise ConfigError(f"{path}.size: must be >= 1")
        return cls(
            name=str(data.get("name", default.name)),
            description=str(data.get("description", default.description)),
            size=size,
            start_index=as_int(data.get("start_index", default.start_index), f"{path}.start_index"),
            seed=None if data.get("seed") is None else as_int(data["seed"], f"{path}.seed"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "size": self.size,
            "start_index": self.start_index,
            "seed": self.seed,
        }


@dataclass(frozen=True, slots=True)
class CompositionConfig:
    """How the fractal is placed on the background canvas."""

    background: Path | None = None
    scale: tuple[float, float] = (0.55, 0.55)
    rotation: tuple[float, float] = (0.0, 0.0)
    flip: bool = True
    offset: tuple[float, float] = (0.0, 0.0)

    @classmethod
    def from_dict(cls, value: Any, path: str = "composition") -> "CompositionConfig":
        data = _mapping(value, path)
        _check(data, ["background", "scale", "rotation", "flip", "offset"], path)
        default = cls()
        background = data.get("background", None)
        return cls(
            background=None if background is None else Path(str(background)).expanduser(),
            scale=_span(data.get("scale"), default.scale, f"{path}.scale"),
            rotation=_span(data.get("rotation"), default.rotation, f"{path}.rotation"),
            flip=bool(data.get("flip", default.flip)),
            offset=_span(data.get("offset"), default.offset, f"{path}.offset"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "background": None if self.background is None else str(self.background),
            "scale": list(self.scale),
            "rotation": list(self.rotation),
            "flip": self.flip,
            "offset": list(self.offset),
        }


@dataclass(frozen=True, slots=True)
class NftConfig:
    collection: CollectionConfig = field(default_factory=CollectionConfig)
    composition: CompositionConfig = field(default_factory=CompositionConfig)
    render: Config = field(default_factory=Config)
    directory: Path = Path("nft")
    write_metadata: bool = True
    base_dir: Path = field(default=Path("."))

    @property
    def fractal(self) -> FractalConfig:
        return self.render.fractal

    @property
    def colour(self) -> ColourConfig:
        return self.render.colour

    @property
    def image(self) -> ImageConfig:
        return self.render.image

    def token_config(self, index: int, seed: int) -> Config:
        """Per-token render config: same style, distinct random polynomial."""
        fractal = self.render.fractal
        polynomial = replace(fractal.polynomial, seed=seed)
        if fractal.polynomial.mode != "random":
            polynomial = fractal.polynomial
        return replace(
            self.render,
            fractal=replace(fractal, polynomial=polynomial),
            runtime=replace(self.render.runtime, seed=seed),
            output=None,
        )

    def resolve(self, path: Path | None) -> Path | None:
        """Resolve an *input* asset against the collection config's directory."""
        if path is None:
            return None
        return path if path.is_absolute() else self.base_dir / path

    def resolve_output(self) -> Path:
        """The output directory, resolved against the working directory."""
        return self.directory if self.directory.is_absolute() else Path.cwd() / self.directory

    def to_dict(self) -> dict[str, Any]:
        return {
            "collection": self.collection.to_dict(),
            "composition": self.composition.to_dict(),
            "output": {"directory": str(self.directory), "metadata": self.write_metadata},
            "fractal": self.render.fractal.to_dict(),
            "colour": self.render.colour.to_dict(),
            "image": self.render.image.to_dict(),
        }


def load_nft_config(path: str | Path) -> NftConfig:
    path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path}: invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc

    data = _mapping(raw, "config")
    _check(data, ["collection", "composition", "output", "fractal", "colour", "image", "video", "runtime"], "config")
    output = _mapping(data.get("output"), "output")
    directory = Path(str(output.get("directory", "nft"))).expanduser()
    metadata = bool(output.get("metadata", True))

    render = Config.from_dict(
        {
            "form": "image",
            "fractal": data.get("fractal", {}),
            "colour": data.get("colour", {}),
            "image": data.get("image", {}),
            "runtime": data.get("runtime", {}),
        },
        base_dir=path.parent,
    )
    return NftConfig(
        collection=CollectionConfig.from_dict(data.get("collection"), "collection"),
        composition=CompositionConfig.from_dict(data.get("composition"), "composition"),
        render=render,
        directory=directory,
        write_metadata=metadata,
        base_dir=path.parent,
    )
