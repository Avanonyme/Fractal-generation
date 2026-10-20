"""Optional dependency probing.

Features that are nice but not required (Canny edges, seaborn palettes, image
palette extraction, MP4 encoding) are probed once here instead of being wrapped
in ``try: import ... except ImportError: pass`` at every call site. Missing
features raise :class:`~fractalgen.config.ConfigError` with the exact pip hint.
"""

from __future__ import annotations

from functools import lru_cache

from .config import ConfigError

__all__ = ["available", "require", "FEATURE_HINTS"]

#: feature -> (importable module, pip extra)
FEATURE_HINTS = {
    "edges": ("skimage", "fractal-generation[edges]"),
    "palette": ("extcolors", "fractal-generation[palette]"),
    "seaborn": ("seaborn", "fractal-generation[palette]"),
    "mp4": ("imageio", "fractal-generation[video]"),
}


@lru_cache(maxsize=None)
def available(feature: str) -> bool:
    """True when the module backing ``feature`` can be imported."""
    module = FEATURE_HINTS.get(feature, (feature, None))[0]
    try:
        import importlib.util

        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installation
        return False


def require(feature: str) -> None:
    """Raise a helpful error when ``feature`` is unavailable."""
    if available(feature):
        return
    _, extra = FEATURE_HINTS.get(feature, (feature, feature))
    module = FEATURE_HINTS.get(feature, (feature, None))[0]
    raise ConfigError(f"this option needs {module!r}: pip install '{extra}'")
