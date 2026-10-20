import numpy as np
import pytest
from scipy import ndimage

from fractalgen import postprocess, render
from fractalgen.config import Config, ConfigError, PostprocessConfig


def _banded_image(size: int = 128) -> np.ndarray:
    """Concentric colour steps on a radial ramp: the ring artifact, synthetically."""
    coordinates = np.arange(size) - size / 2
    grid_x, grid_y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(grid_x, grid_y)
    # Palette-sized steps (~4 levels), which is what an iteration band looks like.
    steps = np.floor(radius / 6.0)
    values = 60 + (steps % 24) * 4
    rgb = np.dstack([values, values * 0.9 + 20, 240 - values * 0.5])
    return render.to_uint8(rgb)


def _flat_ring_energy(image: np.ndarray) -> float:
    """Mean Laplacian energy inside flat regions (rings live there)."""
    gray = np.asarray(image, dtype=float).mean(axis=2)
    laplacian = np.abs(
        np.gradient(np.gradient(gray, axis=0), axis=0) + np.gradient(np.gradient(gray, axis=1), axis=1)
    )
    local_std = ndimage.generic_filter(gray, np.std, size=7, mode="nearest")
    flat = local_std < 5
    return float(laplacian[flat].mean()) if flat.any() else float(laplacian.mean())


def test_deband_reduces_rings():
    ringed = _banded_image()
    cleaned = postprocess.deband(ringed, {})
    assert _flat_ring_energy(cleaned) < 0.6 * _flat_ring_energy(ringed)
    assert cleaned.shape == ringed.shape and cleaned.dtype == np.uint8


def test_deband_keeps_real_edges():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :32] = 20
    image[:, 32:] = 235
    edge = float(np.abs(np.asarray(postprocess.deband(image, {}), dtype=float)[:, 31] - 235).mean())
    assert edge > 150, "a hard edge must survive the de-banding pass"


def test_unsharp_restores_contrast():
    blurred = ndimage.gaussian_filter(_banded_image().astype(float), 2.0)
    sharpened = postprocess.unsharp(render.to_uint8(blurred), {"amount": 0.6})
    assert np.asarray(sharpened, dtype=float).std() > np.asarray(render.to_uint8(blurred), dtype=float).std()


def test_default_config_has_the_banded_look_with_debanding():
    config = Config()
    assert config.image.return_type == "iteration"
    assert [step.name for step in config.image.postprocess] == ["deband", "unsharp"]
    assert config.to_dict()["image"]["postprocess"][0]["name"] == "deband"
    assert Config.from_dict(config.to_dict()).image.postprocess == config.image.postprocess


def test_postprocess_is_configurable_and_strict():
    config = Config.from_dict({"image": {"postprocess": ["deband", {"name": "unsharp", "amount": 0.2}]}})
    assert config.image.postprocess[1].params == {"amount": 0.2}
    assert Config.from_dict({"image": {"postprocess": []}}).image.postprocess == ()

    with pytest.raises(ConfigError, match="expected one of"):
        PostprocessConfig.from_dict("sharpen", "postprocess")
    with pytest.raises(ConfigError, match="unknown parameter"):
        postprocess.resolve_params("deband", {"radii": 3})


def test_postprocess_runs_in_the_pipeline(tmp_path):
    from fractalgen import render_image

    base = {
        "fractal": {"size": 64, "iterations": 20, "polynomial": {"mode": "roots", "roots": [[1, 0], [-1, 0], [0, 1]]}},
        "colour": {"mode": "matplotlib", "name": "viridis"},
    }
    raw = render_image(Config.from_dict({**base, "image": {"return_type": "iteration", "postprocess": []}})).rgb
    filtered = render_image(Config.from_dict(base)).rgb
    assert raw.shape == filtered.shape
    assert not np.array_equal(raw, filtered), "the default filters must change the frame"
