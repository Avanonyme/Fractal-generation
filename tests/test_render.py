from pathlib import Path

import numpy as np
import pytest

from fractalgen import render
from fractalgen.colour import (
    analogous,
    blend,
    build_colormap,
    complementary,
    hex_to_rgb,
    rgb_to_hex,
    sample_colormap,
    shades,
    split_complementary,
    tetradic,
    triadic,
)
from fractalgen.config import ColourConfig, ConfigError


def test_normalise_handles_constant_and_nan():
    constant = render.normalise(np.full((4, 4), 7.0))
    assert np.all(constant == 0)
    with_nan = render.normalise(np.array([[0.0, 1.0], [np.nan, 2.0]]))
    assert np.isfinite(with_nan).all()
    assert with_nan.max() == pytest.approx(1.0)


def test_colorize_shapes():
    colormap = build_colormap(ColourConfig(), Path("."))
    rgb = render.colorize(np.arange(16).reshape(4, 4), colormap)
    assert rgb.shape == (4, 4, 3)
    assert rgb.dtype == np.uint8

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    assert render.colorize(rgba, colormap).shape == (4, 4, 3)


def test_save_and_load_roundtrip(tmp_path: Path):
    image = np.random.default_rng(0).integers(0, 255, (8, 8, 3), dtype=np.uint8)
    path = render.save(image, tmp_path / "nested" / "image.png")
    assert path.exists()
    loaded = render.load(path)
    assert loaded.shape == (8, 8, 4)
    assert np.array_equal(loaded[..., :3], image)


def test_compose_over_alpha_zero_is_background():
    foreground = np.full((4, 4, 3), 255, dtype=np.uint8)
    background = np.zeros((4, 4, 3), dtype=np.uint8)
    assert render.compose_over(foreground, background, 0).max() == 0
    assert render.compose_over(foreground, background, 255).min() == 255


def test_centre_crop_is_centred():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[3:5, 3:5] = 255
    cropped = render.resize(image[2:6, 2:6], 8)
    assert cropped[3:5, 3:5].mean() > 0


def test_colour_helpers():
    assert rgb_to_hex([255, 0, 16]) == "#FF0010"
    assert hex_to_rgb("#FF0010") == (255, 0, 16)
    assert blend((0, 0, 0), (100, 200, 40), 0.5) == (50, 100, 20)
    assert complementary(["#000000"]) == ["#FFFFFF"]
    assert analogous(["#FF0000"])[0].startswith("#")
    assert len(triadic(["#FF0000"])[0]) == 1
    assert len(split_complementary(["#FF0000"])[1]) == 1
    assert len(tetradic(["#FF0000"])) == 4
    assert len(shades(["#FF0000"], steps=3)) == 4


def test_build_colormap_modes():
    explicit = build_colormap(ColourConfig(mode="colors", colors=("#ff0000", "#00ff00")), Path("."))
    assert np.allclose(explicit(0.0)[:3], (1.0, 0.0, 0.0))
    assert len(sample_colormap(build_colormap(ColourConfig(), Path(".")), 5)) == 5


def test_build_colormap_rejects_unknown_name():
    with pytest.raises(ConfigError, match="unknown colormap"):
        build_colormap(ColourConfig(name="not-a-colormap"), Path("."))


def test_accents_without_image_is_rejected():
    with pytest.raises(ConfigError, match="colour.image"):
        build_colormap(ColourConfig(mode="accents"), Path("."))
