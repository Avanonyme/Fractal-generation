import json
from pathlib import Path

import pytest

from fractalgen.config import (
    ColourConfig,
    Config,
    ConfigError,
    Domain,
    FractalConfig,
    load_config,
    save_config,
)


def test_defaults_are_valid():
    config = Config()
    assert config.fractal.size == 800
    assert config.fractal.method == "newton"
    assert config.form == "image"
    assert config.colour.mode == "matplotlib"


def test_roundtrip_through_to_dict():
    config = Config.from_dict(
        {
            "fractal": {"size": 128, "method": "halley", "damping": [0.5, -0.25]},
            "colour": {"mode": "colors", "colors": ["#ff0000", "#00ff00"]},
            "image": {"return_type": "distance"},
        }
    )
    assert config.fractal.damping == complex(0.5, -0.25)
    assert config.image.return_type == "distance"
    assert Config.from_dict(config.to_dict()).to_dict() == config.to_dict()


def test_unknown_key_is_reported_with_suggestion():
    with pytest.raises(ConfigError) as error:
        Config.from_dict({"fractal": {"itermax": 30}})
    message = str(error.value)
    assert "unknown key" in message
    assert "iterations" in message


def test_bad_values_are_rejected():
    with pytest.raises(ConfigError, match="must be >= 16"):
        FractalConfig.from_dict({"size": 8})
    with pytest.raises(ConfigError, match="expected an integer"):
        FractalConfig.from_dict({"size": "big"})
    with pytest.raises(ConfigError, match="min must be < max"):
        Domain.from_dict([[1, 0], [-1, 1]])


def test_domain_zoom_keeps_centre():
    domain = Domain(-2, 2, -1, 1)
    zoomed = domain.zoomed(0.5, 1 + 0j)
    assert zoomed.width == pytest.approx(2.0)
    assert zoomed.center == pytest.approx(1 + 0j)
    assert zoomed.x_min == pytest.approx(0.0)


def test_load_and_save(tmp_path: Path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"fractal": {"size": 96}, "output": "img.png"}))
    config = load_config(path)
    assert config.fractal.size == 96
    assert config.resolve_path(config.output) == tmp_path / "img.png"

    written = save_config(config, tmp_path / "copy.json")
    assert load_config(written).fractal.size == 96


def test_missing_file_and_bad_json(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{nope}")
    with pytest.raises(ConfigError, match="invalid JSON"):
        load_config(broken)


def test_colour_modes():
    assert ColourConfig.from_dict({"mode": "colors", "colors": [[1, 2, 3]]}).colors == ("#010203",)
    with pytest.raises(ConfigError, match="required"):
        ColourConfig.from_dict({"mode": "colors"})


def test_animation_and_video_defaults():
    config = Config.from_dict({"video": {"fps": 10, "duration": 2.5, "animations": ["pulse", {"name": "alpha"}]}})
    assert config.video.frame_count == 25
    assert config.video.animation_names == ("pulse", "alpha")
    with pytest.raises(ConfigError, match="expected one of"):
        Config.from_dict({"video": {"animations": ["wobble"]}})
