from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from fractalgen import Config, render_image
from fractalgen.config import ConfigError
from fractalgen.image_import import analyse_image, config_from_image, otsu_threshold, write_assets


def _photo(path: Path, size: int = 192) -> Path:
    """A blob image with continuous tones plus one hard edge - photo-ish enough."""
    grid = np.mgrid[0:size, 0:size] / size
    y, x = grid
    luma = np.zeros((size, size))
    for cx, cy, sigma, amplitude in ((0.3, 0.3, 0.16, 1.0), (0.7, 0.6, 0.12, 0.8)):
        luma += amplitude * np.exp(-(((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2)))
    luma = (luma - luma.min()) / (luma.max() - luma.min())
    rgb = np.dstack([luma**0.8, np.roll(luma, 12, axis=1) ** 1.2, 1 - luma**0.6])
    image = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    ImageDraw.Draw(image).line([(0, size - 1), (size - 1, 0)], fill=(12, 30, 90), width=6)
    image.save(path)
    return path


def _base_config(tmp_path: Path) -> Config:
    return Config.from_dict(
        {
            "fractal": {
                "size": 96,
                "iterations": 30,
                "polynomial": {"mode": "roots", "roots": [[1, 0], [-1, 0], [0, 1], [0, -1]]},
            },
            "colour": {"mode": "matplotlib", "name": "viridis"},
            "image": {"return_type": "iteration"},
            "output": str(tmp_path / "out" / "fractal.png"),
        },
        base_dir=tmp_path,
    )


def test_analysis_shapes_ranges_and_palette(tmp_path: Path):
    source = _photo(tmp_path / "photo.png")
    analysis = analyse_image(source, colours=5, raster_size=48)
    assert analysis.raster.shape == (48, 48)
    assert analysis.raster.dtype == np.uint8
    assert analysis.raster.min() < analysis.raster.max()
    assert len(analysis.colours) == 5
    assert all(colour.startswith("#") and len(colour) == 7 for colour in analysis.colours)
    assert analysis.palette.mode == "colors"
    assert analysis.palette.colors == analysis.colours
    assert analysis.trap_bitmap.any() and not analysis.trap_bitmap.all()
    assert analysis.source_size == (192, 192)


def test_analysis_rejects_bad_input(tmp_path: Path):
    with pytest.raises(ConfigError, match="not found"):
        analyse_image(tmp_path / "missing.png")
    source = _photo(tmp_path / "photo.png")
    with pytest.raises(ConfigError, match="colours"):
        analyse_image(source, colours=1)
    with pytest.raises(ConfigError, match="raster_size"):
        analyse_image(source, raster_size=2)


def test_otsu_threshold_separates_two_levels():
    values = np.concatenate([np.full(500, 20), np.full(500, 220)])
    threshold = otsu_threshold(values)
    assert 20 < threshold < 220


def test_assets_are_written(tmp_path: Path):
    analysis = analyse_image(_photo(tmp_path / "photo.png"), colours=4, raster_size=32)
    raster_path, palette_path = write_assets(analysis, tmp_path / "assets", stem="shot")
    assert raster_path.exists() and palette_path.exists()
    assert Image.open(raster_path).size == (32, 32)
    import json

    payload = json.loads(palette_path.read_text())
    assert payload["colours"] == list(analysis.colours)
    assert payload["raster"] == raster_path.name


def test_config_from_image_keeps_settings_and_wires_the_map(tmp_path: Path):
    source = _photo(tmp_path / "photo.png")
    base = _base_config(tmp_path)
    config = config_from_image(base, source, colours=4, raster_size=48, strength=1.2)

    assert config.fractal.size == base.fractal.size
    assert config.fractal.polynomial == base.fractal.polynomial
    assert config.colour.mode == "colors"
    assert config.fractal.map.enabled
    assert config.fractal.map.strength == pytest.approx(1.2)
    assert config.fractal.map.blend == pytest.approx(0.8)
    assert config.fractal.orbit_trap.image is not None
    assert Path(config.fractal.orbit_trap.image).exists()
    assert config.output == base.output


def test_render_shows_the_map(tmp_path: Path):
    """The map modifier must change the frame, and not by a rounding error."""
    source = _photo(tmp_path / "photo.png")
    base = _base_config(tmp_path)
    plain = render_image(base).rgb
    mapped = render_image(config_from_image(base, source, colours=4, raster_size=48, strength=1.4)).rgb
    difference = np.abs(plain.astype(float) - mapped.astype(float))
    assert difference.mean() > 2.0, "the image map should visibly alter the fractal"

    # The distance field should follow the raster, i.e. be non-uniform where the
    # source has structure - otherwise the "map" is not doing anything positional.
    config = config_from_image(base, source, colours=4, raster_size=48, strength=1.4)
    from fractalgen import choose_polynomial, solve
    from fractalgen.solver import _map_raster

    result = solve(config.fractal, choose_polynomial(config.fractal, progress=False), workers=1)
    assert np.isfinite(result.roots).all()
    raster = _map_raster(str(config.fractal.map.image), config.fractal.map.size)
    assert raster.std() > 0.05


def test_blend_makes_the_image_readable(tmp_path: Path):
    """The image-space blend must dominate the difference with a plain fractal."""
    from fractalgen.solver import _map_raster

    source = _photo(tmp_path / "photo.png")
    base = _base_config(tmp_path)
    mapped = config_from_image(base, source, colours=4, raster_size=48, strength=0.0, blend=0.9)

    plain = render_image(_base_config(tmp_path)).rgb.astype(float).mean(axis=2)
    blended = render_image(mapped).rgb.astype(float).mean(axis=2)
    raster = _map_raster(str(mapped.fractal.map.image), mapped.fractal.map.size)
    # The blend resizes the raster to the frame, so compare on a common grid.
    from PIL import Image

    raster = np.asarray(
        Image.fromarray((raster * 255).astype(np.uint8)).resize(blended.shape[::-1], Image.Resampling.BICUBIC),
        dtype=float,
    ) / 255.0

    def correlation(a: np.ndarray, b: np.ndarray) -> float:
        a = (a - a.mean()) / (a.std() or 1.0)
        b = (b - b.mean()) / (b.std() or 1.0)
        return float((a * b).mean())

    plain_correlation = abs(correlation(raster, plain))
    blended_correlation = correlation(raster, blended)
    assert blended_correlation > 0.55
    assert blended_correlation > plain_correlation


def test_cli_from_image_end_to_end(tmp_path: Path):
    from fractalgen.cli import main

    source = _photo(tmp_path / "photo.png")
    output = tmp_path / "out" / "fractal.png"
    exit_code = main(
        ["from-image", str(source), "-o", str(output), "--colours", "4", "--raster-size", "48", "--strength", "1.2"]
    )
    assert exit_code == 0
    assert output.exists()
    assert (output.parent / "photo_raster.png").exists()
    assert (output.parent / "photo_palette.json").exists()

    assert main(["from-image", str(source), "-o", str(tmp_path / "x.png"), "--raster-only"]) == 0
    assert not (tmp_path / "x.png").exists()


def test_cli_processes_a_whole_folder(tmp_path: Path):
    from fractalgen.cli import main

    folder = tmp_path / "photos"
    folder.mkdir()
    _photo(folder / "one.png")
    _photo(folder / "two.png")
    (folder / "notes.txt").write_text("ignored")

    output_dir = tmp_path / "renders"
    output_dir.mkdir()
    assert main(["from-image", str(folder), "-o", str(output_dir), "--colours", "3", "--raster-size", "32"]) == 0

    assert (output_dir / "one.png").exists()
    assert (output_dir / "two.png").exists()
    assert (output_dir / "one_raster.png").exists()
    assert (output_dir / "two_palette.json").exists()
    assert not (output_dir / "notes.png").exists()


def test_cli_reports_an_empty_folder(tmp_path: Path, capsys):
    from fractalgen.cli import main

    empty = tmp_path / "photos"
    empty.mkdir()
    assert main(["from-image", str(empty)]) == 2
    assert "no images found" in capsys.readouterr().err


def test_cli_missing_source_is_reported(tmp_path: Path, capsys):
    from fractalgen.cli import main

    assert main(["from-image", str(tmp_path / "nope")]) == 2
    assert "no such file or directory" in capsys.readouterr().err
