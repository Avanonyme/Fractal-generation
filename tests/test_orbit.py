import numpy as np
import pytest

from fractalgen.config import Domain
from fractalgen.orbit import OrbitTrap


def _half_bitmap(size: int = 64, left_bright: bool = True) -> np.ndarray:
    bitmap = np.zeros((size, size), dtype=float)
    if left_bright:
        bitmap[:, : size // 2] = 255
    else:
        bitmap[:, size // 2 :] = 255
    return bitmap


def test_trap_orientation_matches_image_axes():
    """Columns are the real axis, rows the imaginary axis - not transposed."""
    trap = OrbitTrap.from_bitmap(_half_bitmap(left_bright=True))
    left = trap.sample(np.array([-0.5 + 0j]), Domain())
    right = trap.sample(np.array([0.5 + 0j]), Domain())
    assert left[0] < right[0], "the bright half must be the real-negative half"

    flipped = OrbitTrap.from_bitmap(_half_bitmap(left_bright=False))
    assert flipped.sample(np.array([-0.5 + 0j]), Domain())[0] > flipped.sample(np.array([0.5 + 0j]), Domain())[0]


def test_vertical_axis_is_imaginary():
    bitmap = np.zeros((64, 64))
    bitmap[:32, :] = 255  # top half bright -> bright at negative imaginary
    trap = OrbitTrap.from_bitmap(bitmap)
    top = trap.sample(np.array([0 - 0.5j]), Domain())
    bottom = trap.sample(np.array([0 + 0.5j]), Domain())
    assert top[0] < bottom[0]


def test_trap_follows_the_render_domain():
    """A zoomed window must sample its own neighbourhood, not clamp to the border."""
    bitmap = np.zeros((128, 128))
    bitmap[60:68, 60:68] = 255  # a small square in the middle
    trap = OrbitTrap.from_bitmap(bitmap)

    centre = complex(0.0, 0.0)
    wide = Domain(-1, 1, -1, 1)
    tiny = Domain(centre.real - 1e-3, centre.real + 1e-3, centre.imag - 1e-3, centre.imag + 1e-3)

    # In the tiny window every point is inside the bright square: distance 0.
    assert trap.sample(np.array([centre]), tiny)[0] == pytest.approx(0.0, abs=1e-6)
    # In the wide window the corner is far from it.
    assert trap.sample(np.array([-0.9 + 0.9j]), wide)[0] > 0.3


def test_points_outside_the_window_are_marked_far():
    bitmap = np.zeros((32, 32))
    bitmap[8:24, 8:24] = 255
    trap = OrbitTrap.from_bitmap(bitmap)
    small = Domain(-0.1, 0.1, -0.1, 0.1)
    inside = trap.sample(np.array([0j]), small)[0]
    outside = trap.sample(np.array([5 + 5j]), small)[0]
    assert outside > inside


def test_sampling_is_bilinear_and_bounded():
    bitmap = np.zeros((64, 64))
    bitmap[:, :32] = 255
    trap = OrbitTrap.from_bitmap(bitmap)
    grid = np.linspace(-0.9, 0.9, 257) + 0j
    values = trap.sample(grid, Domain())
    assert np.isfinite(values).all()
    assert values.min() >= 0.0 and values.max() <= 1.0 + 1e-9
    # Bilinear tap: neighbouring samples differ smoothly, no jumps between cells.
    assert np.abs(np.diff(values)).max() < 0.1


def test_load_orbit_trap_scales_and_errors(tmp_path):
    from PIL import Image

    from fractalgen.config import OrbitTrapConfig

    path = tmp_path / "trap.png"
    Image.fromarray((_half_bitmap(64) ).astype(np.uint8)).save(path)
    from fractalgen.orbit import load_orbit_trap

    trap = load_orbit_trap(OrbitTrapConfig(image=path, size=32), tmp_path)
    assert trap is not None and trap.field.shape == (32, 32)

    with pytest.raises(FileNotFoundError):
        load_orbit_trap(OrbitTrapConfig(image=tmp_path / "nope.png"), tmp_path)


def _tree_bitmap(size: int = 64) -> np.ndarray:
    """A crude tree: dark canopy + trunk on a light background."""
    bitmap = np.full((size, size), 220.0)
    bitmap[size // 2 :, size // 2 - 2 : size // 2 + 2] = 70.0  # trunk
    centre = size // 2
    radius = size // 3
    yy, xx = np.mgrid[0:size, 0:size]
    canopy = (xx - centre) ** 2 + (yy - centre // 2) ** 2 < radius**2
    bitmap[canopy] = 40.0
    return bitmap


def test_trap_colour_mode_default_and_validation():
    from fractalgen.config import Config, ConfigError, OrbitTrapConfig

    assert OrbitTrapConfig().colour_mode == "closest"
    config = Config.from_dict({"fractal": {"orbit_trap": {"colour_mode": "average"}}})
    assert config.fractal.orbit_trap.colour_mode == "average"
    with pytest.raises(ConfigError, match="colour_mode"):
        Config.from_dict({"fractal": {"orbit_trap": {"colour_mode": "rainbow"}}})


def test_trap_colouring_paints_the_source_shape(tmp_path):
    """return_type 'trap' must reproduce the trap image's colours."""
    import numpy as np
    from PIL import Image

    from fractalgen import Config, render_image

    bitmap = _tree_bitmap(96)
    rgb = np.dstack([bitmap, bitmap, bitmap]).astype(np.uint8)
    path = tmp_path / "tree.png"
    Image.fromarray(rgb).save(path)

    config = Config.from_dict(
        {
            "fractal": {
                "size": 192,
                "iterations": 60,
                "polynomial": {"mode": "roots", "roots": [[1, 0], [-1, 0], [0, 1], [0, -1]]},
                "orbit_trap": {"image": str(path), "size": 96, "colour_mode": "closest"},
            },
            "image": {"return_type": "trap", "shading": {"type": "none"}},
        }
    )
    rendered = render_image(config).rgb
    assert rendered.shape == (192, 192, 3)
    # the source has dark (tree) and light (background) regions, so the render must
    # contain both, in roughly the source's proportions
    dark_share = float((rendered.mean(axis=2) < 120).mean())
    # Orbit trapping repeats the picture across the basins, so exact proportions do
    # not survive - but both tones must be clearly present.
    assert dark_share > 0.05
    assert float(rendered.mean()) < 235.0
    assert float(rendered.std()) > 25.0


def test_trap_colour_requires_a_trap_image(tmp_path):
    from fractalgen import Config, render_image

    config = Config.from_dict({"fractal": {"size": 64}, "image": {"return_type": "trap"}})
    with pytest.raises(ValueError, match="needs an orbit trap"):
        render_image(config)
