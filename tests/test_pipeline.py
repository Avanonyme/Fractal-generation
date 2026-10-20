from pathlib import Path

import numpy as np
import pytest

from fractalgen import Config
from fractalgen.frames import FrameStore
from fractalgen.pipeline import build_frame, build_video_frames, render_image, render_video


def _config(tmp_path: Path, **overrides) -> Config:
    data = {
        "fractal": {
            "size": 48,
            "iterations": 20,
            "polynomial": {"mode": "roots", "roots": [[1, 0], [-1, 0], [0, 1], [0, -1]]},
        },
        "colour": {"mode": "matplotlib", "name": "viridis"},
        "output": str(tmp_path / "fractal.png"),
    }
    data.update(overrides)
    return Config.from_dict(data, base_dir=tmp_path)


def test_render_image_writes_png(tmp_path: Path):
    config = _config(tmp_path)
    rendered = render_image(config)
    assert rendered.rgb.shape == (48, 48, 3)
    assert rendered.path is not None and rendered.path.exists()


def test_render_image_without_output_path(tmp_path: Path):
    config = _config(tmp_path)
    config = Config.from_dict({**config.to_dict(), "output": None}, base_dir=tmp_path)
    assert render_image(config).path is None


@pytest.mark.parametrize("shading", ["matplotlib", "blinn-phong", "none"])
def test_shading_variants_produce_variation(tmp_path: Path, shading: str):
    config = _config(tmp_path, image={"return_type": "iteration", "shading": {"type": shading}})
    rendered = render_image(config)
    assert rendered.rgb.std() > 1.0


def test_all_return_types(tmp_path: Path):
    for return_type in ("iteration", "boundary", "roots"):
        config = _config(tmp_path, image={"return_type": return_type, "shading": {"type": "none"}})
        assert render_image(config).rgb.shape == (48, 48, 3)


def test_distance_return_type_needs_trap(tmp_path: Path):
    config = _config(tmp_path, image={"return_type": "distance"})
    with pytest.raises(ValueError, match="orbit trap"):
        render_image(config)


def test_frame_store_memory_and_disk(tmp_path: Path):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    memory = FrameStore()
    memory.append(frame)
    memory.append(frame + 1)
    assert len(memory) == 2
    assert np.asarray(memory[0]).shape == (8, 8, 3)

    disk = FrameStore(tmp_path / "frames")
    disk.extend([frame, frame])
    assert len(disk) == 2
    assert np.asarray(disk[1]).shape == (8, 8, 3)
    assert (tmp_path / "frames").glob("*.npy")


def test_video_writes_gif(tmp_path: Path):
    config = _config(
        tmp_path,
        form="video",
        output=str(tmp_path / "out.gif"),
        video={"fps": 5, "frames": 4, "animations": [{"name": "alpha"}]},
    )
    path = render_video(config)
    assert path.exists() and path.stat().st_size > 0


def test_effect_chain(tmp_path: Path):
    config = _config(
        tmp_path,
        form="video",
        output=str(tmp_path / "chain.gif"),
        video={
            "fps": 5,
            "frames": 4,
            "animations": [{"name": "flicker", "amplitude": 0.3}, {"name": "pulse"}, {"name": "grain", "amount": 0.12, "grain_size": 1.5}],
        },
    )
    store = build_video_frames(config)
    assert len(store) == 4
    assert all(np.asarray(frame).shape[0] == 48 for frame in store)


def test_build_frame_is_deterministic(tmp_path: Path):
    config = _config(tmp_path)
    rendered = render_image(config)
    again = build_frame(config, rendered.fractal, __import__("fractalgen").build_colormap(config.colour, tmp_path))
    assert np.array_equal(rendered.rgb, again)


def test_unknown_effect_parameter_is_rejected(tmp_path: Path):
    from fractalgen.animations import resolve_effect_params
    from fractalgen.config import ConfigError

    with pytest.raises(ConfigError, match="unknown parameter"):
        resolve_effect_params("flicker", {"amplitood": 1.0})


def test_flicker_does_not_change_exposure(tmp_path: Path):
    from fractalgen.animations import AnimationContext, flicker

    config = _config(tmp_path, video={"fps": 10, "frames": 8, "animations": []})
    rendered = render_image(config)
    from fractalgen import build_colormap

    store = FrameStore.repeat(rendered.rgb, 8)
    context = AnimationContext(
        config=config, image=rendered.fractal, cmap=build_colormap(config.colour, tmp_path), frames=8, fps=10
    )
    frames = [np.asarray(frame, dtype=float) for frame in flicker(store, {}, context)]
    means = [frame.mean() for frame in frames]
    # A global brightness pump was the old failure mode; it must stay flat.
    assert max(means) - min(means) < 2.0
    assert any(np.abs(frame - frames[0]).max() > 5 for frame in frames[1:])


def test_grain_is_seeded_and_mean_preserving(tmp_path: Path):
    from fractalgen.animations import AnimationContext, grain

    config = _config(tmp_path, video={"fps": 10, "frames": 4, "animations": []})
    rendered = render_image(config)
    from fractalgen import build_colormap

    cmap = build_colormap(config.colour, tmp_path)
    store = FrameStore.repeat(rendered.rgb, 4)
    context = AnimationContext(config=config, image=rendered.fractal, cmap=cmap, frames=4, fps=10)
    first = [np.asarray(frame, dtype=float) for frame in grain(store, {"seed": 5}, context)]
    second = [np.asarray(frame, dtype=float) for frame in grain(store, {"seed": 5}, context)]
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert abs(first[0].mean() - rendered.rgb.mean()) < 1.0


def test_flicker_puts_stars_on_the_fractal(tmp_path: Path):
    """Sparkles must sit on the fractal body, not speckle the whole frame."""
    from scipy import ndimage

    from fractalgen import build_colormap
    from fractalgen.animations import AnimationContext, flicker
    from fractalgen.frames import FrameStore

    config = _config(tmp_path, video={"fps": 10, "frames": 8, "animations": []})
    rendered = render_image(config)
    context = AnimationContext(
        config=config, image=rendered.fractal, cmap=build_colormap(config.colour, tmp_path), frames=8, fps=10
    )
    frames = [
        np.asarray(frame, dtype=float)
        for frame in flicker(FrameStore.repeat(rendered.rgb, 8), {"seed": 3}, context)
    ]
    delta = np.abs(frames[4] - rendered.rgb.astype(float)).max(axis=2)
    lit = delta > 6
    assert lit.any(), "the effect changed nothing"
    body = np.asarray(ndimage.binary_dilation(rendered.fractal.boundary, iterations=3), dtype=bool)
    assert bool((lit & body).sum()) / max(1, int(lit.sum())) > 0.85
    # points, not blobs: the lit area stays sparse (small frames amplify the share)
    assert lit.mean() < 0.12


def test_grain_and_alpha_do_not_darken(tmp_path: Path):
    from fractalgen import build_colormap
    from fractalgen.animations import AnimationContext, alpha, grain
    from fractalgen.frames import FrameStore

    config = _config(tmp_path, video={"fps": 10, "frames": 6, "animations": []})
    rendered = render_image(config)
    cmap = build_colormap(config.colour, tmp_path)
    context = AnimationContext(config=config, image=rendered.fractal, cmap=cmap, frames=6, fps=10)
    store = FrameStore.repeat(rendered.rgb, 6)
    base = rendered.rgb.astype(float).mean()

    grained = [np.asarray(frame, dtype=float) for frame in grain(store, {"seed": 3}, context)]
    assert abs(grained[-1].mean() - base) < 1.0, "grain must not shift the mean"

    faded = [np.asarray(frame, dtype=float) for frame in alpha(store, {}, context)]
    assert faded[-1].mean() > 0.9 * base, "alpha must not end on a black frame"
