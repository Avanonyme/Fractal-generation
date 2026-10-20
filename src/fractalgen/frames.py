"""Frame storage for animations.

Animations used to be written twice: once for "frames in memory" (a list of
numpy arrays) and once for "frames on disk" (a list of file paths), which
duplicated every effect body. :class:`FrameStore` removes that split: effects
see one sequence type that yields ``uint8`` RGB arrays, and the store decides
whether frames live in RAM or in a memory-mapped temp directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Sequence, cast

import numpy as np

from .render import resize, to_uint8

__all__ = ["FrameStore"]


class FrameStore:
    """Ordered sequence of RGB frames, optionally memory-mapped to disk."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._frames: list[np.ndarray] = []
        self._paths: list[Path] = []
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_arrays(cls, frames: Sequence[np.ndarray], cache_dir: Path | None = None) -> "FrameStore":
        store = cls(cache_dir)
        store.extend(frames)
        return store

    @classmethod
    def repeat(cls, frame: np.ndarray, count: int, cache_dir: Path | None = None) -> "FrameStore":
        store = cls(cache_dir)
        store.extend([frame] * count)
        return store

    # ------------------------------------------------------------------ #
    # sequence protocol
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._frames) + len(self._paths)

    def __iter__(self) -> Iterator[np.ndarray]:
        for index in range(len(self)):
            yield cast(np.ndarray, self[index])

    def __getitem__(self, index: int | slice) -> np.ndarray | FrameStore:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            store = FrameStore(self._cache_dir)
            store.extend(self[position] for position in indices)
            return store
        position = index if index >= 0 else len(self) + index
        if position < 0 or position >= len(self):
            raise IndexError(f"frame {index} out of range (0..{len(self) - 1})")
        if position < len(self._frames):
            return self._frames[position]
        # np.load without mmap_mode closes the handle immediately; a memory map
        # would leak one file descriptor per touched frame on long animations.
        return np.load(self._paths[position - len(self._frames)])

    def append(self, frame: np.ndarray) -> None:
        array = to_uint8(frame)
        if array.ndim == 2:
            array = np.dstack([array] * 3)
        if self._cache_dir is None:
            self._frames.append(array)
            return
        path = self._cache_dir / f"frame_{len(self):06d}.npy"
        np.save(path, array)
        self._paths.append(path)

    def extend(self, frames) -> None:
        for frame in frames:
            self.append(frame)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def size(self) -> int:
        return np.asarray(self[0]).shape[0] if len(self) else 0

    def resized(self, size: int, cache_dir: Path | None = None) -> "FrameStore":
        store = FrameStore(cache_dir if cache_dir is not None else self._cache_dir)
        store.extend(resize(frame, size) for frame in self)
        return store

    def stacked(self) -> np.ndarray:
        """All frames as one ``(n, h, w, 3)`` array (requires uniform frames)."""
        return np.stack([np.asarray(frame) for frame in self])

    def clear(self) -> None:
        if self._cache_dir is not None:
            for path in self._paths:
                path.unlink(missing_ok=True)
        self._frames.clear()
        self._paths.clear()

    # ------------------------------------------------------------------ #
    # output
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path, fps: int = 20, codec: str = "gif", loop: int = 0) -> Path:
        """Write the animation as a GIF (Pillow) or MP4 (imageio)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if len(self) == 0:
            raise ValueError("cannot save an empty animation")

        if codec == "mp4":
            return self._save_mp4(path, fps)

        from PIL import Image

        frames = [Image.fromarray(np.asarray(frame)[..., :3]) for frame in self]
        frames[0].save(
            path,
            save_all=True,
            append_images=frames[1:],
            duration=round(1000 / max(1, fps)),
            loop=loop,
            optimize=False,
            disposal=2,
        )
        return path

    def _save_mp4(self, path: Path, fps: int) -> Path:
        from .optional import require

        require("mp4")
        import imageio.v2 as imageio

        with cast(Any, imageio.get_writer(path, fps=fps, macro_block_size=1)) as writer:
            for frame in self:
                writer.append_data(np.asarray(frame)[..., :3])
        return path
