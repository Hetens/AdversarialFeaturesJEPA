from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import ogbench

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "Data"
DATASET_DIR = DATA_DIR / "ogbench"
PREVIEW_DIR = DATA_DIR / "preview"

# Single-cube play data. Goal-conditioned train/val splits download as
# `{name}.npz` and `{name}-val.npz` (default cache: Data/ogbench).
DATASET_NAME = "cube-single-play-v0"


# Full cube-single play is ~1M train transitions. 
PREVIEW_TRAIN_STEPS = 256
PREVIEW_VAL_STEPS = 64


def load_og_bench(
    dataset_name: str = DATASET_NAME,
    dataset_dir: str | Path = DATASET_DIR,
    compact_dataset: bool = True,
    render_mode: str = "rgb_array",
):
    """Return (env, train_dataset, val_dataset) for an OGBench task.

    Datasets are dicts with at least ``observations``, ``actions``, and
    ``terminals``. With ``compact_dataset=True`` (default) they also have
    ``valids`` and omit ``next_observations``.
    """
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name,
        dataset_dir=str(dataset_dir),
        compact_dataset=compact_dataset,
        render_mode=render_mode,
    )
    return env, train_dataset, val_dataset


def _summarize(name: str, dataset: dict) -> str:
    parts = [name]
    for key, value in dataset.items():
        arr = np.asarray(value)
        parts.append(f"  {key}: shape={arr.shape} dtype={arr.dtype}")
    return "\n".join(parts)

#just to inspect the environment
def _write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {rgb.shape}")
    height, width, _ = rgb.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in np.ascontiguousarray(rgb, dtype=np.uint8))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


def _slice_dataset(dataset: dict, steps: int) -> dict:
    return {key: np.asarray(value)[:steps] for key, value in dataset.items()}


def save_preview(
    env,
    train_dataset: dict,
    val_dataset: dict,
    preview_dir: str | Path = PREVIEW_DIR,
    train_steps: int = PREVIEW_TRAIN_STEPS,
    val_steps: int = PREVIEW_VAL_STEPS,
    n_frames: int = 6,
) -> Path:
    """Write a small train/val slice and a few rendered Cube frames."""
    preview_dir = Path(preview_dir)
    frames_dir = preview_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    train_preview = _slice_dataset(train_dataset, train_steps)
    val_preview = _slice_dataset(val_dataset, val_steps)
    np.savez_compressed(preview_dir / "cube_single_train.npz", **train_preview)
    np.savez_compressed(preview_dir / "cube_single_val.npz", **val_preview)

    _, info = env.reset(options={"task_id": 1, "render_goal": True})
    goal = info.get("goal_rendered")
    if goal is not None:
        _write_png(frames_dir / "goal_task1.png", goal)
    _write_png(frames_dir / "frame_00.png", env.render())

    for i in range(1, n_frames):
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        _write_png(frames_dir / f"frame_{i:02d}.png", env.render())
        if terminated or truncated:
            env.reset(options={"task_id": 1})

    return preview_dir


if __name__ == "__main__":
    env, train_dataset, val_dataset = load_og_bench()
    print(f"dataset: {DATASET_NAME}")
    print(_summarize("train", train_dataset))
    print(_summarize("val", val_dataset))
    preview_path = save_preview(env, train_dataset, val_dataset)
    print(f"preview saved under {preview_path}")

