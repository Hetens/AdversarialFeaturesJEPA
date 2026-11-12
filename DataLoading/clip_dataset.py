"""Fixed-length LeWM training clips from an OGBench visual dataset.

Each item is ``history_size + num_preds`` frames spaced ``frameskip`` steps
apart, all from one episode. The ``frameskip`` raw actions following each kept
frame are concatenated into that frame's action vector. The corner square is
stamped per item, so the source arrays are never copied or modified.

Shape symbols: N raw frames, E episodes, A raw action dim (5 for cube),
T = history_size + num_preds frames per clip, H x W native image size (64 x 64),
S = img_size when resizing.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

SQUARE_MODES = ("baseline", "episode", "frame")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # (1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)  # (1, 3, 1, 1)


def episode_ends(dataset: dict) -> np.ndarray:
    """Boolean mask of each episode's last frame, shape (N,).

    Compact datasets set ``terminals`` on the last *two* frames of an episode,
    so ``valids == 0`` is the reliable marker there.
    """
    if "valids" in dataset:
        return np.asarray(dataset["valids"]) == 0
    return np.asarray(dataset["terminals"]).astype(bool)

def episode_ids(dataset: dict) -> np.ndarray:
    """Episode index for every frame, shape (N,), values 0..E-1."""
    ends = episode_ends(dataset)
    return np.concatenate([[0], np.cumsum(ends[:-1])])


def square_colors(dataset: dict, mode: str, seed: int = 0) -> np.ndarray | None:
    """Per-frame RGB colors of the corner square, shape (N, 3), or None for baseline.

    ``episode``: one random color per episode (the predictable slow feature).
    ``frame``: an independent random color every frame (the matched control).
    """
    if mode not in SQUARE_MODES:
        raise ValueError(f"square mode must be one of {SQUARE_MODES}, got {mode!r}")
    if mode == "baseline":
        return None
    rng = np.random.default_rng(seed)
    episodes = episode_ids(dataset)
    if mode == "episode":
        # (E, 3) table indexed by each frame's episode -> (N, 3)
        return rng.integers(0, 256, size=(episodes[-1] + 1, 3), dtype=np.uint8)[episodes]
    return rng.integers(0, 256, size=(len(episodes), 3), dtype=np.uint8)


def action_stats(dataset: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-dimension mean and std of the raw actions, each shape (A,)."""
    actions = np.asarray(dataset["actions"], dtype=np.float32)
    return actions.mean(0), actions.std(0) + 1e-6


class ClipDataset(Dataset):
    """Clips of ``{"pixels": (T, 3, H, W) float, "action": (T, frameskip * A) float}``.

    Pass the train split's ``action_stats`` as ``stats`` when building the
    validation dataset so both are normalized identically.
    """

    def __init__(
        self,
        dataset: dict,
        history_size: int = 3,
        num_preds: int = 1,
        frameskip: int = 5,
        square_mode: str = "baseline",
        square_size: int = 5,
        seed: int = 0,
        stats: tuple[np.ndarray, np.ndarray] | None = None,
        img_size: int | None = None,
    ):
        self.observations = dataset["observations"]  # (N, H, W, 3) uint8
        self.actions = np.asarray(dataset["actions"], dtype=np.float32)  # (N, A)
        self.num_frames = history_size + num_preds  # T
        self.frameskip = frameskip
        self.square_size = square_size
        self.img_size = img_size
        self.colors = square_colors(dataset, square_mode, seed)  # (N, 3) uint8 or None

        # Raw steps one clip consumes: frames at start, start+fs, ..., start+(T-1)*fs,
        # plus the fs actions after the last of them, i.e. start .. start+T*fs-1.
        # A start is valid only if that whole range lies inside one episode.
        self.span = self.num_frames * frameskip
        episodes = episode_ids(dataset)  # (N,)
        starts = np.arange(len(episodes) - self.span + 1)  # (N - span + 1,)
        self.starts = starts[episodes[starts] == episodes[starts + self.span - 1]]  # (num_clips,)

        mean, std = stats if stats is not None else action_stats(dataset)  # (A,), (A,)
        # tiled to match the concatenated action vector: (fs * A,)
        self.action_mean = torch.from_numpy(np.tile(mean, frameskip)).float()
        self.action_std = torch.from_numpy(np.tile(std, frameskip)).float()

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = int(self.starts[idx])
        frame_idx = start + self.frameskip * np.arange(self.num_frames)  # (T,)

        # Fancy indexing returns a copy, so stamping never touches the source.
        frames = self.observations[frame_idx]  # (T, H, W, 3) uint8
        if self.colors is not None:
            n = self.square_size
            # (T, 3) -> (T, 1, 1, 3), broadcast over the n x n corner
            frames[:, :n, :n, :] = self.colors[frame_idx][:, None, None, :]

        pixels = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255)  # (T, 3, H, W) in [0, 1]
        if self.img_size is not None and pixels.shape[-1] != self.img_size:  # -> (T, 3, S, S)
            pixels = F.interpolate(
                pixels,
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
        pixels = (pixels - IMAGENET_MEAN) / IMAGENET_STD  # same shape, ImageNet-normalized

        # (span, A) -> (T, fs * A): row k holds the fs raw actions after kept frame k
        action = torch.from_numpy(
            self.actions[start : start + self.span].reshape(self.num_frames, -1)
        )
        action = (action - self.action_mean) / self.action_std  # (T, fs * A)
        return {"pixels": pixels, "action": action}
