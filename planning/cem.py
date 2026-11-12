"""Cross-entropy method (CEM) planner over action sequences."""

from __future__ import annotations

from typing import Callable

import torch


class CEM:
    """Sample candidate action sequences from a Gaussian, keep the lowest-cost
    ``topk``, refit the Gaussian to them, and repeat for ``iters`` rounds.

    Defaults are LeWM's cube settings: 300 samples, top 30, 30 iterations.
    Actions are sampled in the environment's raw space and clipped to its bounds.
    """

    def __init__(
        self,
        num_samples: int = 300,
        topk: int = 30,
        iters: int = 30,
        init_std: float = 1.0,
        low: float = -1.0,
        high: float = 1.0,
    ):
        self.num_samples = num_samples
        self.topk = topk
        self.iters = iters
        self.init_std = init_std
        self.low = low
        self.high = high

    @torch.no_grad()
    def plan(
        self,
        cost_fn: Callable[[torch.Tensor], torch.Tensor],
        horizon: int,
        action_dim: int,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """cost_fn maps (S, horizon, action_dim) raw candidates to (S,) costs.
        Returns the final mean, shape (horizon, action_dim)."""
        mean = torch.zeros(horizon, action_dim, device=device)
        std = torch.full((horizon, action_dim), self.init_std, device=device)
        for _ in range(self.iters):
            noise = torch.randn(self.num_samples, horizon, action_dim, device=device)
            candidates = (mean + std * noise).clamp(self.low, self.high)  # (S, horizon, action_dim)
            costs = cost_fn(candidates)                                    # (S,)
            elite = candidates[costs.topk(self.topk, largest=False).indices]  # (topk, horizon, action_dim)
            mean, std = elite.mean(dim=0), elite.std(dim=0)
        return mean
