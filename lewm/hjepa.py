"""Two-level H-JEPA: LeWM at level 1 plus a level-2 JEPA over every other level-1 state.

Cube settings from Zhang et al. (2026), "H-JEPA: End-to-End Learning of Hierarchical
World Models for Visual Planning": stride 2, window 1, 2-layer MLP upper encoder
(hidden 384), 1-layer transformer macro-action encoder to 4 dims, upper predictor
like level 1, SIGReg 0.09 at both levels.

One clip = 7 level-1 frames (frameskip 5, so 35 env steps):
  level 1: a random crop of 4 consecutive states, exactly LeWM's loss
  level 2: z2_t = E2(z1_{2t}) for t = 0..3, a2_t = A2(a1_{2t}, a1_{2t+1}) for t = 0..2

Planning is top-down: level 2 plans macro-actions toward the goal's level-2
embedding, its first predicted state becomes a subgoal, and level 1 plans
``stride2`` primitive blocks whose final prediction, lifted by E2, matches it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.jepa import DEFAULT_CONFIG, JEPA, build_lewm
from lewm.layers import MLP, Block
from lewm.predictor import ARPredictor

HJEPA_CONFIG = dict(
    DEFAULT_CONFIG,
    model="hjepa",
    stride2=2,
    encoder2_hidden=384,
    macro_dim=4,
    predictor2_depth=6,
    predictor2_heads=16,
    predictor2_dim_head=64,
    predictor2_mlp_dim=2048,
    sigreg_weight2=0.09,
)


class MacroActionEncoder(nn.Module):
    """Pools the ``stride`` level-1 action embeddings of one level-2 step into a macro-action."""

    def __init__(self, dim: int = 192, stride: int = 2, macro_dim: int = 4, heads: int = 3):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, 1, stride, dim))
        self.block = Block(dim, heads, dim // heads, 4 * dim)
        self.out = nn.Linear(dim, macro_dim)

    def forward(self, act_emb: torch.Tensor) -> torch.Tensor:
        """(B, T2, stride, D) level-1 action embeddings -> (B, T2, macro_dim)."""
        B, T2, s, D = act_emb.shape
        x = (act_emb + self.pos).reshape(B * T2, s, D)
        x = self.block(x, causal_mask=False).mean(dim=1)  # (B*T2, D)
        return normalize_macro(self.out(x)).view(B, T2, -1)


def normalize_macro(macro: torch.Tensor) -> torch.Tensor:
    """Unit-scale macro-actions, so a planner can sample them from N(0, 1)."""
    return F.layer_norm(macro, macro.shape[-1:])


def rollout(predict, emb: torch.Tensor, act_emb: torch.Tensor, history: int) -> torch.Tensor:
    """(N, 1, D) start, (N, L, D_act) actions -> (N, 1 + L, D), using a sliding history window."""
    for k in range(act_emb.shape[1]):
        window_act = act_emb[:, max(0, k + 1 - history) : k + 1]
        next_emb = predict(emb[:, -history:], window_act)[:, -1:]
        emb = torch.cat([emb, next_emb], dim=1)
    return emb


class HJEPA(nn.Module):
    def __init__(
        self,
        level1: JEPA,
        encoder2: MLP,
        macro_encoder: MacroActionEncoder,
        macro_proj: nn.Linear,
        predictor2: ARPredictor,
        pred_proj2: MLP,
        stride2: int = 2,
        sigreg_weight2: float = 0.09,
    ):
        super().__init__()
        self.level1 = level1
        self.encoder2 = encoder2
        self.macro_encoder = macro_encoder
        self.macro_proj = macro_proj
        self.predictor2 = predictor2
        self.pred_proj2 = pred_proj2
        self.stride2 = stride2
        self.sigreg_weight2 = sigreg_weight2
        self.history = level1.history_size
        self.clip_frames = self.history * stride2 + 1  # level-1 frames for history + 1 level-2 states

    def upper_parameters(self) -> list[nn.Parameter]:
        return [p for name, p in self.named_parameters() if not name.startswith("level1.")]

    def encode2(self, z1: torch.Tensor) -> torch.Tensor:
        """(..., D) level-1 embeddings -> (..., D) level-2 embeddings (window 1, so pointwise)."""
        return self.encoder2(z1)

    def predict2(self, z2: torch.Tensor, macro: torch.Tensor) -> torch.Tensor:
        """(B, T, D), (B, T, macro_dim) -> (B, T, D); row t predicts z2 at t + 1."""
        return self.pred_proj2(self.predictor2(z2, self.macro_proj(macro)))

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """batch: {"pixels": (B, 7, 3, H, W), "action": (B, 7, frameskip * A)}."""
        m1, H, s = self.level1, self.history, self.stride2
        z1 = m1.encode(batch["pixels"])           # (B, 7, D)
        a1 = m1.encode_actions(batch["action"])   # (B, 7, D)
        B, T1, D = z1.shape

        # Level 1: LeWM on a random 4-state crop
        r = int(torch.randint(0, T1 - H, ()).item())
        z1c, a1c = z1[:, r : r + H + 1], a1[:, r : r + H + 1]  # (B, 4, D)
        pred1 = m1.predict(z1c[:, :H], a1c[:, :H])
        pred_loss = (pred1 - z1c[:, 1:]).pow(2).mean()
        sigreg_loss = m1.sigreg(z1c.transpose(0, 1))

        # Level 2: every s-th level-1 state, macro-actions pooled from the s actions between them
        z2 = self.encode2(z1[:, ::s])                                   # (B, 4, D)
        macro = self.macro_encoder(a1[:, : H * s].reshape(B, H, s, D))  # (B, 3, macro_dim)
        pred2 = self.predict2(z2[:, :H], macro)
        pred_loss2 = (pred2 - z2[:, 1:]).pow(2).mean()
        sigreg_loss2 = m1.sigreg(z2.transpose(0, 1))

        loss = pred_loss + m1.sigreg_weight * sigreg_loss + pred_loss2 + self.sigreg_weight2 * sigreg_loss2
        return {"loss": loss, "pred_loss": pred_loss, "sigreg_loss": sigreg_loss,
                "pred_loss2": pred_loss2, "sigreg_loss2": sigreg_loss2}

    @torch.no_grad()
    def rollout2(self, z2: torch.Tensor, macros: torch.Tensor) -> torch.Tensor:
        """(N, 1, D) level-2 start, (N, L, macro_dim) raw candidates -> (N, 1 + L, D)."""
        return rollout(self.predict2, z2, normalize_macro(macros), self.history)

    @torch.no_grad()
    def top_cost(self, z2: torch.Tensor, goal2: torch.Tensor, macros: torch.Tensor) -> torch.Tensor:
        """Level-2 cost: final predicted level-2 state vs the goal's.
        z2, goal2 (1, 1, D), macros (S, L, macro_dim) -> (S,)."""
        final = self.rollout2(z2.expand(macros.shape[0], -1, -1), macros)[:, -1]
        return (final - goal2[:, 0]).pow(2).sum(-1)

    @torch.no_grad()
    def low_cost(self, z1: torch.Tensor, subgoal2: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Level-1 cost: level-1 rollout lifted by E2, vs the level-2 subgoal.
        z1 (1, 1, D), subgoal2 (1, D), actions (S, stride2, frameskip * A) normalized -> (S,)."""
        final = self.level1.rollout(z1.expand(actions.shape[0], -1, -1), actions)[:, -1]
        return (self.encode2(final) - subgoal2).pow(2).sum(-1)


def build_hjepa(config: dict | None = None) -> HJEPA:
    """Build an HJEPA from a flat config dict; missing keys take HJEPA_CONFIG values."""
    cfg = {**HJEPA_CONFIG, **(config or {})}
    D = cfg["embed_dim"]
    return HJEPA(
        level1=build_lewm(cfg),
        encoder2=MLP(D, cfg["encoder2_hidden"], D),
        macro_encoder=MacroActionEncoder(D, cfg["stride2"], cfg["macro_dim"]),
        macro_proj=nn.Linear(cfg["macro_dim"], D),
        predictor2=ARPredictor(
            num_frames=cfg["history_size"],
            input_dim=D,
            hidden_dim=D,
            output_dim=D,
            depth=cfg["predictor2_depth"],
            heads=cfg["predictor2_heads"],
            dim_head=cfg["predictor2_dim_head"],
            mlp_dim=cfg["predictor2_mlp_dim"],
            dropout=cfg["predictor_dropout"],
        ),
        pred_proj2=MLP(D, cfg["proj_hidden"], D),
        stride2=cfg["stride2"],
        sigreg_weight2=cfg["sigreg_weight2"],
    )


def build_model(config: dict) -> nn.Module:
    """LeWM or H-JEPA, from a saved checkpoint config (LeWM configs have no "model" key)."""
    return build_hjepa(config) if config.get("model") == "hjepa" else build_lewm(config)
