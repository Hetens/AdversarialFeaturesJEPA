"""LeWM: ties the encoder, action encoder, predictor and SIGReg into one model.

Training:  pixels -> encode -> emb (B, T, D); actions -> act_emb (B, T, D)
           pred = predict(emb[:, :H], act_emb[:, :H]) vs target emb[:, num_preds:]
           loss = MSE(pred, target) + sigreg_weight * SIGReg(emb)
Planning:  encode the current frame and the goal frame, roll the predictor
           forward under candidate action sequences, and score each candidate by
           the distance of its final predicted embedding to the goal embedding.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from lewm.action_encoder import ActionEncoder
from lewm.encoder import Encoder
from lewm.layers import MLP
from lewm.predictor import ARPredictor
from lewm.sigreg import SIGReg

# Upstream LeWM sizes, with a 64 px / patch 8 ViT-Tiny for OGBench frames.
DEFAULT_CONFIG = dict(
    img_size=64,
    patch_size=8,
    embed_dim=192,
    encoder_depth=12,
    encoder_heads=3,
    encoder_mlp_dim=768,
    proj_hidden=2048,
    action_dim=5,
    frameskip=5,
    action_smoothed_dim=10,
    predictor_depth=6,
    predictor_heads=16,
    predictor_dim_head=64,
    predictor_mlp_dim=2048,
    predictor_dropout=0.1,
    history_size=3,
    num_preds=1,
    sigreg_weight=0.09,
    sigreg_knots=17,
    sigreg_num_proj=1024,
)


class JEPA(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        action_encoder: ActionEncoder,
        predictor: ARPredictor,
        pred_proj: MLP,
        sigreg: SIGReg,
        history_size: int = 3,
        num_preds: int = 1,
        sigreg_weight: float = 0.09,
    ):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.predictor = predictor
        self.pred_proj = pred_proj
        self.sigreg = sigreg
        self.history_size = history_size
        self.num_preds = num_preds
        self.sigreg_weight = sigreg_weight

    def encode(self, pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) -> (B, T, D) projected frame embeddings."""
        return self.encoder(pixels)

    def encode_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """(B, T, frameskip * A) normalized actions -> (B, T, D)."""
        return self.action_encoder(actions)

    def predict(self, emb: torch.Tensor, act_emb: torch.Tensor) -> torch.Tensor:
        """(B, T, D), (B, T, D) -> (B, T, D); row t predicts the embedding at t + 1."""
        return self.pred_proj(self.predictor(emb, act_emb))

    def loss(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """batch: {"pixels": (B, T, 3, H, W), "action": (B, T, frameskip * A)}
        with T = history_size + num_preds."""
        emb = self.encode(batch["pixels"])            # (B, T, D)
        act_emb = self.encode_actions(batch["action"])  # (B, T, D)

        H = self.history_size
        pred = self.predict(emb[:, :H], act_emb[:, :H])  # (B, H, D)
        # Not detached: gradients flow through the targets too (no EMA / stop-grad).
        target = emb[:, self.num_preds:]                 # (B, H, D)

        pred_loss = (pred - target).pow(2).mean()
        sigreg_loss = self.sigreg(emb.transpose(0, 1))   # SIGReg wants (T, B, D)
        loss = pred_loss + self.sigreg_weight * sigreg_loss
        return {"loss": loss, "pred_loss": pred_loss, "sigreg_loss": sigreg_loss}

    @torch.no_grad()
    def rollout(self, emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Autoregressively predict future embeddings.

        emb:     (N, 1, D) embedding of the current frame
        actions: (N, L, frameskip * A) normalized action blocks to apply in order
        returns  (N, 1 + L, D): the current embedding followed by L predictions.
        Action k is paired with embedding k; only the last history_size steps are
        fed to the predictor, matching what it saw in training.
        """
        H = self.history_size
        act_emb = self.encode_actions(actions)  # (N, L, D)
        for k in range(actions.shape[1]):
            window_emb = emb[:, -H:]                              # (N, <=H, D)
            window_act = act_emb[:, max(0, k + 1 - H) : k + 1]    # (N, <=H, D), aligned with window_emb
            next_emb = self.predict(window_emb, window_act)[:, -1:]  # (N, 1, D)
            emb = torch.cat([emb, next_emb], dim=1)
        return emb

    @torch.no_grad()
    def plan_cost(self, emb: torch.Tensor, goal_emb: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Cost of S candidate action sequences from one start state.

        emb: (1, 1, D) current embedding, goal_emb: (1, 1, D) goal embedding,
        actions: (S, L, frameskip * A) normalized candidates -> (S,) squared
        distance between each final predicted embedding and the goal.
        """
        S = actions.shape[0]
        final = self.rollout(emb.expand(S, -1, -1), actions)[:, -1]  # (S, D)
        return (final - goal_emb[:, 0]).pow(2).sum(dim=-1)


def build_lewm(config: dict | None = None) -> JEPA:
    """Build a JEPA from a flat config dict; missing keys take DEFAULT_CONFIG values."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    D = cfg["embed_dim"]
    encoder = Encoder(
        img_size=cfg["img_size"],
        patch_size=cfg["patch_size"],
        dim=D,
        proj_hidden=cfg["proj_hidden"],
        depth=cfg["encoder_depth"],
        heads=cfg["encoder_heads"],
        dim_head=D // cfg["encoder_heads"],
        mlp_dim=cfg["encoder_mlp_dim"],
    )
    action_encoder = ActionEncoder(
        input_dim=cfg["frameskip"] * cfg["action_dim"],
        smoothed_dim=cfg["action_smoothed_dim"],
        embed_dim=D,
    )
    predictor = ARPredictor(
        num_frames=cfg["history_size"],
        input_dim=D,
        hidden_dim=D,
        output_dim=D,
        depth=cfg["predictor_depth"],
        heads=cfg["predictor_heads"],
        dim_head=cfg["predictor_dim_head"],
        mlp_dim=cfg["predictor_mlp_dim"],
        dropout=cfg["predictor_dropout"],
    )
    pred_proj = MLP(D, cfg["proj_hidden"], D)
    sigreg = SIGReg(knots=cfg["sigreg_knots"], num_proj=cfg["sigreg_num_proj"])
    return JEPA(
        encoder,
        action_encoder,
        predictor,
        pred_proj,
        sigreg,
        history_size=cfg["history_size"],
        num_preds=cfg["num_preds"],
        sigreg_weight=cfg["sigreg_weight"],
    )
