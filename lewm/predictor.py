import torch
import torch.nn as nn

from lewm.layers import ConditionalBlock


class ARPredictor(nn.Module):
    """Autoregressive latent predictor.

    Takes the frame embeddings of a clip and the action embedding at each step,
    and outputs, at every position t, a prediction of the embedding at t + 1.
    Causal attention means position t only sees frames and actions up to t, so a
    single forward pass trains all next-step predictions at once.
    """

    def __init__(
        self,
        num_frames=3,
        input_dim=192,
        hidden_dim=192,
        output_dim=None,
        depth=6,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.0,
    ):
        super().__init__()
        # One learned vector per time step: attention has no notion of order otherwise.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.emb_dropout = nn.Dropout(emb_dropout)

        same = input_dim == hidden_dim
        self.input_proj = nn.Identity() if same else nn.Linear(input_dim, hidden_dim)
        self.cond_proj = nn.Identity() if same else nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        output_dim = output_dim or input_dim
        self.output_proj = nn.Identity() if output_dim == hidden_dim else nn.Linear(hidden_dim, output_dim)

    def forward(self, emb, act_emb):
        # emb:     (B, T, input_dim) frame embeddings from the encoder
        # act_emb: (B, T, input_dim) action embeddings from the action encoder
        T = emb.shape[1]
        if T > self.pos_embedding.shape[1]:
            raise ValueError(f"got {T} frames, predictor was built for at most {self.pos_embedding.shape[1]}")

        x = emb + self.pos_embedding[:, :T]  # (B, T, input_dim)
        x = self.emb_dropout(x)
        x = self.input_proj(x)               # (B, T, hidden_dim)
        c = self.cond_proj(act_emb)          # (B, T, hidden_dim)

        for block in self.blocks:
            x = block(x, c)                  # causal attention, AdaLN conditioned on the actions

        x = self.norm(x)
        return self.output_proj(x)           # (B, T, output_dim): row t predicts emb at t + 1
