import torch
import torch.nn.functional as F
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def compute_attention(self, q, k, v, causal_mask=True):
        # q, k, v: (B, h, T, dh)
        attn = (q @ k.transpose(-2, -1)) / (self.dim_head ** 0.5)  # (B, h, T, T)
        if causal_mask:
            T = attn.shape[-1]
            mask = torch.ones(T, T, dtype=torch.bool, device=attn.device).tril()
            attn = attn.masked_fill(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        return attn @ v  # (B, h, T, dh)

    def split_heads(self, t):
        # (B, T, h * dh) -> (B, h, T, dh)
        B, T, _ = t.shape
        return t.view(B, T, self.heads, self.dim_head).transpose(1, 2)

    def forward(self, x, causal_mask=True):
        # x: (B, T, D)
        q = self.split_heads(self.to_q(x))
        k = self.split_heads(self.to_k(x))
        v = self.split_heads(self.to_v(x))
        out = self.compute_attention(q, k, v, causal_mask)  # (B, h, T, dh)
        out = out.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)  # (B, T, h * dh)
        return self.dropout(self.to_out(out))  # (B, T, D)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.net(x))

class Block(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, causal_mask=True):
        x = x + self.attention(self.norm1(x), causal_mask)
        x = x + self.feed_forward(self.norm2(x))
        return x

class ConditionalBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super(ConditionalBlock, self).__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attention = Attention(dim, heads, dim_head, dropout)
        self.feed_forward = FeedForward(dim, mlp_dim, dropout)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6*dim),
        )
        # Zero init makes every gate 0, so the block starts as the identity.
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        #x (B, T, D)
        #c (B, T, D) Action embedding
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)
        x = x + gate1 * self.attention(self.norm1(x) * (1 + scale1) + shift1)
        x = x + gate2 * self.feed_forward(self.norm2(x) * (1 + scale2) + shift2)
        return x

class MLP(nn.Module):
    """Projector head (used as ``projector`` after the encoder and ``pred_proj``
    after the predictor). BatchNorm1d needs 2-D input, so leading dims are
    flattened to one batch dim and restored afterwards."""

    def __init__(self, input_dim, hidden_dim, output_dim=None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        # x: (..., input_dim), e.g. (B, D) or (B, T, D)
        lead = x.shape[:-1]
        out = self.net(x.reshape(-1, x.shape[-1]))  # (prod(lead), output_dim)
        return out.reshape(*lead, -1)