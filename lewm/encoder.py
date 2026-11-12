import torch
import torch.nn as nn
from lewm.layers import Block, MLP

class PatchEmbed(nn.Module):
    """(B, 3, H, W) -> (B, N, D), N= (H/ patch) * (W/ patch)"""
    def __init__(self, img_size = 64, patch_size = 8, in_chans = 3, dim = 192):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)#(B, D, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)#(B, N, D)
        return x

class ViT(nn.Module):
    def __init__(self, img_size =64, 
                    patch_size = 8, 
                    dim= 192, 
                    depth = 12, 
                    heads=3, 
                    dim_head=64, 
                    mlp_dim=768, 
                    dropout=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1,self.patch_embed.num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.blocks = nn.ModuleList(
            [Block(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
            )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.patch_embed(x) #(B, N, D)
        cls = self.cls_token.expand(x.size(0), -1, -1) #(B, 1, D)
        x = torch.cat([cls, x], dim=1) #(B, N+1, D)
        x = x + self.pos_embed #(B, N+1, D)
        for block in self.blocks:
            x = block(x, causal_mask=False) #(B, N+1, D)
        x = self.norm(x) #(B, N+1, D)
        return x[:,0] #(B, D)


class Encoder(nn.Module):
    def __init__(self, img_size=64, patch_size=8, dim=192, proj_hidden=2048, **vit_kwargs):
        super().__init__()
        self.vit = ViT(img_size, patch_size, dim, **vit_kwargs)
        self.projector = MLP(dim, proj_hidden, dim)

    def forward(self, pixels):
        # pixels: (B, T, 3, H, W) from ClipDataset
        B, T = pixels.shape[:2]
        cls = self.vit(pixels.flatten(0, 1))  # (B*T, D): frames encoded independently
        emb = self.projector(cls)             # (B*T, D)
        return emb.view(B, T, -1)             # (B, T, D)