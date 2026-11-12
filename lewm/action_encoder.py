import torch
import torch.nn as nn

class ActionEncoder(nn.Module):
    def __init__(self, input_dim=25, smoothed_dim=10, embed_dim=192, mlp_scale=4):
        super().__init__()
        # input_dim = frameskip * raw action dim (5 * 5 for cube)
        self.smooth = nn.Linear(input_dim, smoothed_dim)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * embed_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * embed_dim, embed_dim),
        )

    def forward(self, actions):
        # actions: (B, T, input_dim) float
        return self.embed(self.smooth(actions.float()))  # (B, T, embed_dim)

    #embed dim = 192 -> goes into conditionalBlock AdaLN because block is conditioned on the action

# enc = ActionEncoder()
# a = torch.randn(2, 4, 25)   # (B, T, 25)

# a2 = a.clone()              # copy, so `a` stays as it was
# a2[:, 2] += 1.0             # every sample, time step 2, all 25 numbers

# out, out2 = enc(a), enc(a2)                # (2, 4, 192) each
# changed = (out != out2).any(dim=-1)        # (2, 4): did any of the 192 numbers change?
# print(out)
