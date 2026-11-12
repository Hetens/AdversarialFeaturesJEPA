import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularizer (anti-collapse loss).

    Pushes a batch of embeddings toward an isotropic standard Gaussian N(0, I).
    Testing that in D dimensions directly is hard, so it projects the batch onto
    ``num_proj`` random unit directions and runs a 1-D normality test on each
    (Cramer-Wold: a distribution is N(0, I) iff every 1-D projection is N(0, 1)).

    The 1-D test is Epps-Pulley. It compares the empirical characteristic function
    of the projected samples, mean_n exp(i t x_n) = mean cos(t x) + i mean sin(t x),
    with the standard normal's exp(-t^2 / 2), and integrates the squared gap
    over t in [-3, 3] with weight exp(-t^2 / 2). Both functions are symmetric in t,
    so only t in [0, 3] is evaluated and the trapezoid weights are doubled.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots)                 # (K,) evaluation points
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt)          # trapezoid rule, x2 for the negative half
        weights[[0, -1]] = dt                           # trapezoid endpoints get half weight
        gauss_cf = torch.exp(-t.square() / 2)           # (K,) N(0, 1) characteristic function
        self.register_buffer("t", t)
        self.register_buffer("gauss_cf", gauss_cf)
        self.register_buffer("weights", weights * gauss_cf)  # quadrature weight x exp(-t^2/2) window

    def forward(self, emb):
        """emb: (T, B, D) or (B, D). The test runs across the batch dim B,
        separately for each time step, and the result is averaged."""
        B, D = emb.shape[-2:]

        # A fresh set of random unit directions every call: (D, P)
        A = torch.randn(D, self.num_proj, device=emb.device, dtype=emb.dtype)
        A = A / A.norm(dim=0, keepdim=True)

        x_t = (emb @ A).unsqueeze(-1) * self.t          # (..., B, P, K): projected samples times each t
        real = x_t.cos().mean(dim=-3)                   # (..., P, K) mean over the batch
        imag = x_t.sin().mean(dim=-3)                   # (..., P, K)
        err = (real - self.gauss_cf).square() + imag.square()  # |empirical CF - Gaussian CF|^2

        statistic = (err @ self.weights) * B            # (..., P) integrate over t, scale by sample size
        return statistic.mean()                         # average over directions (and time steps)
