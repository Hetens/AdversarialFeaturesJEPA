"""Measure what a trained LeWM encoder keeps in its embedding, on validation frames.

    python scripts/analyze.py --checkpoint runs/episode_s0/epoch_009.pt

Writes analysis_<checkpoint stem>.json next to the checkpoint. Three measurements:

  probes       ridge regression from embeddings to cube xyz, arm joint angles and
               the corner square's RGB (train/test split by episode). Frames get an
               independent random square color each, which matches the per-frame
               marginal of both the episode and frame arms.
  sensitivity  each frame re-encoded under K random square colors; the share of
               total embedding variance explained by the square color alone.
  rollout      predictor rolled out L blocks from one frame in the model's own
               square mode; error vs encoded ground truth, next to a "copy the
               first embedding" baseline.

For H-JEPA checkpoints, ``--level 2`` runs all three on level-2 embeddings E2(z1);
the rollout then steps 2 level-1 blocks (10 env steps) per prediction, driven by
macro-actions, and writes analysis_l2_<stem>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from DataLoading.clip_dataset import episode_ids, preprocess_frames, square_colors, stamp_square  # noqa: E402
from lewm.hjepa import build_model  # noqa: E402
from scripts.train import load_split, pick_device  # noqa: E402

ARM_JOINTS = slice(0, 4)   # UR5e shoulder/elbow/wrist-1 angles; joint 4 is ~constant, joint 5 wraps
CUBE_XYZ = slice(-7, -4)   # cube free joint: xyz then quaternion, last in qpos


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--level", type=int, choices=[1, 2], default=1, help="2 = H-JEPA level-2 embeddings")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "Data" / "ogbench"))
    p.add_argument("--probe-frames", type=int, default=6000)
    p.add_argument("--sens-frames", type=int, default=500)
    p.add_argument("--sens-colors", type=int, default=8)
    p.add_argument("--rollout-clips", type=int, default=2000)
    p.add_argument("--rollout-steps", type=int, default=5)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    return p.parse_args()


@torch.no_grad()
def encode(encode_fn, frames: np.ndarray, img_size: int, device, batch_size: int) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, D) on CPU; encode_fn maps (B, 1, 3, H, W) -> (B, 1, D)."""
    out = []
    for i in range(0, len(frames), batch_size):
        pixels = preprocess_frames(frames[i : i + batch_size], img_size)  # (B, 3, H, W)
        out.append(encode_fn(pixels[:, None].to(device))[:, 0].float().cpu())
    return torch.cat(out)


def ridge_r2(x: torch.Tensor, y: torch.Tensor, train: np.ndarray, lam: float) -> list[float]:
    """Held-out R^2 per target column of a ridge fit on standardized features."""
    x, y = x.double(), y.double()
    mu, sd = x[train].mean(0), x[train].std(0) + 1e-6
    x = torch.cat([(x - mu) / sd, torch.ones(len(x), 1, dtype=x.dtype)], dim=1)
    xt, yt = x[train], y[train]
    w = torch.linalg.solve(xt.T @ xt + lam * torch.eye(x.shape[1], dtype=x.dtype), xt.T @ yt)
    test = ~train
    resid = (y[test] - x[test] @ w).pow(2).sum(0)
    total = (y[test] - y[test].mean(0)).pow(2).sum(0)
    return (1 - resid / total).tolist()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config, cfg = ckpt["config"], ckpt["config"]["model"]
    train_cfg = run_config["train"]
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    img_size, size, fs = cfg["img_size"], train_cfg["square_size"], cfg["frameskip"]
    hjepa = cfg.get("model") == "hjepa"
    if args.level == 2 and not hjepa:
        raise SystemExit("--level 2 needs an H-JEPA checkpoint")
    lewm = model.level1 if hjepa else model
    if args.level == 2:
        encode_fn = lambda pixels: model.encode2(lewm.encode(pixels))  # noqa: E731
        blocks_per_step = model.stride2
    else:
        encode_fn, blocks_per_step = lewm.encode, 1

    stem = f"{train_cfg['dataset']}-val"
    data = load_split(train_cfg["dataset"], Path(args.data_dir), "val")
    qpos = np.load(Path(args.data_dir) / f"{stem}.npz")["qpos"]  # (N, 21)
    obs = data["observations"]                                      # (N, H, W, 3) uint8, mmap
    episodes = episode_ids(data)                                    # (N,)
    enc = lambda frames: encode(encode_fn, frames, img_size, device, args.batch_size)  # noqa: E731
    random_colors = lambda n: rng.integers(0, 256, size=(n, 3), dtype=np.uint8)    # noqa: E731

    # Probes
    idx = np.sort(rng.choice(len(obs), args.probe_frames, replace=False))
    frames = obs[idx]                                               # (P, H, W, 3)
    colors = random_colors(len(idx))                                # (P, 3)
    emb_clean = enc(frames)
    emb_sq = enc(stamp_square(frames.copy(), colors, size))
    train = episodes[idx] % 5 != 0                                  # 80/20 split by episode
    cube = torch.from_numpy(qpos[idx, CUBE_XYZ])
    arm = torch.from_numpy(qpos[idx, ARM_JOINTS])
    rgb = torch.from_numpy(colors.astype(np.float32))
    probes = {
        "cube_xyz_r2_clean": ridge_r2(emb_clean, cube, train, args.ridge),
        "cube_xyz_r2_square": ridge_r2(emb_sq, cube, train, args.ridge),
        "arm_joints_r2_clean": ridge_r2(emb_clean, arm, train, args.ridge),
        "arm_joints_r2_square": ridge_r2(emb_sq, arm, train, args.ridge),
        "square_rgb_r2": ridge_r2(emb_sq, rgb, train, args.ridge),
    }
    probes = {k: {"per_dim": v, "mean": float(np.mean(v))} for k, v in probes.items()}

    # Sensitivity: (M, K, D) embeddings of M frames under K colors each
    idx = rng.choice(len(obs), args.sens_frames, replace=False)
    M, K = len(idx), args.sens_colors
    frames = np.repeat(obs[idx], K, axis=0)                         # (M*K, H, W, 3)
    emb = enc(stamp_square(frames, random_colors(M * K), size)).view(M, K, -1)
    clean = enc(obs[idx])                                           # (M, D)
    total_var = emb.reshape(M * K, -1).var(0).sum()
    color_var = emb.var(1).sum(-1).mean()
    sensitivity = {
        "color_variance_share": float(color_var / total_var),
        "mean_dist_same_frame_diff_color": float(torch.cdist(emb, emb).sum((1, 2)).mean() / (K * (K - 1))),
        "mean_dist_diff_frame_clean": float(torch.pdist(clean).mean()),
        "mean_dist_clean_vs_square": float((emb - clean[:, None]).norm(dim=-1).mean()),
    }

    # Rollout in the model's own square mode
    L, step = args.rollout_steps, fs * blocks_per_step              # env steps per prediction
    span = L * step + fs
    starts = np.arange(len(obs) - span + 1)
    starts = starts[episodes[starts] == episodes[starts + span - 1]]
    starts = rng.choice(starts, args.rollout_clips, replace=False)
    frame_idx = starts[:, None] + step * np.arange(L + 1)           # (C, L+1)
    frames = obs[frame_idx.reshape(-1)]                             # (C*(L+1), H, W, 3)
    own = square_colors(data, train_cfg["square_mode"], train_cfg["seed"] + 1)
    if own is not None:
        stamp_square(frames, own[frame_idx.reshape(-1)], size)
    true = enc(frames).view(len(starts), L + 1, -1)                 # (C, L+1, D)
    mean = np.tile(run_config["action_mean"], fs)
    std = np.tile(run_config["action_std"], fs)
    acts = data["actions"][starts[:, None] + np.arange(L * step)].reshape(len(starts), L * blocks_per_step, -1)
    acts = torch.from_numpy(((acts - mean) / std).astype(np.float32))  # (C, L*blocks_per_step, fs*A)

    def predict_rollout(e0: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        if args.level == 1:
            return lewm.rollout(e0, a)
        a1 = lewm.encode_actions(a)                                  # (B, L*s, D)
        macros = model.macro_encoder(a1.view(len(a), L, blocks_per_step, -1))
        return model.rollout2(e0, macros)

    preds = []
    with torch.no_grad():
        for i in range(0, len(starts), args.batch_size):
            e0 = true[i : i + args.batch_size, :1].to(device)
            preds.append(predict_rollout(e0, acts[i : i + args.batch_size].to(device))[:, 1:].float().cpu())
    pred = torch.cat(preds)                                         # (C, L, D)
    target = true[:, 1:]
    emb_var = true.reshape(-1, true.shape[-1]).var(0).sum()
    rollout = {
        "mse_per_step": (pred - target).pow(2).sum(-1).mean(0).tolist(),
        "copy_mse_per_step": (true[:, :1] - target).pow(2).sum(-1).mean(0).tolist(),
        "embedding_total_variance": float(emb_var),
    }
    rollout["relative_mse_per_step"] = [m / float(emb_var) for m in rollout["mse_per_step"]]
    rollout["model_over_copy_per_step"] = [m / c for m, c in zip(rollout["mse_per_step"], rollout["copy_mse_per_step"])]

    summary = {"checkpoint": str(args.checkpoint), "square_mode": train_cfg["square_mode"], "level": args.level,
               "probes": probes, "sensitivity": sensitivity, "rollout": rollout, "args": vars(args)}
    level_tag = "l2_" if args.level == 2 else ""
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(
        f"analysis_{level_tag}{Path(args.checkpoint).stem}.json"
    )
    out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in ("square_mode", "sensitivity")}, indent=2))
    print({k: round(v["mean"], 3) for k, v in probes.items()})
    print("rollout model/copy", [round(r, 3) for r in rollout["model_over_copy_per_step"]])
    print(f"-> {out}")


if __name__ == "__main__":
    main()
