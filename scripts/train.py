"""Train LeWM on an OGBench visual dataset.

    python scripts/train.py --square-mode baseline --run-name baseline_s0
    python scripts/train.py --square-mode episode  --run-name episode_s0
    python scripts/train.py --square-mode frame    --run-name frame_s0

Writes runs/<run-name>/{config.json, metrics.jsonl, last.pt, epoch_XXX.pt}.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import ogbench
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from DataLoading.clip_dataset import ClipDataset, action_stats  # noqa: E402
from lewm.jepa import DEFAULT_CONFIG, build_lewm  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="visual-cube-single-play-v0")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "Data" / "ogbench"))
    p.add_argument("--run-name", default=None, help="defaults to <square-mode>_s<seed>")
    p.add_argument("--runs-dir", default=str(REPO_ROOT / "runs"))
    p.add_argument("--square-mode", choices=["baseline", "episode", "frame"], default="baseline")
    p.add_argument("--square-size", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--warmup-frac", type=float, default=0.05, help="fraction of total steps spent warming up")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0, help="keep 0 on macOS: each worker copies the dataset")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--max-steps-per-epoch", type=int, default=None, help="cap for quick smoke tests")
    p.add_argument("--max-val-batches", type=int, default=None)
    # Any DEFAULT_CONFIG key can be overridden, e.g. --img-size 224 --patch-size 14
    for key, value in DEFAULT_CONFIG.items():
        p.add_argument(f"--{key.replace('_', '-')}", type=type(value), default=value)
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split(dataset: str, data_dir: Path, split: str) -> dict:
    path = data_dir / (f"{dataset}.npz" if split == "train" else f"{dataset}-val.npz")
    if not path.exists():
        ogbench.download_datasets([dataset], dataset_dir=str(data_dir))
    return ogbench.load_dataset(str(path), ob_dtype=np.uint8, action_dtype=np.float32, compact_dataset=True)


def warmup_cosine(total_steps: int, warmup_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    return factor


@torch.no_grad()
def validate(model, loader, device, autocast, max_batches=None) -> dict[str, float]:
    model.eval()
    totals, n = {}, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with autocast():
            out = model.loss(batch)
        for k, v in out.items():
            totals[k] = totals.get(k, 0.0) + v.item()
        n += 1
    model.train()
    return {f"val/{k}": v / max(n, 1) for k, v in totals.items()}


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    run_name = args.run_name or f"{args.square_mode}_s{args.seed}"
    run_dir = Path(args.runs_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_config = {k: getattr(args, k) for k in DEFAULT_CONFIG}
    data_dir = Path(args.data_dir)

    print(f"loading {args.dataset} ...")
    train_data = load_split(args.dataset, data_dir, "train")
    val_data = load_split(args.dataset, data_dir, "val")
    mean, std = action_stats(train_data)

    clip_kwargs = dict(
        history_size=args.history_size,
        num_preds=args.num_preds,
        frameskip=args.frameskip,
        square_mode=args.square_mode,
        square_size=args.square_size,
        stats=(mean, std),
        img_size=args.img_size,
    )
    train_set = ClipDataset(train_data, seed=args.seed, **clip_kwargs)
    # A different seed keeps val episode colors independent of train's.
    val_set = ClipDataset(val_data, seed=args.seed + 1, **clip_kwargs)
    pin = device.type == "cuda"
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True,
                              num_workers=args.num_workers, pin_memory=pin)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"train clips {len(train_set)}  val clips {len(val_set)}  device {device}")

    model = build_lewm(model_config).to(device)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    if args.max_steps_per_epoch is not None:
        steps_per_epoch = min(steps_per_epoch, args.max_steps_per_epoch)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, warmup_cosine(total_steps, max(1, int(args.warmup_frac * total_steps)))
    )

    # bf16 autocast on CUDA as upstream; plain fp32 elsewhere.
    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda")

    run_config = {"model": model_config, "train": {k: v for k, v in vars(args).items() if k not in DEFAULT_CONFIG},
                  "action_mean": mean.tolist(), "action_std": std.tolist()}
    (run_dir / "config.json").write_text(json.dumps(run_config, indent=2))
    metrics_file = open(run_dir / "metrics.jsonl", "a")

    def log(record: dict):
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()

    step = 0
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        for i, batch in enumerate(train_loader):
            if i >= steps_per_epoch:
                break
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with autocast():
                out = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            out["loss"].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            step += 1

            if step % args.log_every == 0 or step == 1:
                record = {"step": step, "epoch": epoch, "lr": scheduler.get_last_lr()[0],
                          "grad_norm": grad_norm.item(), **{f"train/{k}": v.item() for k, v in out.items()}}
                log(record)
                print(f"ep {epoch} step {step}/{total_steps}  loss {record['train/loss']:.4f}  "
                      f"pred {record['train/pred_loss']:.4f}  sigreg {record['train/sigreg_loss']:.3f}  "
                      f"lr {record['lr']:.2e}")

        val = validate(model, val_loader, device, autocast, args.max_val_batches)
        log({"step": step, "epoch": epoch, **val})
        print(f"epoch {epoch} done in {time.time() - t0:.0f}s  " + "  ".join(f"{k} {v:.4f}" for k, v in val.items()))

        ckpt = {"model": model.state_dict(), "config": run_config, "epoch": epoch, "step": step}
        torch.save(ckpt, run_dir / f"epoch_{epoch:03d}.pt")
        torch.save(ckpt, run_dir / "last.pt")

    metrics_file.close()


if __name__ == "__main__":
    main()
