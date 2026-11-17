"""Figures comparing the baseline / episode / frame runs.

    python scripts/plot_runs.py --runs-dir runs/unity --out docs/figures

Reads each <mode>_s0/{metrics.jsonl, analysis_epoch_009.json} and writes PNGs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from DataLoading.clip_dataset import square_colors, stamp_square  # noqa: E402
from scripts.train import load_split  # noqa: E402

MODES = ("baseline", "episode", "frame")
COLORS = {"baseline": "tab:gray", "episode": "tab:red", "frame": "tab:blue"}


def read_metrics(path: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return [r for r in rows if "train/loss" in r], [r for r in rows if "val/loss" in r]


def smooth(y: np.ndarray, k: int = 25) -> np.ndarray:
    return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs/unity")
    p.add_argument("--out", default="docs/figures")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    runs, out = Path(args.runs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for mode in MODES:
        train, val = read_metrics(runs / f"{mode}_s{args.seed}" / "metrics.jsonl")
        steps = np.array([r["step"] for r in train])
        for ax, key in zip(axes[:2], ("train/pred_loss", "train/sigreg_loss")):
            y = smooth(np.array([r[key] for r in train]))
            ax.plot(steps[len(steps) - len(y):], y, color=COLORS[mode], label=mode)
        axes[2].plot([r["epoch"] + 1 for r in val], [r["val/pred_loss"] for r in val],
                     "o-", color=COLORS[mode], label=mode)
    for ax, title in zip(axes, ("train prediction loss (MSE)", "train SIGReg", "val prediction loss per epoch")):
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_xlabel("step")
    axes[1].set_xlabel("step")
    axes[2].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=120)

    # Probes and sensitivity
    analyses = {m: json.loads((runs / f"{m}_s{args.seed}" / "analysis_epoch_009.json").read_text()) for m in MODES}
    probe_keys = [("cube_xyz_r2_clean", "cube xyz\n(no square)"), ("cube_xyz_r2_square", "cube xyz\n(square)"),
                  ("arm_joints_r2_clean", "arm joints\n(no square)"), ("arm_joints_r2_square", "arm joints\n(square)"),
                  ("square_rgb_r2", "square RGB")]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), gridspec_kw={"width_ratios": [3, 1, 1.5]})
    x = np.arange(len(probe_keys))
    for i, mode in enumerate(MODES):
        vals = [max(analyses[mode]["probes"][k]["mean"], 0) for k, _ in probe_keys]
        axes[0].bar(x + (i - 1) * 0.27, vals, 0.27, color=COLORS[mode], label=mode)
    axes[0].set_xticks(x, [label for _, label in probe_keys])
    axes[0].set_ylim(0, 1.2)
    axes[0].set_title("linear probe R² on held-out val episodes (clipped at 0)")
    axes[0].legend(loc="upper center", ncol=3)
    shares = [analyses[m]["sensitivity"]["color_variance_share"] for m in MODES]
    axes[1].bar(MODES, shares, color=[COLORS[m] for m in MODES])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("embedding variance\nexplained by square color")
    for mode in MODES:
        r = analyses[mode]["rollout"]["model_over_copy_per_step"]
        axes[2].plot(range(1, len(r) + 1), r, "o-", color=COLORS[mode], label=mode)
    axes[2].set_title("rollout error / copy-baseline error")
    axes[2].set_xlabel("predicted block (5 env steps each)")
    axes[2].set_yscale("log")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(out / "probes.png", dpi=120)

    # Planning on dataset-goal tasks
    evals = {m: runs / f"{m}_s{args.seed}" / "eval_dataset_epoch_009.json" for m in MODES}
    if all(p.exists() for p in evals.values()):
        evals = {m: json.loads(p.read_text()) for m, p in evals.items()}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        rates = np.array([evals[m]["success_rate"] for m in MODES])
        n = np.array([len(evals[m]["episodes"]) for m in MODES])
        axes[0].bar(MODES, rates, yerr=1.96 * np.sqrt(rates * (1 - rates) / n), capsize=6,
                    color=[COLORS[m] for m in MODES])
        axes[0].set_ylim(0, 1)
        axes[0].set_title("planning success, dataset-goal tasks (95% CI)")
        x = np.arange(4)
        keys = ["cube_dist_start", "cube_dist_end", "gripper_dist_start", "gripper_dist_end"]
        for i, mode in enumerate(MODES):
            axes[1].bar(x + (i - 1) * 0.27, [100 * evals[mode][f"mean_{k}"] for k in keys], 0.27,
                        color=COLORS[mode], label=mode)
        axes[1].axhline(100 * 0.04, color="k", ls="--", lw=1, label="4 cm success radius")
        axes[1].set_xticks(x, ["cube\nstart", "cube\nend", "gripper\nstart", "gripper\nend"])
        axes[1].set_ylabel("cm to goal position")
        axes[1].set_title("mean distance to goal")
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(out / "planning.png", dpi=120)

    # What the inputs look like
    cfg = json.loads((runs / f"baseline_s{args.seed}" / "config.json").read_text())["train"]
    data = load_split(cfg["dataset"], REPO_ROOT / "Data" / "ogbench", "val")
    idx = 1001 * 3 + np.arange(0, 20, 5)  # 4 model steps from val episode 3
    fig, axes = plt.subplots(3, len(idx), figsize=(2.2 * len(idx), 7))
    for row, mode in enumerate(MODES):
        frames = data["observations"][idx]
        colors = square_colors(data, mode, args.seed + 1)
        if colors is not None:
            stamp_square(frames, colors[idx], cfg["square_size"])
        for col, frame in enumerate(frames):
            axes[row, col].imshow(frame)
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
        axes[row, 0].set_ylabel(mode, fontsize=12)
    fig.suptitle("one training clip (frameskip 5) under each square mode")
    fig.tight_layout()
    fig.savefig(out / "square_modes.png", dpi=120)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
