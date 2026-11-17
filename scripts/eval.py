"""Evaluate a trained LeWM by planning in the OGBench environment.

    python scripts/eval.py --checkpoint runs/episode_s0/last.pt
    python scripts/eval.py --checkpoint runs/episode_s0/last.pt --tasks near --episodes 50
    python scripts/eval.py --checkpoint runs/episode_s0/last.pt --tasks dataset --max-steps 50

Each episode: encode the current frame and the goal frame, plan ``horizon``
action blocks with CEM on predicted-embedding distance to the goal, execute
``receding`` of them in the env, and replan until success or ``max_steps``.

Task sets:
  ogbench  the 5 standard OGBench cube-single tasks (cube moved 15-45 cm; long horizon)
  near     random start, goal 5-10 cm away; arm starts at home and the goal image
           shows the arm at home, so the cube has to be grasped and carried
  dataset  LeWM / DINO-WM style: the sim is set to a val-split state and the goal is
           the frame ``goal_offset`` steps later in the same trajectory; success when
           cube and gripper are both within 4 cm of their goal positions

The corner square is applied to env frames exactly as in training: one color
per eval episode (also on the goal frame) for ``episode``, a fresh color every
frame for ``frame``, none for ``baseline``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import ogbench
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from DataLoading.clip_dataset import episode_ids, preprocess_frames, stamp_square  # noqa: E402
from lewm.jepa import build_lewm  # noqa: E402
from planning.cem import CEM  # noqa: E402
from scripts.train import load_split  # noqa: E402

SUCCESS_DIST = 0.04  # OGBench's cube success radius, also used for the gripper in dataset tasks


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tasks", choices=["ogbench", "near", "dataset"], default="ogbench")
    p.add_argument("--goal-offset", type=int, default=25, help="dataset tasks: env steps from start to goal frame")
    p.add_argument("--data-dir", default=str(REPO_ROOT / "Data" / "ogbench"))
    p.add_argument("--episodes", type=int, default=50, help="total episodes (split evenly over the 5 ogbench tasks)")
    p.add_argument("--max-steps", type=int, default=200, help="env steps per episode (OGBench uses 200)")
    p.add_argument("--horizon", type=int, default=5, help="planned action blocks (model steps)")
    p.add_argument("--receding", type=int, default=5, help="blocks executed before replanning")
    p.add_argument("--samples", type=int, default=300)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--square-mode", choices=["baseline", "episode", "frame"], default=None,
                   help="defaults to the mode the checkpoint was trained with")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="results json; defaults next to the checkpoint")
    return p.parse_args()


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def near_task(rng: np.random.Generator, bounds: np.ndarray) -> dict:
    """Random start inside the target bounds, goal 5-10 cm away in a random direction."""
    lo, hi = bounds[0] + 0.05, bounds[1] - 0.05
    start = rng.uniform(lo, hi)
    angle, dist = rng.uniform(0, 2 * np.pi), rng.uniform(0.05, 0.10)
    goal = np.clip(start + dist * np.array([np.cos(angle), np.sin(angle)]), bounds[0], bounds[1])
    return dict(
        task_name="near",
        init_xyzs=np.array([[*start, 0.02]]),
        goal_xyzs=np.array([[*goal, 0.02]]),
    )


class DatasetTasks:
    """Start/goal pairs from the val split, ``offset`` env steps apart in one trajectory."""

    def __init__(self, env, dataset: str, data_dir: Path, offset: int):
        self.u = env.unwrapped
        data = load_split(dataset, data_dir, "val")
        raw = np.load(data_dir / f"{dataset}-val.npz")
        self.obs = data["observations"]                      # (N, H, W, 3) uint8
        self.qpos, self.qvel = raw["qpos"], raw["qvel"]      # (N, 21), (N, 20)
        episodes = episode_ids(data)
        starts = np.arange(len(episodes) - offset)
        self.starts = starts[episodes[starts] == episodes[starts + offset]]
        self.offset = offset

    def _set(self, t: int):
        self.u._data.qpos[:] = self.qpos[t]
        self.u._data.qvel[:] = self.qvel[t]
        mujoco.mj_forward(self.u._model, self.u._data)

    def positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Current (cube xyz, gripper pinch-site xyz)."""
        return (self.u._data.joint("object_joint_0").qpos[:3].copy(),
                self.u._data.site_xpos[self.u._pinch_site_id].copy())

    def reset(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, tuple]:
        """Put the sim at a sampled start that is not already a success;
        return (start frame, goal frame, goal positions)."""
        while True:
            t = int(rng.choice(self.starts))
            self._set(t + self.offset)
            goal_pos = self.positions()
            self._set(t)
            if any(np.linalg.norm(a - b) > SUCCESS_DIST for a, b in zip(self.positions(), goal_pos)):
                return self.u.get_pixel_observation(), np.array(self.obs[t + self.offset]), goal_pos


class SquareTagger:
    """Stamps env frames the same way ClipDataset stamps training frames."""

    def __init__(self, mode: str, size: int, rng: np.random.Generator):
        self.mode, self.size, self.rng = mode, size, rng
        self.episode_color = None

    def new_episode(self):
        self.episode_color = self.rng.integers(0, 256, size=3, dtype=np.uint8)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if self.mode == "baseline":
            return frame
        color = self.episode_color if self.mode == "episode" else self.rng.integers(0, 256, size=3, dtype=np.uint8)
        return stamp_square(frame[None].copy(), color[None], self.size)[0]


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = pick_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    run_config = ckpt["config"]
    cfg = run_config["model"]
    train_cfg = run_config["train"]
    model = build_lewm(cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    frameskip, action_dim = cfg["frameskip"], cfg["action_dim"]
    block_dim = frameskip * action_dim
    mean = torch.tensor(np.tile(run_config["action_mean"], frameskip), dtype=torch.float32, device=device)
    std = torch.tensor(np.tile(run_config["action_std"], frameskip), dtype=torch.float32, device=device)
    square_mode = args.square_mode or train_cfg["square_mode"]
    # Own generator, so every square mode sees the same sequence of tasks.
    tagger = SquareTagger(square_mode, train_cfg["square_size"], np.random.default_rng(args.seed + 1))

    env = ogbench.make_env_and_datasets(train_cfg["dataset"], env_only=True)
    bounds = env.unwrapped._target_sampling_bounds
    planner = CEM(num_samples=args.samples, topk=args.topk, iters=args.iters)
    if args.tasks == "dataset":
        dataset_tasks = DatasetTasks(env, train_cfg["dataset"], Path(args.data_dir), args.goal_offset)

    def embed(frame: np.ndarray) -> torch.Tensor:
        pixels = preprocess_frames(tagger(frame)[None], cfg["img_size"])  # (1, 3, H, W)
        return model.encode(pixels[None].to(device))                      # (1, 1, D)

    results = []
    t0 = time.time()
    for ep in range(args.episodes):
        if args.tasks == "ogbench":
            task_id = ep % 5 + 1
            options = dict(task_id=task_id)
            name = f"task{task_id}"
        else:
            options = dict(task_info=near_task(rng, bounds))
            name = "near"
        ob, info = env.reset(seed=int(rng.integers(1 << 31)), options=options)
        if args.tasks == "dataset":
            name = "dataset"
            ob, goal_frame, goal_pos = dataset_tasks.reset(rng)
            info = {"goal": goal_frame}
            start_dists = [float(np.linalg.norm(a - b)) for a, b in zip(dataset_tasks.positions(), goal_pos)]
        tagger.new_episode()
        goal_emb = embed(info["goal"])

        def dataset_success() -> bool:
            return all(np.linalg.norm(a - b) <= SUCCESS_DIST for a, b in zip(dataset_tasks.positions(), goal_pos))

        success, steps = False, 0
        while steps < args.max_steps and not success:
            emb = embed(ob)

            def cost_fn(candidates: torch.Tensor) -> torch.Tensor:
                return model.plan_cost(emb, goal_emb, (candidates - mean) / std)

            plan = planner.plan(cost_fn, args.horizon, block_dim, device=device)  # (horizon, block_dim) raw
            raw_actions = plan[: args.receding].reshape(-1, action_dim).cpu().numpy()  # (receding * fs, A)
            for action in raw_actions:
                ob, _, terminated, truncated, info = env.step(action)
                steps += 1
                success = success or (dataset_success() if args.tasks == "dataset" else bool(info["success"]))
                if success or terminated or truncated or steps >= args.max_steps:
                    break
            if terminated or truncated:
                break

        result = {"episode": ep, "task": name, "success": success, "steps": steps}
        if args.tasks == "dataset":
            end_dists = [float(np.linalg.norm(a - b)) for a, b in zip(dataset_tasks.positions(), goal_pos)]
            result.update(cube_dist_start=start_dists[0], gripper_dist_start=start_dists[1],
                          cube_dist_end=end_dists[0], gripper_dist_end=end_dists[1])
        results.append(result)
        rate = np.mean([r["success"] for r in results])
        print(f"episode {ep:3d} {name:6s} success {int(success)} steps {steps:3d}  "
              f"running success {rate:.2%}  ({time.time() - t0:.0f}s)")

    per_task = {}
    for r in results:
        per_task.setdefault(r["task"], []).append(r["success"])
    summary = {
        "checkpoint": str(args.checkpoint),
        "square_mode": square_mode,
        "tasks": args.tasks,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "per_task": {k: float(np.mean(v)) for k, v in per_task.items()},
        **({f"mean_{k}": float(np.mean([r[k] for r in results]))
            for k in ("cube_dist_start", "cube_dist_end", "gripper_dist_start", "gripper_dist_end")}
           if args.tasks == "dataset" else {}),
        "args": vars(args),
        "episodes": results,
    }
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name(
        f"eval_{args.tasks}_{Path(args.checkpoint).stem}.json"
    )
    out.write_text(json.dumps(summary, indent=2))
    print(f"success rate {summary['success_rate']:.2%}  per task {summary['per_task']}  -> {out}")


if __name__ == "__main__":
    main()
