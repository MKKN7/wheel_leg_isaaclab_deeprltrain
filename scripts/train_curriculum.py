"""Train RMUC in gated stages.

Each stage is retrained until its evaluation gate passes. A failed stage never
advances; stages after it are not launched. For stage N>0, retries restart from
the last checkpoint that passed stage N-1.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
ISAACLAB_BAT = Path(os.environ.get("ISAACLAB_BAT", r"D:\issac\IsaacLab\isaaclab.bat"))
STAGES = ("stand", "command", "rough_robust")


def _load_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "configs" / "rmuc_wheel_leg.yaml").read_text(encoding="utf-8"))


def _checkpoint_number(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def _latest_new_checkpoint(before_dirs: set[Path]) -> Path | None:
    root = REPO_ROOT / "logs" / "rsl_rl" / "rmuc_wheel_leg_course"
    candidates = []
    for directory in root.glob("*"):
        if not directory.is_dir() or directory in before_dirs:
            continue
        for path in directory.glob("model_*.pt"):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, _checkpoint_number(path)))


def _run(command: list[str]) -> int:
    print("[RMUC_CURRICULUM] RUN " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_stage", choices=STAGES, default="stand")
    parser.add_argument("--num_envs", type=int, default=128)
    parser.add_argument("--max_retries", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gate_steps", type=int, default=None)
    args = parser.parse_args()
    config = _load_config()
    curriculum = dict(config.get("curriculum", {}))
    gate = dict(curriculum.get("gate", {}))
    max_retries = int(args.max_retries or gate.get("max_retries_per_stage", 3))
    stage_index = STAGES.index(args.start_stage)
    checkpoint: Path | None = None

    for stage in STAGES[stage_index:]:
        stage_cfg = dict(curriculum.get(stage, {}))
        task = str(stage_cfg["task"])
        iterations = int(stage_cfg["max_iterations"])
        passed = False
        for attempt in range(1, max_retries + 1):
            before_dirs = {
                path for path in (REPO_ROOT / "logs" / "rsl_rl" / "rmuc_wheel_leg_course").glob("*") if path.is_dir()
            }
            train_cmd = [
                str(ISAACLAB_BAT),
                "-p",
                "scripts\\rsl_rl\\train.py",
                "--task",
                task,
                "--num_envs",
                str(args.num_envs),
                "--max_iterations",
                str(iterations),
                "--viz",
                "none",
                "--device",
                args.device,
            ]
            if checkpoint is not None:
                train_cmd.extend(["--checkpoint", str(checkpoint), "--resume_training"])
            print(f"[RMUC_CURRICULUM] stage={stage} attempt={attempt}/{max_retries}", flush=True)
            if _run(train_cmd) != 0:
                print(f"[RMUC_CURRICULUM] training failed at stage={stage}; retrying", flush=True)
                continue
            candidate = _latest_new_checkpoint(before_dirs)
            if candidate is None:
                print(f"[RMUC_CURRICULUM] no checkpoint found at stage={stage}; retrying", flush=True)
                continue
            eval_cmd = [
                str(ISAACLAB_BAT),
                "-p",
                "scripts\\evaluate_rmuc_stage.py",
                "--task",
                task,
                "--checkpoint",
                str(candidate),
                "--num_envs",
                str(min(args.num_envs, 8)),
                "--device",
                args.device,
                "--viz",
                "none",
            ]
            if args.gate_steps is not None:
                eval_cmd.extend(["--steps", str(args.gate_steps)])
            gate_code = _run(eval_cmd)
            if gate_code == 0:
                checkpoint = candidate
                passed = True
                print(f"[RMUC_CURRICULUM] PASS stage={stage} checkpoint={candidate}", flush=True)
                break
            print(
                f"[RMUC_CURRICULUM] FAIL stage={stage}; retrying from "
                f"{checkpoint or 'scratch'}",
                flush=True,
            )
        if not passed:
            print(f"[RMUC_CURRICULUM] BLOCKED stage={stage}; later stages will not run", flush=True)
            return 2
    print(f"[RMUC_CURRICULUM] COMPLETE checkpoint={checkpoint}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
