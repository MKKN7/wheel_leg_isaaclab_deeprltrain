"""Isaac Lab keyboard controller entry point for the RMUC Level18 policy.

W/S command forward/reverse speed, A/D command left/right yaw. Commands ramp
to their targets and ramp down after release so the policy never sees a hard
step in velocity or yaw. Extra arguments are forwarded to ``play.py``.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs/rsl_rl/rmuc_wheel_leg_course/2026-09-18_21-50-10_command_level17_stage_survival_cert_v87"
    / "model_level_18_entry_36.pt"
)


def main() -> None:
    args = list(sys.argv[1:])

    def has_option(name: str) -> bool:
        return any(item == name or item.startswith(f"{name}=") for item in args)

    defaults = [
        ("--task", "Robotics-RMUC-WheelLeg-Command-v0"),
        ("--num_envs", "1"),
        ("--checkpoint", str(DEFAULT_CHECKPOINT)),
        ("--command_level", "18"),
    ]
    for name, value in defaults:
        if not has_option(name):
            args.extend((name, value))
    if not has_option("--keyboard"):
        args.append("--keyboard")
    if not has_option("--real-time"):
        args.append("--real-time")
    if not has_option("--viz"):
        args.extend(("--viz", "kit"))

    play_script = ROOT / "scripts/rsl_rl/play.py"
    sys.path.insert(0, str(play_script.parent))
    sys.argv = [str(play_script), *args]
    runpy.run_path(str(play_script), run_name="__main__")


if __name__ == "__main__":
    main()
