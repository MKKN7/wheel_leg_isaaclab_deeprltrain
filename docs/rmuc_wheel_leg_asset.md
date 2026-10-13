# RMUC closed-loop asset integration

The tracked RMUC asset is the byte-for-byte copy of the supplied
`rmuc_robot_closed_loop.usd`.  The open-chain `rmuc_robot_only.usd` is kept as
a diagnostic fallback.  No mesh, mass, inertia, joint limit, axis, damping,
drive, or hardware parameter is rewritten by the asset configuration.

## What is configured

- `RMUC_WHEEL_LEG_CFG` points at the closed-loop USD and enables contact
  sensors.
- Only the six authored policy motors are exposed (`active`, `hip`, and
  `wheel` on each side).  The linkage and guide-wheel joints remain passive
  and are not assigned a synthetic actuator.
- The initial state uses a closure-consistent zero-angle pose at z=0.50 m and
  the source asset's +90 degree X frame.  The former approximate standing
  angles are retained as a non-default tuning profile because they generated
  a large first-step constraint impulse.
- The task-side contact sensor regex covers the chassis and all links;
  wheel contact reward still selects `left_wheel` and `right_wheel`.
- Safety thresholds and torque ratios are read from
  `configs/rmuc_wheel_leg.yaml`.
- The Isaac articulation audit reports 39 rigid bodies with a total mass of
  `18.970039 kg`. The root/chassis body alone is `16.227064 kg`; do not use
  that root-only value as the complete robot mass in actuator calculations.

## V14 compatibility decision

The legacy `WheelbipeV14Env` cannot consume this USD directly.  It requires
literal names such as `left_front1_joint`, `base_link`, `*_wheel_link`, and
two `*_spring2_joint` entries.  The RMUC USD intentionally has different
hardware names (`left_active_link_pitch`, `left_wheel_spin`, etc.), so adding
fake aliases would create duplicate/ambiguous physics objects and invalidate
the closed loop.

V14-style task IDs are therefore registered as aliases to the RMUC adapter:

```text
Robotics-Wheelbipe-V14-RMUC-Flat-v0
Robotics-Wheelbipe-V14-RMUC-Flat-Play-v0
Robotics-Wheelbipe-V14-RMUC-Play-v0
```

The adapter reports both the literal V14 check and the semantic RMUC mapping
(4 active leg joints, 2 wheel joints, 12 linkage joints, 12 linkage links).

## Validation order

Run the USD validator with Isaac Sim's Python before PPO:

```powershell
python scripts/validate_rmuc_usd.py --usd source/agent_world/agent_world/assets/usd_files/rmuc_wheel_leg/rmuc_robot_closed_loop.usd --allow-authored-zero-mass
```

The `--allow-authored-zero-mass` flag is diagnostic only.  The imported source
contains massless linkage shells and the measured chassis mass; those values
are preserved.  A single-environment Isaac Sim smoke test must confirm that
the articulation stays stable, loop targets are valid, wheels reach the
ground, and contact forces are finite before large-scale training.

If the source USD must be regenerated, use:

```powershell
python scripts/prepare_rmuc_closed_loop_usd.py --input <open-chain.usd> --output <closed-loop.usd>
```

That script moves only `ArticulationRootAPI` to the discovered physical
`robot` prim and adds four excluded spherical loop constraints with validated
body paths.

## Training entry point

Install the three local packages into the same Isaac Lab environment.  The
project accepts `rsl-rl-lib` 2.x through 5.x; the local compatibility helper
supplies the removed `store_code_state` API, while the stock runner config is
normalized to the installed RSL-RL model schema.

```powershell
python -m pip install -e source/agent_world
python -m pip install -e source/agent_tasks
python -m pip install -e source/agent_rl
$env:WARP_CACHE_PATH = "$PWD/.warp_cache"
$env:PYTHONPATH = "$PWD/source/agent_world;$PWD/source/agent_tasks;$PWD/source/agent_rl;$PWD"
& <ISAACLAB_ROOT>\isaaclab.bat -p scripts/rsl_rl/train.py --task Robotics-Wheelbipe-V14-RMUC-Flat-v0 --num_envs 128 --device cuda:0 --max_iterations 20000 --viz none
```

Before a long run, execute `scripts/probe_rmuc_asset.py --steps 100` with the
same environment.  The validated smoke profile uses the closure-consistent
zero-angle reset; do not replace it with the old approximate standing angles
until a new single-environment residual test passes.
