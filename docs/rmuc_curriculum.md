# RMUC wheel-leg staged training

The RMUC asset uses a custom environment because its closed-loop topology and
joint names are not identical to Wheelbipe V14. Train the same six-action,
25-observation policy through these separate tasks and warm-start each changed
task from the prior checkpoint. A stage is not complete from iteration count alone:
`evaluate_rmuc_stage.py` must return exit code 0 for its configured gate.
`train_curriculum.py` retries failed stages and refuses to launch later stages.

## Stage 0: stand and closed-loop stability

Use `Robotics-Wheelbipe-V14-RMUC-Flat-v0` with the default flat terrain,
zero linear/yaw command, no pushes, no friction randomization, and no
observation delay. Train until the base remains upright and both drive wheels
stay in contact for a full episode. The gate checks upright fraction, tilt,
wheel contact, closed-loop residual, and velocity limits over an evaluation
window.

## Stage 1: gated flat command curriculum

Finetune from the validated `model_1225.pt` checkpoint with
`Robotics-Wheelbipe-V14-RMUC-Command-v0`. The policy remains 25-D: the
legacy lateral-command slot (zero-based index 7) is reused for absolute target
forward velocity, while index 6 carries heading-speed tracking error. Legacy
checkpoints must zero column 7 during the one-time migration. The task advances
through these zero-based gated levels (displayed one-based below for
readability):

1. `1.0 m/s`, straight tracking-error recalibration (new run starts here)
2. `1.0 m/s`, alternating `+/-0.10 rad/s` turns
3. `1.0 m/s`, alternating `+/-0.20 rad/s` turns
4. `1.0 m/s`, alternating `+/-0.30 rad/s` turns
5. `1.0 m/s`, alternating `+/-0.45 rad/s` large turns
6. `1.5 m/s`, straight
7. `1.5 m/s`, alternating `+/-0.10 rad/s` turns
8. `1.5 m/s`, alternating `+/-0.20 rad/s` turns
9. `1.5 m/s`, alternating `+/-0.30 rad/s` strict turn recovery
10. `1.2 m/s`, straight recovery after turn training
11. `1.35 m/s`, straight bridge
12. `1.5 m/s`, strict straight
13. `1.6 m/s`, straight bridge
14. `1.75 m/s`, straight bridge
15. `2.0 m/s`, straight
16. `2.0 m/s`, alternating `+/-0.25 rad/s` turns
17. `2.5 m/s`, straight bridge
18. `3.0 m/s`, straight
19. `2.0 m/s`, alternating `+/-0.30 rad/s` banked-turn reinforcement
20. `2.5 m/s`, straight recertification
21. `2.6 m/s`, straight
22. `2.7 m/s`, straight
23. `2.8 m/s`, straight
24. `2.9 m/s`, straight
25. `2.0 m/s`, alternating `+/-0.15 rad/s` banked turns
26. `2.2 m/s`, alternating `+/-0.15 rad/s` banked turns
27. `2.2 m/s`, alternating `+/-0.30 rad/s` banked turns
28. `2.4 m/s`, alternating `+/-0.15 rad/s` banked turns
29. `2.4 m/s`, alternating `+/-0.30 rad/s` banked turns
30. `2.5 m/s`, alternating `+/-0.15 rad/s` banked turns
31. `2.5 m/s`, alternating `+/-0.30 rad/s` banked turns
32. `2.6 m/s`, alternating `+/-0.15 rad/s` banked turns
33. `2.6 m/s`, alternating `+/-0.30 rad/s` banked turns
34. `2.7 m/s`, alternating `+/-0.15 rad/s` banked turns
35. `2.7 m/s`, alternating `+/-0.30 rad/s` banked turns
36. `2.8 m/s`, alternating `+/-0.15 rad/s` banked turns
37. `2.8 m/s`, alternating `+/-0.30 rad/s` banked turns
38. `2.9 m/s`, alternating `+/-0.15 rad/s` banked turns
39. `2.9 m/s`, alternating `+/-0.30 rad/s` banked turns
40. `3.0 m/s`, straight recertification
41. `3.1 m/s`, straight
42. `3.2 m/s`, straight
43. `3.0 m/s`, alternating `+/-0.15 rad/s` banked turns
44. `3.0 m/s`, alternating `+/-0.30 rad/s` banked turns
45. `3.2 m/s`, alternating `+/-0.15 rad/s` banked turns
46. `3.2 m/s`, alternating `+/-0.30 rad/s` banked turns
47. terminal duplicate used only to save a deterministic all-course checkpoint

Each level is held at its final command for at least 500,000 global environment
steps after its command transition completes. The default transition is 500,000
steps; Level 14 (`1.75 -> 2.0 m/s`) uses 2,000,000 steps and Level 15
(`2.0 m/s` straight -> turn) uses 1,000,000 steps. Advancement requires each ready
environment's 300-step mean horizontal-heading velocity to track the
corresponding 300-step mean target. Environments without a complete window are
reported as not-ready rather than counted as unstable. At least 45% of ready
environments must pass the stable-window gate, together with a yaw-error EMA,
at least 90% safe postures, 55% average wheel contact, and at least 95% of the
recent completed non-rehearsal episodes surviving 1200 steps. Survival uses an
EMA covering roughly the latest 128 complete episodes, so old successes cannot
hide later policy regression and ordinary episode age cannot make the gate
mathematically unreachable.
Retained stages also require at least 64 completed episodes for each stage and
yaw sign. An episode counts as survived only when it reaches the survival
horizon without any unsafe termination. A tilt after step 1200 but before the
normal timeout is therefore still a failure. Episodes that terminate before
the 300-step tracking window also remain survival failures; they cannot
disappear from the retention gate as survivor-biased missing samples.
This avoids treating the inverted pendulum's necessary instantaneous
fore/aft balancing motion as failed locomotion. Posture, contact, tracking, and
survival are evaluated independently instead of requiring both wheels and all
other conditions to be true simultaneously in every environment.
At the `1.0 m/s` baseline, forward-command normalization preserves the
checkpoint input (`0.3 * 2.0 == 1.0 * 0.6`). The new root-velocity input column
is zero-initialized in both actor and critic, preserving the exact initial
`model_1225.pt` output. Later speed, yaw, and leg-scale transitions use the
per-stage interpolation durations above. Turning
levels relax roll/attitude shaping but retain height, wheel-contact, survival,
and command-tracking gates. The three command observation slots carry
heading-speed error, absolute target speed, and yaw-rate error. This keeps the
policy at 25 dimensions while making mixed-speed control observable: equal
tracking errors at 1.0 and 1.75 m/s no longer look identical to the actor. When
importing a legacy `tracking_error` checkpoint, clear first-layer observation
column 7 with `--zero_observation_indices_for_finetune 7`; fixed replay datasets
are migrated to the target-speed meaning at load time. The vx error scale is
`2.1`. A bounded low-level adapter uses `-0.020` common-mode speed feedback
below 1.65 m/s, `-0.36` at and above 1.65 m/s, and `3.6` differential yaw
feedback. A 64-environment Level-8 A/B measured worst-direction errors of
0.188, 0.184, and 0.180 m/s at gains -0.010, -0.015, and -0.020 respectively,
with no tilt or reverse motion. Level-3/4 retests at -0.020 also passed.

The consolidated command checkpoint uses speed-routed frozen experts while
keeping the 25-D observation and 6-D action contract. Non-turning commands use
the certified 2.9 m/s base actor exactly. Turns through 1.5 m/s use the
certified low/medium-speed turn expert, with a smooth expert blend from 1.5 to
1.9 m/s. Turns at and above 1.9 m/s use the certified high-speed base. A
trainable residual adapter is gated off for straight commands and for speeds at
or below 2.0 m/s, so later high-speed turn training cannot overwrite the
already certified low/mid turns or straight-speed actor. Above 2.2 m/s, turn
extensions use a cumulative speed/yaw residual bank. Each small-yaw and
large-yaw stage has a separate zero-initialized head; only the current
curriculum head is trainable and every previously certified head is frozen.
Each head is exactly zero at the preceding speed/yaw anchor, preventing a new
2.4--3.2 m/s stage from changing an earlier command anchor. Checkpoints
containing either the adapter or the residual bank are detected automatically
by training, evaluation, and play.
Every bank stage also has separate positive- and negative-yaw correction
networks. Their command-sign gates isolate opposing turn gradients; a common
head learned by an older checkpoint remains frozen while the zero-initialized
directional corrections learn physical left/right asymmetry.
Command velocity tracking, progress, underspeed, reverse, and overspeed rewards
all use the same horizontal heading-projected root speed as the gate. Tracking
uses a `0.15 m/s` tolerance, and absolute error beyond that band receives an
additional relative-to-target hinge penalty clipped to `[0, 1]`. This preserves
a gradient when the policy is far below 1.5-3.0 m/s without making the speed
penalty unbounded.
Yaw and bank rewards are multiplied by a bounded speed-coupling score: they are
fully available inside the `0.15 m/s` band and decay to zero as the miss reaches
half the target speed. A slow in-place pivot therefore cannot outscore a
coordinated moving turn.
Turning also tracks the coordinated-bank target
`-atan(v_cmd * yaw_cmd / 9.81)` with a `1.5 deg` tolerance. This is about
`2.6 deg` at `1.5 m/s, 0.30 rad/s` and replaces the previous accidental
penalization of pitch as if it were roll.
The precise Gaussian bank score is paired with a bounded absolute bank-error
penalty scaled over 6 degrees. The dense term prevents its learning signal from
vanishing when one turn direction starts several degrees away from the target;
this was required to correct the positive-yaw under-bank seen at 2.4 m/s.
At 2.4 m/s and `|yaw|=0.30`, fixed-policy A/B reduced the bounded yaw-feedback
clip from 0.30 to 0.10 action. Tilt termination fell from 20.8% to 5.21%; its
worst speed error was 0.066 m/s and the remaining yaw miss was 0.0545 rad/s.
The directional policy head learns the final 0.0045 rad/s rather than asking a
hard differential controller to consume pitch-recovery common mode.
The policy residual bank therefore provides the remaining turn authority;
the hard feedback no longer consumes pitch-recovery common mode.
Command reward/observation migrations use only a short critic-only warmup before
actor updates. The high-speed continuation command uses 128 control steps,
`5e-6`, one PPO epoch, and `gamma=0.99, lambda=0.95`. Actor warmup is capped
below 10% of the current stage transition so it cannot miss the command ramp.
The high-speed acquisition term is a bounded progress bonus above the 1.0 m/s
baseline, enabled only for safe non-rehearsal environments; falling, low root
height, and large lateral velocity do not earn it. Rehearsal mixes the previous
stage with fixed 1.0 m/s and 1.5 m/s turning anchors, so later 1.75-3.0 m/s
training cannot erase the low- and medium-speed turning skills.
Level-8 recovery uses a fixed `2e-5` rate and one PPO epoch (adaptive KL is
disabled so a quiet policy cannot trigger a late learning-rate increase),
and frozen action std `[0.004, 0.004, 0.008, 0.004, 0.004, 0.008]`. The fixed
wheel std removes the stochastic late-training collapse seen at the viable
`model_245.pt` state. From Level 8, 30% of environments rehearse previous
turning commands while 70% learn the active turn. After the Level-8 gate passes,
Level 9 uses wheel std `0.012`; do not jump directly to a high-speed turning
level. The same 30% on-policy rehearsal protects earlier skills while 70% learn
the new stage. Population-quota assignment at reset keeps this fraction stable;
independent Bernoulli replacement allowed the live fraction to drift.
Three quarters of the rehearsal cohort use weighted priority anchors: 20% for
the `1.0 m/s` large turn (stage 4), 55% for the fragile
`1.5 m/s + 0.30 rad/s` turn (stage 8), and 25% for the certified `1.75 m/s`
straight skill (stage 13). TensorBoard exposes this as
`Episode/command_rehearsal_priority_fraction`, so later speed stages can be
checked for explicit medium-turn retention rather than inferred from a mixed
command average.
Levels 13 and 14 continue sampling and reporting the Level-4/8 turn anchors,
but their straight-speed gates do not wait for turn retention. Level 15 is the
first `2.0 m/s` yaw stage and enables the mandatory balanced retention gate;
`Episode/command_retention_gate_enforced` makes this boundary explicit. Fixed
low/medium-speed turn replay distills wheel differential plus bounded
common-mode/leg posture on observations carrying their original target-speed
slot. Schema-v2 replay records the current observation mode/scale and
per-sample command; legacy replay without that metadata is rejected. Each replay
batch is balanced across positive and negative yaw. The current Level-15 active
gate also checks positive and negative yaw independently, so one good direction
cannot hide failure of the other in aggregate means.
Retention windows use the same post-reset warmup as the active gate; launch
acceleration is not counted as evidence that an older turn skill was forgotten.
Turn episode signs are rebalanced on reset so frequent failures in one
direction do not remove that direction from the on-policy sample stream. The
trainer prepares and caches the bounded residual wheel/feedback action for
PhysX, while PPO retains the raw actor sample and its original log probability.
The residual is a deterministic outer-loop controller, not a sample from the
actor's narrow Gaussian distribution; scoring it as one can overflow the PPO
likelihood ratio when the high-speed action standard deviation is small.
Level 15 additionally overrides the shared high-speed gate with a `0.10 m/s`
absolute limit, certifying the `2.0 m/s` turn only inside `1.9-2.1 m/s`. A
bounded proportional high-speed common-mode feedback uses the same `0.16 s`
filtered speed state as the reward. It is symmetric across yaw directions and
cannot alternate acceleration and braking every control step. TensorBoard
exposes direction-split speed and wheel-common diagnostics.
From Level 12, rehearsal is reinforced by two fixed deterministic teachers.
Straight rollout samples use the validated Level-12 speed actor, while non-zero
yaw samples use the validated Level-8 turn actor. Each PPO update is preceded by
three balanced distillation updates. Two gate-passing 32,000-sample teacher
trajectory datasets are replayed alongside current on-policy observations; this
prevents the turn target from disappearing when the speed-initialized student
has not yet visited the teacher's stable turning states. The resulting student
is still one 25-D/6-action MLP; teacher networks are training-only.
The XML's `0.01` wheel-joint damping is mechanical damping, while the existing
checkpoint was trained against an effective Isaac motor damping of `0.1`.
Changing it invalidated the old balance controller, so checkpoint continuation
retains `0.1`; a future XML-parity policy must use a staged bridge through
`--wheel_damping_override` rather than a one-step change. A fixed-policy A/B
confirmed that changing `0.1 -> 0.01` directly collapsed the Level-15 actor
from about `1.89 m/s` to `0.49 m/s`; 25% damping compensation also fell to
about `1.31 m/s`. The
optional damping compensation is disabled for checkpoint continuation. Isaac's
`IdealPDActuator` computes `effort_target - damping * joint_velocity`, so any
future compensation must add the matching velocity term; the old unused
subtract implementation had the wrong sign. It is explicitly rejected in
`real_hardware` mode. Deployment must use the measured actuator damping/current
loop and configured hardware torque cap instead of this simulation-only aid.
For an actuator bridge, `--wheel_damping_anneal_target 0.01` leaves the
constructed `0.1` Isaac actuator unchanged and cosine-anneals the exact positive
velocity feed-forward over `--wheel_damping_anneal_steps`. The same cosine
fraction now migrates the normalized wheel torque scale and explicit
`IdealPDActuator` effort limit from the checkpoint-compatible `5.5 Nm` to the
USD-authored `3.0 Nm`. Active and rehearsal environments share this actuator
contract, so retention results cannot rely on extra simulation-only torque.
Logs expose
`wheel_effective_damping`, `wheel_damping_anneal_fraction`, and compensation
torque. The bridge is disabled by default and cannot be combined with the
legacy fixed compensation ratio. When enabled, curriculum advancement is also
blocked until the anneal fraction reaches one; later levels retain the target
effective damping instead of restarting at `0.1`.

For the Level-28 bridge, stage 23 (`2.9 m/s`, zero yaw) is both a priority
rehearsal anchor and a mandatory retention-gate bucket. The active turning
cohort and the 2.9 m/s straight cohort use the same annealed 3 Nm actuator;
speed, posture and full-episode survival must all remain certified before the
curriculum can advance.

The bridge remains an experimental tool, not a prerequisite for the speed
curriculum. Fixed-policy evaluation at an effective damping of `0.01` collapsed
Level-15 speed to about `0.34 m/s`; constant common-mode action biases either
failed to recover steady speed or destabilized the inverted pendulum. The XML
`0.01` value describes passive joint damping, not the complete motor/current
controller's effective damping. Until hardware identification provides that
closed-loop value, the Level-16 speed curriculum retains the validated `0.1`
actuator damping and uses a two-million-transition command ramp. Future
sim-to-real randomization should span a measured neighborhood rather than force
the full actuator model to the XML passive value.
The supplied USD also authors a `3.0 Nm` angular-drive cap on each wheel;
the task's `5.5 Nm` scale is retained for checkpoint compatibility, while
`Episode/wheel_authored_effort_limit_nm` and torque-ratio diagnostics expose
the physical cap instead of treating 5.5 Nm as delivered torque.
high-speed acquisition bonus
uses bounded forward progress directly rather than the minimum of progress and
the narrow tracking kernel, so it retains a gradient while below target.
Its acceleration/regression signal uses a `0.16 s` low-pass speed state instead
of the raw 8 ms difference, preventing contact and balance noise from flipping
the reward sign every control step. A separate high-pass residual
penalty, `|v-lowpass(v)|`, activates after the same grace period and directly
penalizes alternating propulsion/braking without making steady underspeed or
overspeed shaping narrower. The severe-underspeed term is a smoothstep
penalty after 150 physical post-reset steps; its grace counter is independent of
RSL-RL's randomized logging episode age. The reward floor is `-25`, so a severe
underspeed sample and an unsafe terminal transition no longer collapse to the
same clipped return.
At Level 14 and above, a signed pitch target increases with command speed and
temporary underspeed (bounded at 5 degrees). A common-mode wheel residual tracks
that target with a maximum magnitude of `0.04` action (`0.22 Nm`), while pitch
rate and excessive absolute pitch remain separately penalized. Held wheel
exploration is common-mode; independent left/right noise is reserved for the
policy itself so straight-line speed exploration does not become artificial yaw.
The checkpoint-compatible speed-feedback gain remains `-0.005` below 1.65 m/s.
At and above that threshold, a deterministic 1800-step, two-direction Level-15
validation selects `-0.30` with a `0.08` action clip. It held positive/negative
yaw at `1.977/2.097 m/s`, with `0.097 m/s` worst-direction temporal error and
passed the unchanged command gate.
Active-level gate statistics are cleared once per level when the command ramp
finishes, so transition data cannot contaminate hold-only validation. Historical
command sidecars use the legacy cancelable speed average and are treated only as
checkpoint provenance; new gate files use schema v2 rolling absolute error after
the survival warmup. The regression guard arms only after the complete live gate
qualifies. A turn replay whose source policy did not pass its gate may supervise
only wheel differential; wheel common-mode and leg losses must both be zero.
This retains steering structure without letting a lower-speed teacher cap the
student's forward speed or high-speed posture.
Turn-direction sampling is stage-local and adaptive. Each active/rehearsal
stage focuses at most 65% of new episodes on its larger normalized speed/yaw/
stability deficit while retaining at least 35% coverage of the other sign. This
replaces the stale global positive-yaw bias that repaired Level 15 positive yaw
but starved Level 15 negative yaw even while Level 8 had the opposite weakness.
The legacy Level-13 teacher is not used for online straight supervision after
schema-v2 validation exposed its long-horizon acceleration/fall cycle. Only its
quality-filtered stable samples are replayed (`--teacher_speed_replay_only`).
Unsafe terminations receive a bounded terminal penalty, while normal episode
timeouts do not, so the 128-step high-speed PPO rollout receives an immediate signal for
failures that otherwise occur 7.5-9.2 seconds after the responsible acceleration.
Level-13 speed stabilization uses `--num_steps_per_env_override 128`. A direct
64-versus-128-step probe from the same actor reduced completed speed error from
about `0.418` to `0.289 m/s`, raised survival EMA from `0.648` to `0.957`, and
ended at `1.730 m/s`. This changes rollout collection only; the 25-D/6-action
checkpoint contract is unchanged.
Training environments start with randomized episode phases; this removes the
previous synchronized reset every roughly 23 PPO iterations. Those initial
partial episodes are excluded from the survival gate.
From Level 8 onward, stage advancement uses a `0.20 m/s` high-speed gate, with
the Level-15 `2.0 m/s + 0.25 rad/s` turn overriding it to `0.10 m/s`. A command
episode must also survive at least 1200 control steps, with 95%
of completed unassisted episodes passing. Complete-episode tracking averages the
same 300-step temporal errors as the live gate, so the unavoidable acceleration
transient cannot permanently block an otherwise stable policy.
The command gate uses the same 300-step window for linear and yaw error. Using
instantaneous yaw error for the gate made stochastic samples fail despite a
passing fixed-policy evaluation; the windowed yaw criterion is reset per
environment and at every level boundary. Fixed-level holdout evaluation
disables the rehearsal mixture. The flat local
terrain is `128 x 128 m`, covering
the 128-environment grid plus a full twelve-second episode at `3 m/s`; the
bounded XY safety radius is `40 m`.

## Progressive flat-domain robustness

Levels 29 through 46 stay in
`Robotics-Wheelbipe-V14-RMUC-Command-v0`; the terrain remains the same 128 m
flat plane. Domain-randomization strength is tied to gated command progress,
not iteration count. Level 29 begins at about 10.5% strength and every passed
command level expands the range smoothly until Level 46 reaches the configured
maximum. Resampling happens only at episode reset, so physical parameters do
not jump in the middle of a trajectory.

The maximum ranges are base mass/inertia `+/-5%`, robot-contact friction
`+/-15%`, wheel damping `+/-10%`, wheel torque gain `+/-5%`, normalized
observation noise `0.01`, and a probabilistic one-step observation delay. The
first Level-29 exposure is therefore only about `+/-0.53%` mass,
`+/-1.58%` friction, `+/-1.05%` damping, and `+/-0.53%` wheel gain. These
uncertainties are symmetric around the checkpoint-compatible actuator model;
they do not replace the separate, gated migration from `5.5 Nm` to the measured
`3.0 Nm` hardware contract.

Certified base/expert/turn heads remain frozen. A separate zero-initialized
domain residual bank supplies the adaptation capacity required by randomized
dynamics. Its five independently routed heads cover straight, positive/negative
small-yaw, and positive/negative large-yaw commands at every speed; this avoids
opposite-turn and straight/turn gradient cancellation. The selected output is
bounded to `0.04` action, so robustness training can correct dynamics without
replacing a certified skill. Rehearsal and the retention gate remain mandatory
for old speed/yaw points while the current command head learns a new point.

TensorBoard exposes `domain_randomization_strength`,
`domain_base_mass_scale`, `domain_friction_scale`,
`domain_wheel_damping_scale`, `domain_wheel_torque_gain`,
`domain_observation_noise_std`, and `domain_observation_delay_steps`. Playback
and deterministic evaluation disable randomization by default. Use
`--disable_progressive_flat_dr` only for a nominal ablation.

Example commands from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\source\agent_world;$PWD\source\agent_tasks;$PWD\source\agent_rl"
$env:WARP_CACHE_PATH = "$PWD\.warp_cache"

& 'D:\issac\IsaacLab\isaaclab.bat' -p scripts\rsl_rl\train.py `
  --task Robotics-Wheelbipe-V14-RMUC-Flat-v0 --num_envs 128 --max_iterations 3000 --viz none

# Recover Level 8 from the best checkpoint of the previous run. Deterministic
# evaluation of model_245 ran a complete 1499-step episode without a physical
# termination at 1.5 m/s + 0.30 rad/s. This recovery uses 70% active large-yaw
# samples and 30% rehearsal, predominantly 1.5 m/s straight.
PYTHONPATH=source/agent_tasks:source/agent_world:source/agent_rl \
CUDA_VISIBLE_DEVICES=0 \
/home/kellen/anaconda3/envs/issaclab/bin/python scripts/rsl_rl/train.py \
  --task Robotics-Wheelbipe-V14-RMUC-Command-v0 --num_envs 128 --max_iterations 1000 \
  --checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-12_16-17-42_command_turn15_progressive_v6/model_245.pt \
  --preserve_checkpoint_std --load_critic_for_finetune \
  --action_std_override 0.004,0.004,0.008,0.004,0.004,0.008 \
  --freeze_action_std \
  --learning_rate_override 2e-5 --num_learning_epochs_override 1 \
  --critic_warmup_iterations 5 \
  --command_level 8 --start_command_at_target --stop_at_command_level 9 \
  --highspeed_regression_guard \
  --allow_curriculum_override \
  --run_name command_level8_turn_recovery_v2 --viz none \
  2>&1 | tee train_command_level8_turn_recovery_v2.log

# Level-9 measured-speed recovery. The previous nominal 1.5 m/s gate passed a
# policy that actually sustained about 1.1 m/s. Start directly at 1.2 m/s,
# then use strict 1.35, 1.50, 1.60 and 1.75 m/s gates. Assisted reset episodes
# provide high-speed coverage but are excluded from curriculum advancement;
# half of that cohort is initialized exactly at v_cmd as target-state reference,
# while both the rolling window and complete unassisted episodes must pass.
# Held wheel noise is active only below v_cmd-0.05 m/s; overspeed beyond
# v_cmd+0.05 m/s is explicitly penalized.
PYTHONPATH=source/agent_tasks:source/agent_world:source/agent_rl \
CUDA_VISIBLE_DEVICES=0 \
/home/kellen/anaconda3/envs/issaclab/bin/python scripts/rsl_rl/train.py \
  --task Robotics-Wheelbipe-V14-RMUC-Command-v0 --num_envs 128 --max_iterations 3000 \
  --checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-12_22-06-26_temporal_wheel_noise_probe80/model_79.pt \
  --preserve_checkpoint_std --load_critic_for_finetune \
  --action_std_override 0.004,0.004,0.012,0.004,0.004,0.012 \
  --freeze_leg_action_std --critic_warmup_iterations 5 \
  --wheel_noise_hold_steps 32 \
  --command_level 9 --start_command_at_target --highspeed_reset_bootstrap \
  --highspeed_regression_guard \
  --allow_curriculum_override \
  --run_name command_level9_speed_band_v9 --viz none

# Consolidated Level-12 training. The student starts from the checkpoint that
# passed both fixed Level 12 straight and Level 8 large-yaw evaluations. It
# stops after Level 12 passes its online gate and Level 13 is entered.
PYTHONPATH=source/agent_tasks:source/agent_world:source/agent_rl \
CUDA_VISIBLE_DEVICES=0 \
/home/kellen/anaconda3/envs/issaclab/bin/python scripts/rsl_rl/train.py \
  --task Robotics-Wheelbipe-V14-RMUC-Command-v0 --num_envs 128 --max_iterations 1000 \
  --checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-13_17-35-08_smoke60_level12_teacher_replay_v1/model_level_12_best_tracking.pt \
  --preserve_checkpoint_std --load_critic_for_finetune \
  --action_std_override 0.004,0.004,0.012,0.004,0.004,0.012 \
  --freeze_leg_action_std --learning_rate_override 2e-5 --num_learning_epochs_override 1 \
  --command_level 12 --start_command_at_target --stop_at_command_level 13 \
  --highspeed_reset_bootstrap --highspeed_regression_guard \
  --teacher_speed_checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-13_10-38-14_command_level9_speed_band_v9/model_level_12_best_speed.pt \
  --teacher_turn_checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-12_16-17-42_command_turn15_progressive_v6/model_245.pt \
  --teacher_speed_dataset artifacts/rmuc_teacher_datasets/level12_speed_model_best_32000.pt \
  --teacher_turn_dataset artifacts/rmuc_teacher_datasets/level8_turn_model245_32000.pt \
  --teacher_replay_batch_size 2048 --teacher_distill_weight 0.20 --teacher_distill_updates 3 \
  --allow_curriculum_override --run_name command_level12_multiteacher_v1 --viz none \
  2>&1 | tee train_command_level12_multiteacher_v1.log

# Continue from the pre-v57 Level-15 actor/critic. v57 is not a baseline: strong
# common-mode/leg replay from non-passing turn teachers pulled wheel drive down.
# Distillation below is retention-only and cosine-anneals 0.02 -> 0.002. The
# student learns 2.0 m/s from PPO, while replay preserves the transferable
# Level-8/Level-15 steering differentials and the old 1.75 m/s speed states.
PYTHONPATH=source/agent_tasks:source/agent_world:source/agent_rl \
CUDA_VISIBLE_DEVICES=0 \
/home/kellen/anaconda3/envs/issaclab/bin/python scripts/rsl_rl/train.py \
  --task Robotics-RMUC-WheelLeg-Command-v0 \
  --num_envs 128 --max_iterations 2000 --num_steps_per_env_override 128 \
  --command_level 15 --start_command_at_target --stop_at_command_level 16 \
  --checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-16_12-57-05_command_level15_residual_rawppo_validation_v56/model_level_15_best_candidate.pt \
  --preserve_checkpoint_std --load_critic_for_finetune \
  --action_std_override 0.004,0.004,0.012,0.004,0.004,0.012 --freeze_action_std \
  --wheel_noise_hold_steps 16 --learning_rate_override 2e-6 \
  --num_learning_epochs_override 1 --highspeed_reset_bootstrap \
  --highspeed_regression_guard \
  --teacher_speed_checkpoint logs/rsl_rl/rmuc_wheel_leg_course/2026-09-14_15-07-38_ppo_only_ablation_v11/model_level_13_best_tracking.pt \
  --teacher_speed_dataset artifacts/rmuc_teacher_datasets/level13_v11_stable_tracking_v1.pt \
  --teacher_speed_replay_only \
  --teacher_turn_dataset artifacts/rmuc_teacher_datasets/level8_turn030_currentobs_balanced_v4.pt,artifacts/rmuc_teacher_datasets/level15_turn025_v56_currentobs_v1.pt \
  --teacher_turn_replay_only --teacher_distill_weight 0.02 \
  --teacher_distill_final_weight 0.002 --teacher_distill_anneal_iterations 300 \
  --teacher_distill_learning_rate 2e-6 --teacher_distill_updates 1 \
  --teacher_speed_anchor_weight 1.0 --teacher_turn_common_mode_weight 0 \
  --teacher_turn_leg_weight 0 --teacher_replay_batch_size 256 \
  --allow_curriculum_override \
  --run_name command_level15_annealed_diff_teacher_v58 --viz none \
  2>&1 | tee train_command_level15_annealed_diff_teacher_v58.log

# For Level 15, inspect command_active_target_yaw_abs rather than the mixed
# command_target_yaw_abs metric. Both command_active_turn_positive_* and
# command_active_turn_negative_* must pass, together with the Level-4/8/13
# command_retention_* buckets. The guard saves a qualified best checkpoint and
# stops only after a measured post-qualification regression; successful entry
# into Level 16 also saves an entry checkpoint and stops cleanly.

# Replace <stage1_checkpoint> with the resulting model_*.pt path.
& 'D:\issac\IsaacLab\isaaclab.bat' -p scripts\rsl_rl\train.py `
  --task Robotics-Wheelbipe-V14-RMUC-Rough-v0 --num_envs 128 --max_iterations 5000 `
  --checkpoint <stage1_checkpoint> --resume_training --viz none
```

Do not resume `model_8085.pt`: it was trained before the RMUC passive-drive,
reset, and action-semantics corrections.

For automatic gated execution, run `scripts/train_curriculum.py`. If a gate
fails, the same stage is retrained. Stage 1 retries restart from the last
passing Stage 0 checkpoint; Stage 2 retries restart from the last passing Stage
1 checkpoint.

```powershell
& 'D:\issac\IsaacLab\isaaclab.bat' -p scripts\train_curriculum.py `
  --start_stage stand --num_envs 128 --max_retries 3 --device cuda:0
```
