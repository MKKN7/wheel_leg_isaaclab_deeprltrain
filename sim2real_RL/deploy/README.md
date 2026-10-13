# RMUC low-level deployment core

`rmuc_low_level.c` is a portable actuator boundary for the 25D/6D E2E actor.
It does not run ONNX, estimate state, read CAN, or implement a task-level
speed/pitch/yaw controller. The caller must build the exact exported 25D
observation, call the ONNX policy, apply the exported `consume_action` rule,
and provide the resulting six actions every 8 ms. A missing policy update
returns `RMUC_POLICY_DEADLINE` and zero output.

Call `rmuc_low_level_step` every 1 ms with joint-output-side radians and
rad/s in policy order `[right_active, right_hip, right_wheel, left_active,
left_hip, left_wheel]`. Pass the command-dependent leg scale from the
checkpoint contract, not a fixed 0.22 rad. The core computes and holds
`clip(60*(q_target-q)-2*qd, +/-40 Nm)` on 2 ms ticks. It emits direct wheel
output-shaft electromagnetic torque
`clip(5.5*sign*clip(action,+/-1) - D_fb*joint_velocity, +/-5.5 Nm)` every 1 ms.
Set each `wheel_feedback_damping_nm_s_rad` explicitly from the trained
`active_motor_v1` contract. Zero preserves the original direct-torque core.
For the current training stage, active D=.05 and simulated passive b=.002
give a local unsaturated total of .052 Nm*s/rad. The explicit stage target
`RMUC_STAGE_D_COMP_NM_S_RAD` is .052 per wheel only if real passive b is
treated as negligible; otherwise set D_comp=.052-b_real after measurement.
This does not reproduce the simulated post-clip passive term during motor
saturation. The value must be checked in on-robot replay before release.
The output reports policy, requested damping, and clipped electromagnetic
components separately; only the last is converted to current. Passive b/C
must not be added to the current command. Saturation makes active feedback
different from an unlimited post-clip mechanical damper; train this exact law.
The 3 Nm continuous figure is reported only as a thermal/load flag. Mechanical
wheel b/C losses and the 10.775 Nm/leg gas spring are plant terms, not extra
controller commands or current clips.

`RmucHardwareCalibration` must contain measured counts per output-shaft Nm,
joint-to-motor signs, effective current count limits, and verified direct
current/leg torque interfaces. Initialization rejects unknown fields or a
count limit unable to represent the full 5.5 Nm command. The example
calibration in `tests/test_rmuc_low_level.c` is synthetic and must never be
copied into firmware. The old `combat-dm-balance-main` path converts wheel
torque with an admitted provisional coefficient and clips to +/-8192 counts;
under its own formula that clips at about 2.45 Nm. It also updates RobotTask
every 5 ms and adds a near-stop speed correction. Neither behavior is part of
this E2E interface.

The C core is a tested reference, not hardware deployment certification.
Integration still requires a verified 25D estimator, policy/actuator timing,
measured torque-current mapping, wheel signs, encoder units, leg torque
mapping, emergency stop, thermal history, and on-robot replay. The current
checkpoint does not yet pass nominal locomotion or MuJoCo behavior gates.

Host check:

```sh
gcc -std=c11 -Wall -Wextra -Werror -pedantic \
  sim2real_RL/deploy/rmuc_low_level.c tests/test_rmuc_low_level.c \
  -lm -o artifacts/rmuc_low_level_test
artifacts/rmuc_low_level_test
```
