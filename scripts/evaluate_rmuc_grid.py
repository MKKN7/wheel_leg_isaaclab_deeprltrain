"""Evaluate a fixed RMUC actor on an explicit velocity/yaw grid.

Each command has its own complete-episode, steady-window, and terminal-state
statistics. The result is independent of training-level advancement and never
overwrites a checkpoint's curriculum sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
from rmuc_eval_layout import bucket_assignment
from rmuc_eval_commands import ramp_straight_commands
from rmuc_hardware_profile import apply_hardware_profile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--leg-transport-delay-ms', type=float, default=0.0)
parser.add_argument('--leg-transport-delay-probability', type=float, default=.5)
parser.add_argument('--gravity-direction-dr-strength', type=float, default=0.0)
parser.add_argument('--gravity-direction-dr-probability', type=float, default=.5)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--command-sequence', type=Path, default=None, help='Episode-time [seconds,vx,yaw] steps; requires world/leg traces; fixed-grid acceptance invalid.')
parser.add_argument('--distilled-student', type=Path, default=None, help='Student weights; checkpoint remains explicit teacher physical/observation contract source.')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--speeds', default='1.0,1.5,2.0,2.5,2.9,3.0,3.2')
parser.add_argument('--yaws', default='0,0.15,-0.15,0.30,-0.30')
parser.add_argument('--command-grid', type=Path, default=None,
                    help='Exact JSON list of [vx, yaw] pairs instead of a Cartesian product.')
parser.add_argument('--command-leg-scale-profile', type=Path, default=None,
                    help='Explicit training action-mapping profile; positive/negative vx use identical absolute-speed knots.')
parser.add_argument('--diagnostic-wheel-peak-nm', type=float, default=None, help='Explicit electromagnetic peak only; action scale and passive drag unchanged.')
parser.add_argument('--replicas', type=int, default=3)
parser.add_argument('--steps', type=int, default=34000)
parser.add_argument('--episode-length-seconds', type=float, default=None,
                    help='Explicit evaluation episode duration, e.g.13s for 2s settling plus full10s idle window.')
parser.add_argument('--min-episodes', type=int, default=64)
parser.add_argument('--diagnostic-speed-ramp-seconds', type=float, default=0.,
                    help='Diagnostic per-episode ramp from1m/s to final straight target; never a fixed-command certificate.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--layout-seed', type=int, default=None,
                    help='Shuffle command placement across environments with a separate, reproducible RNG.')
parser.add_argument('--diagnostic-plane-terrain', action='store_true',
                    help='Replace generated flat mesh with analytical plane; diagnostic only, never a certificate.')
parser.add_argument('--diagnostic-mesh-subdivisions', type=int, default=0,
                    help='Split flat mesh triangles without moving the surface; diagnostic only (0 to 6).')
parser.add_argument('--dr-strength', type=float, default=0.0)
parser.add_argument('--independent-wheel-dr-streams', action='store_true',
                    help='Paired damping/torque-only diagnostic: isolate each environment reset RNG sequence.')
parser.add_argument('--wheel-dr-reset-budget', type=int, default=None,
                    help='Fix DR preallocation to match a longer paired run; default is steps+2.')
parser.add_argument('--diagnostic-separated-wheel-profile', type=Path, default=None,
                    help='Explicit schema1 JSON/YAML separated wheel diagnostic; never certification.')
parser.add_argument('--diagnostic-hardware-profile', type=Path, default=None,
                    help='Explicit target mechanical/actuator YAML; diagnostic only, never an automatic deployment certificate.')
parser.add_argument('--diagnostic-hardware-fraction', type=float, default=1.,
                    help='Fraction0..1 from current configured mechanics toward the explicit target; requires hardware profile.')
parser.add_argument('--diagnostic-joint-dr-profile', type=Path,
                    help='Explicit physical/sensor/response DR settings; no automatic certification')
parser.add_argument('--motor-response-time-constants-ms', type=float, nargs=2, default=None,
                    help='Diagnostic independent effort-response time-constant range; not communication delay or a policy certificate.')
parser.add_argument('--dr-ablation', choices=['none', 'physics_only', 'observation_only', 'damping_torque_only'], default='none',
                    help='Diagnostic component isolation; never a full DR policy certificate.')
parser.add_argument('--record-leg-tracking', action='store_true', help='Read-only physical velocity/joint trace for post-training charts.')
parser.add_argument('--max-bank-error-deg', type=float, default=3.0)
parser.add_argument('--arena-ground-severity',type=float,default=None,
                    help='Explicit PVC contact-pair prior 0=fresh,1=worn; disables legacy friction scaling, diagnostic only.')
parser.add_argument('--require-analytic-bank', action='store_true', help='Historical analytic lean acceptance only; not operator tracking default.')
parser.add_argument('--pitch-rate-feedback', type=float, default=None,
                    help='Diagnostic gain override in [-0.2,0.2]; not a policy certificate.')
parser.add_argument('--diagnostic-total-residual-budget', type=float, nargs=6, default=None,
                    help='At vx2.4 only: cap complete actor delta from frozen expert blend per action; not a certificate.')
parser.add_argument('--disable-high-speed-feedback', action='store_true',
                    help='Diagnostic removal of high-speed wheel velocity feedback; never a certificate.')
parser.add_argument('--diagnostic-disable-outer-feedback', action='store_true',
                    help='Remove ALL post-actor policy feedback; diagnostic only, not final E2E certification.')
parser.add_argument('--collect-wheel-teacher', action='store_true',
                    help='Save paired policy observations and delivered wheel actions every 8 steps, DR0 only.')
parser.add_argument('--base-wheel-clip', type=float, default=None,
                    help='Diagnostic base MLP wheel clipping before residual composition; never a certificate.')
parser.add_argument('--disable-stage-head', '--stage-head-index', type=int, default=None,
                    help='Diagnostic scaling of one stage head, without changing its parameters.')
parser.add_argument('--stage-head-output-scale', type=float, default=0.0,
                    help='Scale selected diagnostic head in [0,1]; default zero is full ablation.')
parser.add_argument('--radial-spring-force', type=float, default=None, help='Diagnostic virtual-leg extension force in N.')
parser.add_argument('--radial-spring-dr-fraction', type=float, default=0., help='Diagnostic reset-sampled independent left/right force variation.')
parser.add_argument('--gas-spring-torque', type=float, default=None,
                    help='Diagnostic passive spring magnitude in [0,10.775]Nm; not a policy certificate.')
parser.add_argument('--firmware-spring-equivalent', action='store_true',
                    help='Use the read-only RMUC firmware leg spring compensation law as an explicit diagnostic actuator mode.')
parser.add_argument('--record-policy-trace', action='store_true', help='Read-only full per-step action/input/post-state trace for branch diagnosis.')
parser.add_argument('--failure-traces', action='store_true',
                    help='Record up to 16 unsafe episodes with pre-reset state/action histories and DR parameters.')
parser.add_argument('--diagnostic-wheel-velocity-fraction',type=float,default=1.,
                    help='Explicit wheel-only migration fraction0..1; requires velocity profile.')
parser.add_argument('--diagnostic-wheel-velocity-profile',type=Path,default=None,
                    help='Explicit3Nm wheel PI controller proposal; diagnostic only, no hardware certificate.')
parser.add_argument('--diagnostic-wheel-velocity-equivalence', action='store_true',
                    help='Re-express old wheel torque-minus-D as a P velocity loop; diagnostic only, not3Nm migration.')
parser.add_argument('--record-world-trajectory', action='store_true',
                    help='Read-only pre-step/pre-reset world trace for original vx+yaw; no command changes.')
parser.add_argument('--audit-turn-dynamics', action='store_true',
                    help='Read-only heading lateral velocity, mass-weighted COM and per-wheel contact diagnostics.')
parser.add_argument('--audit-reward-contract', action='store_true',
                    help='Compare active/rehearsal rewards at the final identical physical state, without stepping.')
parser.add_argument('--audit-training-env-config', type=Path, default=None,
                    help='Read reward-only bank/yaw/clip overrides from the training params/env.yaml for an audit.')
parser.add_argument('--audit-critic', action='store_true',
                    help='Diagnostic stochastic-policy rollout and critic/Monte Carlo comparison; requires training reward config and DR=0.')
parser.add_argument('--critic-gamma', type=float, default=.99)
parser.add_argument('--action-bias-sweep', type=Path, default=None,
                    help='Diagnostic JSON list of {name, action:[six bounded offsets]}; never a policy certificate.')
parser.add_argument('--real2sim-wheel-profile', type=Path, default=None,
                    help='Explicit reported-fit wheel/compensation overlay; hypotheses only, no deployment certificate.')
parser.add_argument('--real2sim-end-to-end-delay', action='store_true',
                    help='Profile wheel transport owns total observed-to-execution delay; disables old policy delay buffers.')
parser.add_argument('--arena-ground-progress', type=float, default=None,
                    help='Interpolate actual legacy wheel-ground pair toward randomized PVC prior; 0..1.')
parser.add_argument('--base-com-dr-strength', type=float, default=None,
                    help='Root COM engineering prior: strength1 means FLU +/-10/10/5mm.')
parser.add_argument('--action-bias-start-seconds', type=float, default=0.)
parser.add_argument('--action-bias-ramp-seconds', type=float, default=0.)
parser.add_argument('--diagnostic-hold-legs-after-seconds', type=float, default=None)
parser.add_argument('--audit-leg-actuator', action='store_true')
parser.add_argument('--diagnostic-leg-velocity-limit', type=float, default=None)
parser.add_argument('--diagnostic-leg-armature', type=float, default=None)
parser.add_argument('--diagnostic-leg-profile', type=Path, default=None)
parser.add_argument('--diagnostic-leg-dr', action='store_true')
parser.add_argument('--position-authority-consumer', action='store_true')
parser.add_argument('--diagnostic-leg-target-offsets', action='store_true',
                    help='Read four joint-side target offsets from bias cases; diagnostic only.')
parser.add_argument('--diagnostic-physics-dt', type=float, choices=(.001, .002), default=None,
                    help='Separated-wheel diagnostic only; preserves 8ms policy and 2ms leg PD.')
parser.add_argument('--diagnostic-turn-entry', action='store_true',
                    help='Diagnostic: hold target forward speed with zero yaw for1s, ramp yaw during next1s.')
parser.add_argument('--diagnostic-turn-entry-hold-s', type=float, default=1.)
parser.add_argument('--diagnostic-turn-entry-ramp-s', type=float, default=1.)
parser.add_argument('--distill-dataset-dir', type=Path, default=None, help='Stream deterministic raw teacher pairs to CPU shards; no dynamics changes.')
parser.add_argument('--capture-startup-steps', type=int, default=0,
                    help='Save bounded pre-action25D observations and raw actions; diagnostic data only.')
parser.add_argument('--capture-observation-stride', type=int, default=1,
                    help='Spacing between captured policy steps; >1 includes later states within the existing500sample cap.')
parser.add_argument('--diagnostic-tgs-external-forces-every-iteration', action='store_true')
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.capture_observation_stride < 1:
    parser.error('Capture observation stride must be positive')
if not 0 <= args.capture_startup_steps <= 500:
    parser.error('Startup capture must be within0..500control steps')
if any(not 0 <= t <= 4 for t in (args.diagnostic_turn_entry_hold_s, args.diagnostic_turn_entry_ramp_s)):
    parser.error('Turn entry durations must be finite within0..4s')
if not args.diagnostic_turn_entry and (args.diagnostic_turn_entry_hold_s != 1. or args.diagnostic_turn_entry_ramp_s != 1.):
    parser.error('Turn entry durations require --diagnostic-turn-entry')
if args.diagnostic_physics_dt is not None and args.diagnostic_separated_wheel_profile is None:
    parser.error('Diagnostic physics timestep requires an explicit separated wheel profile')
separated_wheel_profile = None
if args.diagnostic_separated_wheel_profile is not None:
    from rmuc_separated_wheel_profile import load_profile, validate_options
    try:
        validate_options(args)
        separated_wheel_profile = load_profile(args.diagnostic_separated_wheel_profile)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
if args.independent_wheel_dr_streams and (args.dr_ablation!='damping_torque_only' or args.dr_strength<=0):
    parser.error('Independent wheel DR streams require positive strength and damping_torque_only')
if args.wheel_dr_reset_budget is not None and (
    not args.independent_wheel_dr_streams or args.wheel_dr_reset_budget < args.steps + 2
):
    parser.error('Explicit wheel DR reset budget requires independent streams and must be >= steps+2')
if args.diagnostic_joint_dr_profile and (args.dr_ablation != 'none' or args.independent_wheel_dr_streams):
    parser.error('Joint DR profile cannot be combined with isolated DR ablations/streams')
if args.diagnostic_hold_legs_after_seconds is not None:
    if not math.isfinite(args.diagnostic_hold_legs_after_seconds) or not 0 < args.diagnostic_hold_legs_after_seconds <= 5:
        parser.error('Leg-hold diagnostic time must be within (0,5] seconds')
    if not args.diagnostic_disable_outer_feedback:
        parser.error('Leg-hold diagnostic requires disabled task outer feedback')
if args.diagnostic_leg_dr and args.diagnostic_leg_profile is None:
    parser.error('Leg DR requires an explicit leg profile')
if args.diagnostic_leg_profile and (args.diagnostic_leg_velocity_limit is not None or args.diagnostic_leg_armature is not None):
    parser.error('Choose leg profile or individual overrides, not both')
if args.diagnostic_leg_velocity_limit is not None and (not math.isfinite(args.diagnostic_leg_velocity_limit) or not 0 < args.diagnostic_leg_velocity_limit <= 100):
    parser.error('Diagnostic leg solver limit must be within (0,100] rad/s')
if args.diagnostic_leg_armature is not None and (not math.isfinite(args.diagnostic_leg_armature) or not 0 <= args.diagnostic_leg_armature <= .1):
    parser.error('Diagnostic leg armature must be within [0,0.1] kg m^2')
if not math.isfinite(args.diagnostic_hardware_fraction) or not 0<=args.diagnostic_hardware_fraction<=1:
    parser.error('Hardware migration fraction must be within0..1')
if args.diagnostic_hardware_fraction!=1 and args.diagnostic_hardware_profile is None:
    parser.error('Hardware migration fraction requires an explicit profile')
if not math.isfinite(args.diagnostic_wheel_velocity_fraction) or not 0<=args.diagnostic_wheel_velocity_fraction<=1:
    parser.error('Wheel velocity fraction must be finite within0..1')
if args.diagnostic_wheel_velocity_fraction!=1 and args.diagnostic_wheel_velocity_profile is None:
    parser.error('Wheel velocity fraction requires explicit velocity profile')
if args.diagnostic_wheel_velocity_profile is not None and (
        args.diagnostic_wheel_velocity_equivalence or args.dr_strength!=0
        or args.diagnostic_hardware_profile is not None or args.motor_response_time_constants_ms is not None):
    parser.error('Target velocity diagnostic requires DR0 and no competing motor/profile wrapper')
if args.diagnostic_wheel_velocity_equivalence and (args.dr_strength!=0 or args.diagnostic_hardware_profile is not None
        or args.motor_response_time_constants_ms is not None):
    parser.error('Velocity equivalence requires DR0 and the unchanged legacy actuator profile')
if args.motor_response_time_constants_ms is not None:
    low,high=args.motor_response_time_constants_ms
    if not (math.isfinite(low) and math.isfinite(high) and 0 < low <= high):
        parser.error('Motor response time constants must be finite, positive and ordered')
if not 0 <= args.diagnostic_mesh_subdivisions <= 6 or (args.diagnostic_plane_terrain and args.diagnostic_mesh_subdivisions):
    parser.error('Mesh subdivisions must be 0..6 and cannot be combined with the plane diagnostic')
speeds = [float(v) for v in args.speeds.split(',')]
yaws = [float(v) for v in args.yaws.split(',')]
commands_list = [(v, yaw) for v in speeds for yaw in yaws]
if args.command_grid is not None:
    pairs = json.loads(args.command_grid.read_text())
    if (not isinstance(pairs, list) or not pairs or any(
            not isinstance(pair, list) or len(pair) != 2 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in pair) for pair in pairs)):
        parser.error('Command grid must contain finite numeric [vx, yaw] pairs')
    commands_list = [(float(v), float(yaw)) for v, yaw in pairs]
    speeds = [v for v, _ in commands_list]
    yaws = [yaw for _, yaw in commands_list]
if not math.isfinite(args.diagnostic_speed_ramp_seconds) or not 0<=args.diagnostic_speed_ramp_seconds<=4:
    parser.error('Diagnostic ramp duration must be within0..4seconds')
if args.diagnostic_speed_ramp_seconds and any(v<1 or yaw!=0 for v,yaw in commands_list):
    parser.error('Diagnostic ramp requires forward straight final targets >=1m/s')
bias_cases = [{'name': 'unmodified', 'action': [0.] * 6}]
if any(not math.isfinite(x) or not 0 <= x <= 5 for x in
       (args.action_bias_start_seconds, args.action_bias_ramp_seconds)):
    parser.error('Diagnostic bias start/ramp times must be finite and within [0,5] seconds')
if (args.action_bias_start_seconds or args.action_bias_ramp_seconds) and not args.action_bias_sweep:
    parser.error('Diagnostic bias timing requires an action bias sweep')
if args.action_bias_sweep:
    bias_cases = json.loads(args.action_bias_sweep.read_text())
    if not bias_cases or any(len(case['action']) != 6 or
                            any(not -0.3 <= float(v) <= 0.3 for v in case['action'])
                            for case in bias_cases):
        parser.error('Diagnostic action offsets must be six values within [-0.3,0.3]')
bias_labels = [case for command in commands_list for case in bias_cases]
if args.diagnostic_leg_target_offsets:
    if not args.action_bias_sweep or not args.diagnostic_disable_outer_feedback:
        parser.error('Leg target offsets require a bias sweep and no outer feedback')
    for case in bias_cases:
        values = case.get('leg_target_offset_rad', [])
        if len(values) != 4 or any(not math.isfinite(float(v)) or abs(float(v)) > .15 for v in values):
            parser.error('Four finite leg target offsets within +/-0.15 rad required')
        if any(float(v) != 0 for v in case['action']):
            parser.error('Separate leg target diagnostics from raw action bias')
commands_list = [command for command in commands_list for case in bias_cases]
count = len(commands_list) * args.replicas
if not 1 <= count <= 128 or args.replicas < 1:
    parser.error('Grid must use 1–128 environments; larger scenes are not validated')
if not 0 <= args.dr_strength <= 1 or args.steps < 1 or args.min_episodes < 1:
    parser.error('Invalid strength, steps, or episode requirement')
if args.episode_length_seconds is not None and not 12. <= args.episode_length_seconds <= 60.:
    parser.error('Evaluation episode duration must be finite within12..60seconds')
if args.pitch_rate_feedback is not None and not -.2 <= args.pitch_rate_feedback <= .2:
    parser.error('Diagnostic pitch-rate feedback must be within [-0.2,0.2]')
if args.base_wheel_clip is not None and not 0.5 <= args.base_wheel_clip <= 1.0:
    parser.error('Diagnostic base wheel clip must be in [0.5,1.0]')
if args.diagnostic_total_residual_budget is not None:
    if any(not 0 < v <= .3 for v in args.diagnostic_total_residual_budget) or any(abs(v-2.4)>1e-6 for v in speeds):
        parser.error('Residual budget diagnostic requires six finite bounds in (0,.3] and vx2.4')
if not 0 <= args.stage_head_output_scale <= 1 or (
        args.stage_head_output_scale != 0 and args.disable_stage_head is None):
    parser.error('Head scale must be in [0,1] and requires a stage head index')
if args.gas_spring_torque is not None and not 0 <= args.gas_spring_torque <= 10.775:
    parser.error('Gas spring magnitude must be in [0,10.775]Nm')
if args.radial_spring_force is not None and not 0 <= args.radial_spring_force < float('inf'):
    parser.error('Radial force must be finite and nonnegative')
if not 0 <= args.radial_spring_dr_fraction <= 1 or (args.radial_spring_dr_fraction and args.radial_spring_force is None):
    parser.error('Radial DR requires a force and fraction in [0,1]')
if args.output.exists():
    parser.error(f'Refusing to overwrite {args.output}')
if args.audit_critic and (not args.audit_training_env_config or args.dr_strength != 0
                          or not 0 < args.critic_gamma < 1):
    parser.error('Critic audit requires training reward config, DR=0 and gamma in (0,1)')
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
import agent_tasks  # noqa: F401,E402
import agent_world  # noqa: F401,E402
from agent_tasks.direct.rmuc_wheel_leg.bank_geometry import BANK_CONTRACT, bank_reference

sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_rmuc_export import restore_actor  # noqa: E402


def main() -> int:
    cfg = parse_env_cfg('Robotics-Wheelbipe-V14-RMUC-Command-v0',
                        device=args.device, num_envs=count, use_fabric=True)
    cfg.seed = args.seed
    leg_scale_profile_record = None
    stored_leg_profile = (torch.load(args.checkpoint, map_location='cpu', weights_only=False).get('infos') or {}).get('rmuc_action_leg_scale_profile')
    if args.command_leg_scale_profile is not None or stored_leg_profile is not None:
        from agent_tasks.direct.rmuc_wheel_leg.command_contract import validate_leg_scale_profile
        profile = (json.loads(args.command_leg_scale_profile.read_text())
                   if args.command_leg_scale_profile is not None else stored_leg_profile)
        if stored_leg_profile is not None and profile != stored_leg_profile:
            raise ValueError('Explicit leg-scale profile differs from checkpoint action contract')
        speeds, scales = validate_leg_scale_profile(profile)
        if max(abs(v) for v, _ in commands_list) > speeds[-1]:
            raise ValueError('Command grid exceeds explicit leg-scale profile')
        cfg.command_leg_scale_profile = profile
        leg_scale_profile_record = dict(profile=profile,checkpoint_bound=stored_leg_profile is not None,
            file_sha256=hashlib.sha256(args.command_leg_scale_profile.read_bytes()).hexdigest() if args.command_leg_scale_profile is not None else None)
    applied_velocity_profile = None
    if args.diagnostic_wheel_velocity_profile is not None:
        import yaml
        from rmuc_velocity_profile import apply_velocity_profile
        applied_velocity_profile = apply_velocity_profile(cfg,yaml.safe_load(args.diagnostic_wheel_velocity_profile.read_text()),args.diagnostic_wheel_velocity_fraction)
    applied_hardware_profile = None
    if args.diagnostic_hardware_profile is not None:
        import yaml
        applied_hardware_profile = apply_hardware_profile(cfg, yaml.safe_load(args.diagnostic_hardware_profile.read_text()),args.diagnostic_hardware_fraction)
    if args.diagnostic_mesh_subdivisions:
        from isaaclab.terrains import TerrainImporter

        class SubdividedDiagnosticMesh(TerrainImporter):
            def import_mesh(self, name, mesh):
                for _ in range(args.diagnostic_mesh_subdivisions):
                    mesh = mesh.subdivide()
                super().import_mesh(name, mesh)

        cfg.terrain.class_type = SubdividedDiagnosticMesh
    if args.diagnostic_plane_terrain:
        # The stock plane spawner references a remote visual USD. This
        # headless diagnostic needs only a local analytic collision plane.
        from isaaclab.terrains import TerrainImporter
        import isaaclab.sim as sim_utils
        from pxr import UsdGeom, UsdPhysics

        class LocalDiagnosticPlane(TerrainImporter):
            def import_ground_plane(self, name, size=(2.0e6, 2.0e6)):
                path = self.cfg.prim_path + '/' + name
                if path in self.terrain_prim_paths:
                    raise ValueError(f'Duplicate diagnostic plane: {path}')
                stage = sim_utils.get_current_stage()
                plane = UsdGeom.Plane.Define(stage, path)
                plane.CreateAxisAttr('Z')
                plane.CreateWidthAttr(float(size[0]))
                plane.CreateLengthAttr(float(size[1]))
                UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
                material_path = self.cfg.prim_path + '/physicsMaterial'
                material = self.cfg.physics_material
                material.func(material_path, material)
                sim_utils.bind_physics_material(path, material_path)
                self.terrain_prim_paths.append(path)

        cfg.terrain.class_type = LocalDiagnosticPlane
        cfg.terrain.terrain_type = 'plane'
        cfg.terrain.terrain_generator = None
        cfg.terrain.env_spacing = cfg.scene.env_spacing
    cfg.play = True
    cfg.command_require_analytic_bank = args.require_analytic_bank
    cfg.command_start_level = 46
    cfg.command_start_at_target = True
    cfg.command_stage_hold_steps = 1 << 60
    if args.episode_length_seconds is not None:
        cfg.episode_length_s = args.episode_length_seconds
    cfg.command_rehearsal_enabled = False
    cfg.feedback_contract_by_command = True
    cfg.wheel_yaw_feedback_low_speed_action_clip = .3
    cfg.domain_randomization_enabled = args.dr_strength > 0
    cfg.domain_randomization_apply_in_play = True
    cfg.leg_transport_delay_s = args.leg_transport_delay_ms/1000.
    cfg.leg_transport_delay_probability = args.leg_transport_delay_probability
    cfg.gravity_direction_dr_strength = args.gravity_direction_dr_strength
    cfg.gravity_direction_dr_probability = args.gravity_direction_dr_probability
    cfg.domain_randomization_strength_override = args.dr_strength
    cfg.domain_randomization_command_consistent_noise = True
    if args.motor_response_time_constants_ms is not None:
        cfg.motor_response_enabled = True
        cfg.motor_response_time_constant_min_s = args.motor_response_time_constants_ms[0]/1000.
        cfg.motor_response_time_constant_max_s = args.motor_response_time_constants_ms[1]/1000.
    if args.radial_spring_force is not None:
        cfg.gas_spring_radial_force_n = args.radial_spring_force
        cfg.gas_spring_radial_dr_fraction = args.radial_spring_dr_fraction
    if args.gas_spring_torque is not None:
        cfg.gas_spring_assist_torque_nm = args.gas_spring_torque
    if args.firmware_spring_equivalent:
        cfg.gas_spring_model = 'firmware_motor_equivalent'
        cfg.firmware_spring_enabled = True
    if args.dr_ablation in ('physics_only', 'damping_torque_only'):
        cfg.domain_randomization_sensor_model = {**cfg.domain_randomization_sensor_model, 'enabled': False}
        cfg.domain_randomization_max_observation_noise_std = 0.0
        cfg.domain_randomization_max_observation_delay_steps = 0
        cfg.domain_randomization_max_observation_delay_probability = 0.0
        if args.dr_ablation == 'damping_torque_only':
            cfg.domain_randomization_max_base_mass_fraction = 0.0
            cfg.domain_randomization_max_base_inertia_fraction = 0.0
            cfg.domain_randomization_max_friction_fraction = 0.0
            cfg.motor_response_enabled = False
            cfg.gas_spring_torque_common_dr_fraction = 0.0
            cfg.gas_spring_torque_independent_dr_fraction = 0.0
            cfg.gas_spring_radial_dr_fraction = 0.0
            cfg.leg_actuator_dr_enabled = False
    elif args.dr_ablation == 'observation_only':
        cfg.motor_response_enabled = False
        cfg.domain_randomization_max_base_mass_fraction = 0.0
        cfg.domain_randomization_max_base_inertia_fraction = 0.0
        cfg.domain_randomization_max_friction_fraction = 0.0
        cfg.domain_randomization_max_wheel_damping_fraction = 0.0
        cfg.domain_randomization_max_wheel_torque_gain_fraction = 0.0
    audit_reward_overrides = {}
    if args.audit_training_env_config:
        import yaml
        values = yaml.load(args.audit_training_env_config.read_text(), Loader=yaml.BaseLoader)
        for name, cast in (('command_active_bank_reward_scale', float),
                           ('command_turn_yaw_tracking_start_level', int), ('reward_clip_max', float)):
            value = cast(values[name])
            setattr(cfg, name, value)
            audit_reward_overrides[name] = value
        # These switches are explicit in physical-adaptation training. Omitting
        # them makes the audit evaluate a different yaw reward by default.
        for name in ('command_low_speed_turn_yaw_tracking_enabled',
                     'command_reward_contract_by_command', 'command_require_analytic_bank'):
            value=values[name].lower()=='true'
            setattr(cfg,name,value)
            audit_reward_overrides[name]=value
        if args.audit_critic:
            cfg.command_reward_contract_by_command = values['command_reward_contract_by_command'].lower() == 'true'
    if args.pitch_rate_feedback is not None:
        cfg.wheel_pitch_rate_feedback_action_per_rad_s = args.pitch_rate_feedback
    if args.disable_high_speed_feedback:
        cfg.wheel_speed_feedback_high_speed_action_per_mps = 0.0
    if args.diagnostic_disable_outer_feedback:
        cfg.policy_outer_feedback_enabled = False
    if args.collect_wheel_teacher and (args.dr_strength != 0 or args.diagnostic_disable_outer_feedback
                                      or args.action_bias_sweep is not None):
        raise ValueError('Wheel teacher collection requires DR0, enabled outer feedback and no action bias')
    applied_separated_wheel_profile = None
    if separated_wheel_profile is not None:
        from rmuc_separated_wheel_profile import apply_separated_wheel_profile
        applied_separated_wheel_profile = apply_separated_wheel_profile(
            cfg, separated_wheel_profile, physics_dt=args.diagnostic_physics_dt or .001)
    if args.diagnostic_joint_dr_profile:
        import yaml
        from rmuc_joint_dr_profile import apply_joint_dr_profile
        apply_joint_dr_profile(cfg, yaml.safe_load(args.diagnostic_joint_dr_profile.read_text()))
    if args.real2sim_wheel_profile:
        if separated_wheel_profile is None or not args.diagnostic_disable_outer_feedback:
            raise ValueError('real2sim overlay requires separated wheel profile and disabled outer feedback')
        cfg.real2sim_wheel_profile = str(args.real2sim_wheel_profile.resolve())
    if args.real2sim_end_to_end_delay:
        if not args.real2sim_wheel_profile:
            raise ValueError('End-to-end timing requires real2sim wheel profile')
        cfg.real2sim_end_to_end_delay = True
    from rmuc_training_contract import checkpoint_observation_contract, observation_contract
    checkpoint_observation = checkpoint_observation_contract(
        torch.load(args.checkpoint, map_location='cpu', weights_only=False))
    cfg.command_yaw_observation_error_clip = checkpoint_observation['command_yaw_observation_error_clip']
    if observation_contract(cfg) != checkpoint_observation:
        raise ValueError('Evaluator observation contract differs from checkpoint')
    # Snapshot before rollout: later source edits cannot silently relabel the run.
    source_root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), source_root / 'configs/rmuc_wheel_leg.yaml',
        source_root / 'scripts/rsl_rl/rmuc_training_contract.py',
        source_root / 'scripts/rsl_rl/rmuc_command_adapter.py',
        source_root / 'scripts/rsl_rl/rmuc_state_router.py',
        source_root / 'scripts/rsl_rl/rmuc_position_authority.py',
        source_root / 'scripts/rsl_rl/rmuc_local_authority.py',
        source_root / 'scripts/rmuc_eval_layout.py',
        source_root / 'scripts/rmuc_eval_commands.py',
        source_root / 'scripts/rmuc_turn_diagnostics.py',
        source_root / 'scripts/verify_rmuc_export.py']
    if args.command_grid is not None:
        source_paths.append(args.command_grid)
    if args.diagnostic_hardware_profile is not None:
        source_paths.extend([args.diagnostic_hardware_profile, source_root/'scripts/rmuc_hardware_profile.py'])
    if args.independent_wheel_dr_streams:
        source_paths.append(source_root/'scripts/rmuc_eval_dr_streams.py')
    if args.diagnostic_joint_dr_profile:
        source_paths.extend([args.diagnostic_joint_dr_profile, source_root/'scripts/rmuc_joint_dr_profile.py'])
    task_root = source_root / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg'
    if args.arena_ground_progress is not None:
        source_paths.extend([task_root/'arena_ground.py', Path(cfg.arena_ground_profile_path)])
    if args.base_com_dr_strength is not None:
        source_paths.append(task_root/'com_randomization.py')
    if args.real2sim_wheel_profile:
        source_paths.extend([args.real2sim_wheel_profile, task_root/'real2sim_wheel.py'])
    source_paths += [task_root / name for name in ('env.py', 'env_cfg.py', 'flat_terrain.py', 'observation_noise.py', 'motor_response.py', 'gas_spring.py', 'command_contract.py', 'bank_geometry.py', 'wheel_velocity_control.py')]
    source_paths += [Path(cfg.robot_cfg.spawn.usd_path)]
    if args.diagnostic_separated_wheel_profile is not None:
        source_paths.extend([args.diagnostic_separated_wheel_profile,
                             source_root/'scripts/rmuc_separated_wheel_profile.py',
                             task_root/'wheel_passive.py'])
    if args.diagnostic_wheel_velocity_profile is not None:
        source_paths.extend([args.diagnostic_wheel_velocity_profile,source_root/'scripts/rmuc_velocity_profile.py'])
    source_hashes = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    def primitive(value):
        return value is None or isinstance(value, (bool, int, float, str)) or (
            isinstance(value, (tuple, list)) and all(primitive(item) for item in value)) or (
            isinstance(value, dict) and all(isinstance(key, str) and primitive(item) for key, item in value.items()))
    runtime_contract = {name: getattr(cfg, name) for name in dir(cfg)
        if name.startswith(('command_', 'domain_', 'wheel_', 'motor_response_', 'gas_spring_', 'observation_', 'asset_frame_', 'torque_limit_'))
        and primitive(getattr(cfg, name))}
    runtime_contract.update(sim_dt=cfg.sim.dt, decimation=cfg.decimation, run_mode=cfg.run_mode,
        action_clip=cfg.action_clip, episode_length_s=cfg.episode_length_s,
        feedback_contract_by_command=cfg.feedback_contract_by_command)
    if args.diagnostic_leg_profile is not None:
        import yaml
        from rmuc_leg_actuator_profile import apply_leg_profile
        apply_leg_profile(cfg, yaml.safe_load(args.diagnostic_leg_profile.read_text()), args.diagnostic_leg_dr)
    if args.diagnostic_leg_velocity_limit is not None:
        cfg.robot_cfg.actuators['leg_motors'].velocity_limit_sim = args.diagnostic_leg_velocity_limit
    if args.diagnostic_leg_armature is not None:
        cfg.robot_cfg.actuators['leg_motors'].armature = args.diagnostic_leg_armature
    if args.diagnostic_tgs_external_forces_every_iteration:
        if cfg.sim.physics is None:
            from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
            cfg.sim.physics = PhysxCfg()
        cfg.sim.physics.enable_external_forces_every_iteration = True
    if args.position_authority_consumer:
        if not args.diagnostic_disable_outer_feedback:
            raise ValueError('Position authority consumer requires disabled task feedback')
        cfg.position_authority_enabled = True
        from rmuc_position_authority import checkpoint_authority_speed_plateau_end
        cfg.position_authority_speed_plateau_end = checkpoint_authority_speed_plateau_end(
            torch.load(args.checkpoint, map_location='cpu', weights_only=False)['actor_state_dict'])
        runtime_contract['position_authority_speed_plateau_end'] = cfg.position_authority_speed_plateau_end
    if args.arena_ground_severity is not None:
        if not 0 <= args.arena_ground_severity <= 1:
            raise ValueError('Arena severity must be finite and in[0,1]')
        cfg.arena_ground_severity=args.arena_ground_severity
        cfg.friction_randomization_enabled=False
        cfg.domain_randomization_max_friction_fraction=0.
    if args.arena_ground_progress is not None:
        if args.arena_ground_severity is not None or not math.isfinite(args.arena_ground_progress) or not 0 <= args.arena_ground_progress <= 1:
            raise ValueError('Arena progress requires0..1 and no fixed arena severity')
        cfg.arena_ground_migration_progress=args.arena_ground_progress
        cfg.arena_ground_randomization_enabled=True
        cfg.arena_ground_severity_range=(0.,1.)
        cfg.friction_randomization_enabled=False
        cfg.domain_randomization_max_friction_fraction=0.
    if args.base_com_dr_strength is not None:
        if not math.isfinite(args.base_com_dr_strength) or not 0 <= args.base_com_dr_strength <= 1:
            raise ValueError('COM strength requires0..1')
        cfg.base_com_randomization_enabled=True
        cfg.base_com_randomization_strength=args.base_com_dr_strength
    if args.diagnostic_wheel_peak_nm is not None:
        peak=args.diagnostic_wheel_peak_nm
        if not math.isfinite(peak) or not 0 < peak <= 5.5:
            raise ValueError('Diagnostic wheel peak must be in (0,5.5]')
        cfg.diagnostic_wheel_peak_nm=peak
        cfg.robot_cfg.actuators['wheel_motors'].effort_limit=peak
        cfg.robot_cfg.actuators['wheel_motors'].effort_limit_sim=peak
    env = RslRlVecEnvWrapper(gym.make('Robotics-Wheelbipe-V14-RMUC-Command-v0', cfg=cfg),
                            clip_actions=None if args.position_authority_consumer else 1.0)
    u = env.unwrapped
    device = u.device
    # Some explicit diagnostics and the environment's total-delay adapter
    # finalize config during construction. Report the actual resulting config,
    # not the earlier pre-construction snapshot (which could show stale DR).
    runtime_contract.update({name: getattr(u.cfg, name) for name in dir(u.cfg)
        if name.startswith(('command_', 'domain_', 'wheel_', 'motor_response_', 'gas_spring_',
                            'observation_', 'asset_frame_', 'torque_limit_', 'arena_ground_',
                            'base_com_', 'real2sim_')) and primitive(getattr(u.cfg, name))})
    if args.diagnostic_physics_dt is not None:
        actual_dt = u.sim.get_physics_dt()
        if abs(actual_dt - args.diagnostic_physics_dt) > 1e-9 or abs(u.step_dt - .008) > 1e-9:
            raise RuntimeError(f'Runtime timing differs from requested diagnostic: physics={actual_dt}, policy={u.step_dt}')
    if applied_separated_wheel_profile is not None:
        from rmuc_separated_wheel_profile import readback
        applied_separated_wheel_profile['runtime_readback'] = readback(u.cfg)
        applied_separated_wheel_profile['runtime_motor_actuator_pd_damping'] = u._wheel_actuator.damping.detach().cpu().tolist()
        applied_separated_wheel_profile['runtime_feedback_applied_after_motor_clip'] = (
            u.cfg.wheel_feedback_damping_mode == 'postclip_loss_v1')
        applied_separated_wheel_profile['runtime_motor_peak_nm'] = u._wheel_actuator.effort_limit.detach().cpu().tolist()
        applied_separated_wheel_profile['runtime_separated_passive_enabled'] = u._separated_wheel_passive
        applied_separated_wheel_profile['runtime_leg_pd_interval_steps'] = u._leg_pd_sample_hold.interval_steps
    leg_audit = None
    if args.audit_leg_actuator:
        leg_actuator = u.robot.actuators['leg_motors']
        leg_joint_ids = [u.robot.joint_names.index(name) for name in leg_actuator.joint_names]
        leg_policy_slots = [u._control_dof_ids.index(index) for index in leg_joint_ids]
        original_leg_compute = leg_actuator.compute
        leg_audit = {'physics_calls': 0,
            'max_abs_target_position_rad': torch.zeros(4, device=device),
            'max_abs_speed_rad_s': torch.zeros(4, device=device),
            'steady_max_abs_speed_rad_s': torch.zeros(4, device=device),
            'samples_above_17_rad_s': torch.zeros(4, device=device),
            'solver_velocity_limit_rad_s': u.robot.data.joint_vel_limits.torch[:, leg_joint_ids].cpu().tolist(),
            'armature_kg_m2': u.robot.data.joint_armature.torch[:, leg_joint_ids].cpu().tolist(),
            'policy_slots_in_joint_order': leg_policy_slots,
            'steady_physics_samples_per_env': torch.zeros(u.num_envs, device=device)}
        leg_sums = {name: torch.zeros(u.num_envs, 4, device=device) for name in (
            'target_position_rad', 'actual_position_rad', 'absolute_position_error_rad',
            'proportional_effort_nm', 'computed_effort_nm', 'applied_effort_nm',
            'torque_saturation_fraction', 'action_saturation_fraction',
            'actual_velocity_rad_s', 'target_velocity_rad_s', 'feedforward_effort_nm',
            'position_difference_velocity_rad_s', 'velocity_consistency_abs_error_rad_s')}
        previous_leg_position = None
        previous_leg_age = None
        def audited_leg_compute(control_action, joint_pos, joint_vel):
            nonlocal previous_leg_position, previous_leg_age
            # The actuator clears joint_positions in place during compute.
            target = control_action.joint_positions.clone()
            leg_audit['max_abs_target_position_rad'] = torch.maximum(
                leg_audit['max_abs_target_position_rad'], target.abs().amax(0))
            target_velocity = control_action.joint_velocities.clone()
            feedforward = control_action.joint_efforts.clone()
            difference_velocity = torch.zeros_like(joint_vel)
            if previous_leg_position is not None:
                continuous = u._curriculum_step_counter >= previous_leg_age
                difference_velocity = torch.where(continuous[:, None],
                    (joint_pos-previous_leg_position)/cfg.sim.dt, 0.)
            previous_leg_position = joint_pos.clone()
            previous_leg_age = u._curriculum_step_counter.clone()
            speed = joint_vel.abs()
            leg_audit['physics_calls'] += 1
            leg_audit['max_abs_speed_rad_s'] = torch.maximum(leg_audit['max_abs_speed_rad_s'], speed.amax(0))
            steady = u._curriculum_step_counter >= 1200
            leg_audit['steady_max_abs_speed_rad_s'] = torch.maximum(leg_audit['steady_max_abs_speed_rad_s'],
                torch.where(steady[:, None], speed, 0.).amax(0))
            leg_audit['samples_above_17_rad_s'] += (speed > 17.).sum(0)
            result = original_leg_compute(control_action, joint_pos, joint_vel)
            leg_audit['steady_physics_samples_per_env'] += steady
            values = {
                'target_position_rad': target, 'actual_position_rad': joint_pos,
                'absolute_position_error_rad': (target-joint_pos).abs(),
                'proportional_effort_nm': leg_actuator.stiffness*(target-joint_pos),
                'computed_effort_nm': leg_actuator.computed_effort,
                'applied_effort_nm': leg_actuator.applied_effort,
                'actual_velocity_rad_s': joint_vel,
                'target_velocity_rad_s': target_velocity,
                'feedforward_effort_nm': feedforward,
                'position_difference_velocity_rad_s': difference_velocity,
                'velocity_consistency_abs_error_rad_s': (joint_vel-difference_velocity).abs(),
                'torque_saturation_fraction': (leg_actuator.applied_effort.abs() >= .99*leg_actuator.effort_limit).float(),
                'action_saturation_fraction': (u._policy_actions[:, leg_policy_slots].abs() >= .999).float()}
            for name, value in values.items():
                leg_sums[name] += torch.where(steady[:, None], value, 0.)
            return result
        leg_actuator.compute = audited_leg_compute
    def material_readback():
        import warp as wp
        properties = wp.to_torch(u.robot.root_view.get_material_properties()).detach().cpu()
        expected = u._domain_material_properties
        return {
            'scope': 'Robot collision shape coefficients, not combined wheel-ground friction',
            'shape': list(properties.shape),
            'static_friction_range': [float(properties[..., 0].min()), float(properties[..., 0].max())],
            'dynamic_friction_range': [float(properties[..., 1].min()), float(properties[..., 1].max())],
            'dynamic_exceeds_static': bool((properties[..., 1] > properties[..., 0]).any()),
            'max_abs_error_vs_dr_buffer': None if expected is None else float((properties - expected.cpu()).abs().max()),
            'friction_scale_per_env': u._domain_friction_scale.tolist(),
        }
    from pxr import Usd
    from isaaclab.sim.utils import get_current_stage
    terrain_prim_audit = []
    for prim in Usd.PrimRange(get_current_stage().GetPrimAtPath(cfg.terrain.prim_path)):
        terrain_prim_audit.append({
            'path': str(prim.GetPath()), 'type': prim.GetTypeName(),
            'schemas': list(prim.GetAppliedSchemas()),
            'physics_attributes': {a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                                   if a.GetName().startswith(('physics:', 'physxCollision:', 'physxMaterial:'))},
            'material_bindings': {r.GetName(): [str(t) for t in r.GetTargets()] for r in prim.GetRelationships()
                                  if r.GetName().startswith('material:binding')},
            'face_count': len(prim.GetAttribute('faceVertexCounts').Get() or []) if prim.GetTypeName() == 'Mesh' else None,
        })
    spring_wrench_audit = None
    if args.gas_spring_torque is not None and args.gas_spring_torque > 0:
        from isaaclab.utils.math import quat_apply
        spring_wrench_audit = {'substeps': 0,
            'max_world_torque_error_nm': torch.zeros((), device=device),
            'max_net_world_torque_nm': torch.zeros((), device=device)}
        original_apply_action = u._apply_action
        def apply_action_with_spring_audit():
            original_apply_action()
            ids = u._gas_spring_body_ids
            composer = u.robot.instantaneous_wrench_composer
            composer.compose_to_body_frame()
            body_torque = composer.out_torque_b.torch[:, ids]
            quaternion = u.robot.data.body_quat_w.torch[:, ids]
            actual_world = quat_apply(quaternion.reshape(-1, 4), body_torque.reshape(-1, 3)).reshape_as(body_torque)
            from agent_tasks.direct.rmuc_wheel_leg.gas_spring import spring_body_torques
            expected = spring_body_torques(
                quaternion[:, 0], args.gas_spring_torque,
                quaternion_order=getattr(u.cfg, 'gas_spring_quaternion_order', 'wxyz'))
            spring_wrench_audit['substeps'] += 1
            spring_wrench_audit['max_world_torque_error_nm'].copy_(torch.maximum(
                spring_wrench_audit['max_world_torque_error_nm'], (actual_world - expected).abs().max()))
            spring_wrench_audit['max_net_world_torque_nm'].copy_(torch.maximum(
                spring_wrench_audit['max_net_world_torque_nm'], actual_world.sum(1).abs().max()))
        u._apply_action = apply_action_with_spring_audit
    torch.manual_seed(args.seed)
    actor = restore_actor(args.checkpoint).to(device).eval()
    if hasattr(actor, 'rmuc_position_authority') != args.position_authority_consumer:
        raise ValueError('Position authority checkpoint and consumer must be enabled together')
    if args.position_authority_consumer:
        if float(getattr(actor.rmuc_position_authority, 'speed_plateau_end', 1.0)) != cfg.position_authority_speed_plateau_end:
            raise ValueError('Authority actor and consumer speed support disagree')
    distill_teacher = actor if args.distilled_student and args.distill_dataset_dir else None
    student_sha256 = None
    if args.distilled_student:
        from rmuc_student_runtime import StudentActor
        student_payload = torch.load(args.distilled_student, map_location='cpu', weights_only=False)
        if student_payload['teacher_sha256'] != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
            raise ValueError('Student teacher contract hash mismatch')
        actor = StudentActor(student_payload).to(device).eval()
        student_sha256 = hashlib.sha256(args.distilled_student.read_bytes()).hexdigest()
    if args.diagnostic_total_residual_budget is not None:
        if args.audit_critic:
            raise ValueError('Residual budget diagnostic requires deterministic actions')
        residual_budget = torch.tensor(args.diagnostic_total_residual_budget, device=device)
        def cap_total_residual(module, inputs, output):
            latent = module.get_latent(inputs[0])
            reference = module.mlp(latent)
            if module._rmuc_turn_expert_enabled:
                gate = module.rmuc_turn_adapter.expert_gate(latent).unsqueeze(-1)
                reference = reference + gate * (module.rmuc_turn_expert(latent) - reference)
            # Speed expert support is2.8..3.0 and is exactly zero at this diagnostic's2.4.
            delta = output - reference
            return reference + torch.maximum(torch.minimum(delta, residual_budget), -residual_budget)
        actor.register_forward_hook(cap_total_residual)
    if args.disable_stage_head is not None:
        heads = getattr(actor, 'rmuc_turn_delta_adapters', [])
        if not 0 <= args.disable_stage_head < len(heads):
            raise ValueError('Diagnostic stage head index is not present in this checkpoint')
        heads[args.disable_stage_head].register_forward_hook(
            lambda module, inputs, output: output * args.stage_head_output_scale)
    if args.base_wheel_clip is not None:
        def clip_base_wheels(module, inputs, output):
            bounded = output.clone()
            bounded[..., [2, 5]] = output[..., [2, 5]].clamp(
                -args.base_wheel_clip, args.base_wheel_clip)
            return bounded
        actor.mlp.register_forward_hook(clip_base_wheels)
    bucket_ids = torch.tensor(bucket_assignment(len(commands_list), args.replicas, args.layout_seed), device=device)
    command = torch.tensor([[v, 0., yaw] for v, yaw in commands_list], device=device)[bucket_ids]
    action_bias = torch.tensor([case['action'] for case in bias_labels], device=device)[bucket_ids]
    if args.diagnostic_leg_target_offsets:
        from rmuc_eval_commands import offset_leg_targets, diagnostic_bias_fraction
        leg_offsets = torch.tensor([case['leg_target_offset_rad'] for case in bias_labels], device=device)[bucket_ids]
        original_position_target = u.robot.set_joint_position_target
        def position_target_with_diagnostic_offset(target, joint_ids=None, env_ids=None):
            if list(joint_ids) == list(u._leg_control_ids):
                if env_ids is not None:
                    raise RuntimeError('Diagnostic leg targets expect full environment batch')
                limits = u.robot.data.joint_pos_limits.torch[:, u._leg_control_ids]
                fraction = diagnostic_bias_fraction(u._curriculum_step_counter, u.step_dt,
                    args.action_bias_start_seconds, args.action_bias_ramp_seconds)
                target = offset_leg_targets(target, leg_offsets, fraction, limits,
                                            cfg.leg_joint_limit_margin_rad)
            return original_position_target(target, joint_ids=joint_ids, env_ids=env_ids)
        u.robot.set_joint_position_target = position_target_with_diagnostic_offset
    u.set_manual_command(command)
    sequence_table = None
    if args.command_sequence is not None:
        from rmuc_command_sequence import validate_sequence, sequence_commands
        if args.diagnostic_turn_entry or args.diagnostic_speed_ramp_seconds or not args.record_world_trajectory or not args.record_leg_tracking:
            raise ValueError('Sequence requires world/leg traces and no other command schedule')
        if len(commands_list)!=1:
            raise ValueError('Sequence uses one bucket with replicas, not a fixed multi-command grid')
        sequence_table=validate_sequence(json.loads(args.command_sequence.read_text()),u.step_dt,cfg.episode_length_s)
        original_sequence_observations=u._get_observations
        def observations_with_sequence():
            command.copy_(sequence_commands(sequence_table,u._curriculum_step_counter))
            u.set_manual_command(command)
            return original_sequence_observations()
        u._get_observations=observations_with_sequence
    if args.diagnostic_turn_entry and args.diagnostic_speed_ramp_seconds:
        raise ValueError('Use only one diagnostic command entry schedule')
    if args.diagnostic_turn_entry:
        final_turn_command = command.clone()
        original_turn_observations = u._get_observations
        def observations_with_turn_entry():
            from rmuc_eval_commands import diagnostic_bias_fraction
            fraction = diagnostic_bias_fraction(u._curriculum_step_counter, u.step_dt,
                args.diagnostic_turn_entry_hold_s, args.diagnostic_turn_entry_ramp_s)
            command.copy_(final_turn_command)
            command[:, 2] *= fraction
            u.set_manual_command(command)
            return original_turn_observations()
        u._get_observations = observations_with_turn_entry
    if args.diagnostic_speed_ramp_seconds:
        final_command = command.clone()
        original_get_observations = u._get_observations
        def observations_with_command_ramp():
            command.copy_(ramp_straight_commands(final_command, u._curriculum_step_counter,
                u.step_dt, args.diagnostic_speed_ramp_seconds))
            u.set_manual_command(command)
            return original_get_observations()
        # Changes command BEFORE creating the next observation, including resets.
        # No second observation call: sensor histories advance exactly once.
        u._get_observations = observations_with_command_ramp
    velocity_bridge = None
    velocity_bridge_audit = None
    if args.diagnostic_wheel_velocity_equivalence:
        from agent_tasks.direct.rmuc_wheel_leg.wheel_velocity_control import install_legacy_velocity_equivalence
        velocity_bridge,velocity_bridge_audit = install_legacy_velocity_equivalence(u._wheel_actuator,cfg.sim.dt)
    target_velocity_controller = None
    target_velocity_audit = None
    if applied_velocity_profile is not None:
        from agent_tasks.direct.rmuc_wheel_leg.wheel_velocity_control import install_wheel_velocity_controller
        p=applied_velocity_profile
        reference_scale=p['legacy_action_reference_scale_rad_s']
        target_velocity_controller,target_velocity_audit = install_wheel_velocity_controller(
            u._wheel_actuator,cfg.sim.dt,
            lambda:u._policy_actions[:,[2,5]]*reference_scale*u._wheel_action_sign_tensor,
            lambda:(~u._estop)[:,None],
            p['proportional_gain_nm_per_rad_s'],p['integral_gain_nm_per_rad'],
            p['velocity_filter_tau_s'],p['speed_reference_limit_rad_s'])
    wheel_dr_streams = None
    if args.independent_wheel_dr_streams:
        from rmuc_eval_dr_streams import WheelDRStreams
        wheel_dr_streams=WheelDRStreams(count,args.wheel_dr_reset_budget or args.steps+2,args.seed,args.dr_strength,
            cfg.domain_randomization_max_wheel_damping_fraction,
            cfg.domain_randomization_max_wheel_torque_gain_fraction,device)
        original_randomization=u._apply_progressive_domain_randomization
        def independent_randomization(env_ids):
            original_randomization(env_ids)
            values=wheel_dr_streams.sample(env_ids)
            u._domain_wheel_damping_scale[env_ids]=values[:,0]
            u._domain_wheel_torque_gain[env_ids]=values[:,1]
            u._wheel_actuator.damping[env_ids]=u._wheel_actuator_damping*values[:,0,None]
        u._apply_progressive_domain_randomization=independent_randomization
    obs, _ = env.reset()
    initial_state_audit = {
        'environment_origins': u.scene.env_origins.tolist(),
        'root_positions': u.robot.data.root_pos_w.torch.tolist(),
        'root_quaternions': u.robot.data.root_quat_w.torch.tolist(),
        'policy_observations': obs['policy'].tolist(),
        'wheel_damping_scale': u._domain_wheel_damping_scale.tolist(),
        'wheel_torque_gain': u._domain_wheel_torque_gain.tolist(),
    }
    initial_material_readback = material_readback()
    initial_mass_inertia_readback = {
        'root_body_id': u._root_body_id,
        'mass_kg': u.robot.data.body_mass.torch[:, u._root_body_id].tolist(),
        'inertia_tensor_kg_m2': u.robot.data.body_inertia.torch[:, u._root_body_id, :].tolist(),
        'mass_scale': u._domain_base_mass_scale.tolist(),
        'independent_inertia_shape_scale': u._domain_base_inertia_shape_scale.tolist(),
        'total_inertia_scale': u._domain_base_inertia_scale.tolist(),
        'scope': 'Snapshot after initial reset, before rollout; later resets may resample DR',
    }
    critic = None
    critic_samples = []
    last_terminated = torch.zeros(count, dtype=torch.bool, device=device)
    last_truncated = torch.zeros_like(last_terminated)
    if args.audit_critic:
        from rsl_rl.models import MLPModel
        critic = MLPModel(obs, {'critic': ['policy']}, 'critic', 1,
                          hidden_dims=[256, 128, 64]).to(device).eval()
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        critic.load_state_dict(payload['critic_state_dict'], strict=True)
    window = int(cfg.command_gate_steps)
    warmup = max(window, int(cfg.command_survival_gate_steps))
    if args.diagnostic_speed_ramp_seconds:
        warmup = max(warmup, math.ceil(args.diagnostic_speed_ramp_seconds/u.step_dt)+window)
    error_window = torch.zeros(count, window, device=device)
    yaw_window = torch.zeros_like(error_window)
    window_count = torch.zeros(count, dtype=torch.long, device=device)
    samples = torch.zeros(count, device=device)
    linear_sum = torch.zeros_like(samples)
    yaw_sum = torch.zeros_like(samples)
    signed_yaw_sum = torch.zeros_like(samples)
    signed_speed_sum = torch.zeros_like(samples)
    stable_sum = torch.zeros_like(samples)
    bank_sum = torch.zeros_like(samples)
    torque_sum = torch.zeros_like(samples)
    torque_clip_sum = torch.zeros_like(samples)
    demand_hits=torch.zeros(count,2,device=device)
    demand_run=torch.zeros(count,2,device=device)
    demand_longest=torch.zeros(count,2,device=device)
    demand_peak=torch.zeros(count,2,device=device)
    slip_sum = torch.zeros_like(samples)
    turn_dynamics_sum = None
    turn_extrema = None
    if args.audit_turn_dynamics:
        from rmuc_turn_diagnostics import COLUMNS as base_turn_columns, turn_metrics, world_bank_angle, bank_aligned_with_turn_proxy
        from isaaclab.utils.math import quat_apply
        turn_columns = base_turn_columns + ('world_bank_toward_heading_left_deg', 'physical_inward_lean_deg')
        turn_dynamics_sum = torch.zeros(count,len(turn_columns),device=device)
        from rmuc_turn_extrema import TurnExtrema
        turn_extrema = TurnExtrema(count, device, u.step_dt)
    roll_sum = torch.zeros_like(samples)
    pitch_abs_max = torch.zeros_like(samples)
    raw_action_sum = torch.zeros(count, 6, device=device)
    raw_action_saturated = torch.zeros_like(raw_action_sum)
    executed_action_sum = torch.zeros_like(raw_action_sum)
    joint_position_sum = torch.zeros(count, 4, device=device)
    current_raw_action = torch.zeros_like(raw_action_sum)
    completed = torch.zeros(count, dtype=torch.long, device=device)
    survived = torch.zeros_like(completed)
    root_min = torch.full((count,), float('inf'), device=device)
    root_sum = torch.zeros_like(samples)
    reason_counts = {k: torch.zeros_like(completed) for k in
                     ['nonfinite', 'joint_limit', 'tilt', 'root_high', 'root_low', 'xy_distance', 'linear_speed']}
    linear_limits = []
    for vx, yaw in commands_list:
        matches = [cfg.command_stage_advance_linear_error_mps[i]
                   for i, (v, y) in enumerate(zip(cfg.command_stage_linear_x_mps, cfg.command_stage_yaw_rate_rad_s))
                   if abs(v - abs(vx)) < 1e-5 and abs(y - abs(yaw)) < 1e-5]
        linear_limits.append(min(matches) if matches else (.15 if abs(vx) >= 2 else .20))
    limits = torch.tensor(linear_limits, device=device)[bucket_ids]
    # Optional read-only recorder. Terminal states are sampled BEFORE auto-reset.
    world_frames = []
    world_pre = None
    if args.record_world_trajectory:
        from isaaclab.utils.math import quat_apply
    def world_state():
        forward_axis = torch.zeros_like(u.robot.data.root_lin_vel_w.torch)
        forward_axis[:, 2] = -1.
        forward = quat_apply(u.robot.data.root_quat_w.torch, forward_axis)
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        speed = u._heading_forward_velocity(u.robot.data.root_lin_vel_w.torch)
        return torch.cat((u.robot.data.root_pos_w.torch[:, :2], heading[:, None],
                          speed[:, None], u._curriculum_step_counter[:, None]), -1).clone()
    original_get_dones = u._get_dones
    index = 0
    observed_steps = 0
    trace_columns = ['episode_step', 'root_z', 'vx', 'yaw_rate', 'roll_deg',
                     'pitch_deg', 'pitch_rate'] + [f'raw_action_{i}' for i in range(6)] + [
                     f'executed_action_{i}' for i in range(6)]
    # Pair each post-step state with the observation that produced its action.
    # This permits exact offline mixture attribution without changing control.
    trace_columns += [f'action_input_observation_{i}' for i in range(25)] + [
        'speed_feedback', 'propulsion_feedback', 'pitch_feedback', 'yaw_feedback']
    trace_columns += [f'joint_position_{i}' for i in range(6)] + [
        f'joint_velocity_{i}' for i in range(6)] + [f'applied_motor_torque_{i}' for i in range(6)]
    trace_columns += ['wheel_requested_right_nm', 'wheel_requested_left_nm',
                      'wheel_target_right_nm', 'wheel_target_left_nm']
    action_input_observation = torch.zeros(count, 25, device=device)
    # Sample every 10 control steps, retain the last 500 steps on the GPU.
    trace_ring = torch.zeros(50, count, len(trace_columns), device=device) if (args.failure_traces or args.record_policy_trace) else None
    trace_cursor = 0
    trace_count = torch.zeros(count, dtype=torch.long, device=device)
    trace_recorded = torch.zeros(count, dtype=torch.bool, device=device)
    failure_traces = []
    full_policy_frames = []
    # Bounded offline evidence for nonterminal tracking failures. Keep on GPU
    # during stepping; this must not affect control, windows, resets or gates.
    tracking_probe_columns = trace_columns + ['window_linear_error', 'window_yaw_error']
    tracking_probe_limit = 20
    tracking_probe_frames = (torch.zeros(tracking_probe_limit, count, len(tracking_probe_columns), device=device)
                             if (args.failure_traces or args.record_policy_trace) else None)
    tracking_probe_counts = torch.zeros(count, dtype=torch.long, device=device)
    wheel_teacher_samples = []

    def observe_terminal_physics():
        nonlocal index, observed_steps, trace_cursor
        terminated, truncated = original_get_dones()
        last_terminated.copy_(terminated)
        last_truncated.copy_(truncated)
        done = terminated | truncated
        if args.record_world_trajectory:
            world_frames.append(torch.cat((world_pre, world_state(),
                terminated[:, None], truncated[:, None], command[:, [0, 2]]), -1).cpu())
        if args.collect_wheel_teacher and observed_steps % 8 == 0:
            # Capture BEFORE auto-reset clears actions. Input was copied before
            # env.step; DR0 ensures there is no communication-delay mispairing.
            wheel_teacher_samples.append((action_input_observation.clone(),
                current_raw_action.clone(), u._policy_actions[:, [2, 5]].clone(),
                done.clone(), u._curriculum_step_counter.clone()))
        gravity, velocity, angular = u._state()
        vx = u._heading_forward_velocity(u.robot.data.root_lin_vel_w.torch)
        root = u.robot.data.root_pos_w.torch[:, 2]
        root_min.copy_(torch.minimum(root_min, root))
        root_sum.add_(root)
        error_window[:, index] = (vx - command[:, 0]).abs()
        yaw_window[:, index] = (angular[:, 2] - command[:, 2]).abs()
        window_count.clamp_max_(window - 1).add_(1)
        index = (index + 1) % window
        denominator = window_count.clamp_min(1).float()
        linear = error_window.sum(-1) / denominator
        yaw = yaw_window.sum(-1) / denominator
        ready = (window_count >= window) & (u._curriculum_step_counter >= warmup) & ~done
        wheel_forces = u.contact_sensor.data.net_forces_w[:, u._wheel_contact_sensor_ids]
        contact = (torch.linalg.vector_norm(wheel_forces, dim=-1) > cfg.contact_force_threshold_N).float().mean(-1)
        safe = (-gravity[:, 2] > .94) & (velocity[:, 1].abs() < .35) & (root >= cfg.command_min_operating_root_height_m)
        stable = safe & (contact >= cfg.command_min_instantaneous_wheel_contact_fraction) & (linear < limits) & (yaw < cfg.command_advance_yaw_error_rad_s)
        samples.add_(ready.float())
        raw_action_sum.add_(current_raw_action * ready[:, None])
        raw_action_saturated.add_((current_raw_action.abs() >= 1.).float() * ready[:, None])
        executed_action_sum.add_(u._policy_actions * ready[:, None])
        joint_position_sum.add_(u.robot.data.joint_pos.torch[:, u._leg_control_ids] * ready[:, None])
        linear_sum.add_(linear * ready)
        yaw_sum.add_(yaw * ready)
        signed_yaw_sum.add_((angular[:, 2] - command[:, 2]) * ready)
        signed_speed_sum.add_((vx - command[:, 0]) * ready)
        stable_sum.add_((stable & ready).float())
        roll, target_roll = bank_reference(gravity, command)
        bank_sum.add_(torch.rad2deg((roll - target_roll).abs()) * ready)
        roll_sum.add_(torch.rad2deg(roll) * ready)
        pitch = torch.rad2deg(torch.atan2(gravity[:, 0], -gravity[:, 2])).abs()
        pitch_abs_max.copy_(torch.maximum(pitch_abs_max, pitch))
        if trace_ring is not None:
            frame = torch.cat((torch.stack((u._curriculum_step_counter.float(), root, vx,
                angular[:, 2], torch.rad2deg(roll),
                torch.rad2deg(torch.atan2(gravity[:, 0], -gravity[:, 2])), angular[:, 1]), -1),
                current_raw_action, u._policy_actions, action_input_observation,
                torch.stack((u._last_speed_feedback_action,
                             u._last_highspeed_propulsion_action,
                             u._last_pitch_feedback_action,
                             u._last_yaw_feedback_action), -1),
                u.robot.data.joint_pos.torch[:, u._control_dof_ids],
                u.robot.data.joint_vel.torch[:, u._control_dof_ids],
                u.robot.data.applied_torque.torch[:, u._control_dof_ids],
                u._last_requested_wheel_effort_nm,
                u._last_wheel_effort_target_nm), -1)
            if args.record_policy_trace:
                full_policy_frames.append(frame.cpu())
            if observed_steps % 10 == 0:
                tracking_bad = ready & ((linear >= limits) | (yaw >= cfg.command_advance_yaw_error_rad_s))
                probe_ids = (tracking_bad & (tracking_probe_counts < tracking_probe_limit)).nonzero().flatten()
                tracking_probe_frames[tracking_probe_counts[probe_ids], probe_ids] = torch.cat(
                    (frame[probe_ids], linear[probe_ids, None], yaw[probe_ids, None]), -1)
                tracking_probe_counts[probe_ids] += 1
                trace_ring[trace_cursor].copy_(frame)
                trace_cursor = (trace_cursor + 1) % 50
                trace_count.add_(1).clamp_max_(50)
            unsafe = (terminated | (root < cfg.command_min_operating_root_height_m)) & ~trace_recorded
            if len(failure_traces) < 16:
                for env_id in unsafe.nonzero().flatten().tolist()[:16 - len(failure_traces)]:
                    size = int(trace_count[env_id])
                    indices = (torch.arange(size, device=device) + trace_cursor - size) % 50
                    failure_traces.append({
                        'env_id': env_id, 'bucket': int(bucket_ids[env_id]),
                        'command': command[env_id].tolist(), 'global_step': observed_steps + 1,
                        'terminated': bool(terminated[env_id]),
                        'termination_reasons': [key for key, value in u.get_termination_reasons().items()
                                                if bool(value[env_id])],
                        'dr': {key: float(getattr(u, '_domain_' + key)[env_id]) for key in (
                            'base_mass_scale', 'friction_scale', 'wheel_damping_scale',
                            'wheel_torque_gain', 'observation_noise_std', 'observation_delay_steps')},
                        'sensor_model_state': {
                            'enabled': u._sensor_model_enabled,
                            'observation_delay_steps': int(u._sensor_delay_steps[env_id]),
                            'action_delay_steps': int(u._communication_delay_steps[env_id]),
                            'gyro_bias_rad_s': u._sensor_gyro_bias[env_id].tolist(),
                            'accel_bias_mps2': u._sensor_accel_bias[env_id].tolist()},
                        'history': trace_ring[indices, env_id].tolist(),
                        'event_state': frame[env_id].tolist(),
                    })
                    trace_recorded[env_id] = True
            trace_count[done] = 0
            trace_recorded[done] = False
        if turn_dynamics_sum is not None:
            authored_forward = torch.zeros_like(u.robot.data.root_lin_vel_w.torch)
            authored_forward[:,2] = -1.
            forward = quat_apply(u.robot.data.root_quat_w.torch,authored_forward)[:,:2]
            forward = forward / torch.linalg.vector_norm(forward,dim=-1,keepdim=True).clamp_min(1e-6)
            metrics = turn_metrics(
                u.robot.data.root_pos_w.torch, u.robot.data.root_lin_vel_w.torch,
                velocity, angular[:,2], u.robot.data.root_ang_vel_w.torch[:,2], forward,
                u.robot.data.body_com_pos_w.torch, u.robot.data.body_mass.torch,
                u.robot.data.body_pos_w.torch[:,u._wheel_body_ids],
                u.robot.data.body_link_lin_vel_w.torch[:,u._wheel_body_ids],
                u.robot.data.joint_vel.torch[:,u._wheel_joint_ids_by_body]*cfg.wheel_radius_m*u._wheel_action_sign_tensor,
                wheel_forces, u.scene.env_origins[:,2])
            world_bank = torch.rad2deg(world_bank_angle(u.robot.data.root_quat_w.torch,quat_apply))
            physical_inward = bank_aligned_with_turn_proxy(world_bank, vx,
                u.robot.data.root_ang_vel_w.torch[:,2])
            metrics = torch.cat((metrics,world_bank[:,None],physical_inward[:,None]),dim=-1)
            turn_extrema.update(metrics, wheel_forces[:, :, 2], ready, done)
            turn_dynamics_sum.add_(torch.where(ready[:,None],metrics,torch.zeros_like(metrics)))
        wheel_linear = u.robot.data.joint_vel.torch[:, u._wheel_control_ids] * cfg.wheel_radius_m
        wheel_forward = wheel_linear * u._wheel_action_sign_tensor
        slip_sum.add_((wheel_forward - vx[:, None]).abs().mean(-1) * ready)
        effort = u.robot.data.applied_torque.torch[:, u._wheel_control_ids].abs()
        torque_sum.add_(effort.mean(-1))
        torque_clip_sum.add_((effort >= .99 * u._last_effective_wheel_effort_limit[:, None]).float().mean(-1))
        demand=u._last_requested_wheel_effort_nm.abs()
        hit=demand > u._last_effective_wheel_effort_limit[:,None]
        demand_hits.add_(hit)
        demand_run.copy_(torch.where(hit,demand_run+u.step_dt,0.))
        demand_longest.copy_(torch.maximum(demand_longest,demand_run))
        demand_peak.copy_(torch.maximum(demand_peak,demand))
        demand_run[done]=0
        completed.add_(done.long())
        survived.add_((done & ~terminated & (u._curriculum_step_counter >= warmup)).long())
        for key, value in u.get_termination_reasons().items():
            reason_counts[key].add_((done & value).long())
        error_window[done] = 0
        yaw_window[done] = 0
        window_count[done] = 0
        observed_steps += 1
        return terminated, truncated

    u._get_dones = observe_terminal_physics
    held_legs = None
    startup_samples = []
    leg_tracking_samples = []
    # Opt-in short trace piggybacks on startup capture. Sample immediately
    # before physics, after actuator computation/write, without changing state.
    startup_physics = []
    startup_closure = []
    original_sim_step = u.sim.step
    if args.capture_startup_steps:
        from rmuc_closure_audit import ClosureAudit
        closure_audit = ClosureAudit(u.cfg.robot_cfg.spawn.usd_path, u.robot)
        def capture_physics_step(*step_args, **step_kwargs):
            if len(startup_physics) < 16:
                ids = u._control_dof_ids
                data = u.robot.data
                startup_physics.append(dict(
                    substep=len(startup_physics),
                    position=data.joint_pos.torch[0, ids].detach().cpu().tolist(),
                    velocity=data.joint_vel.torch[0, ids].detach().cpu().tolist(),
                    target=data.joint_pos_target.torch[0, ids].detach().cpu().tolist(),
                    effort=data.applied_torque.torch[0, ids].detach().cpu().tolist(),
                    executed_action=u._policy_actions[0].detach().cpu().tolist()))
            return original_sim_step(*step_args, **step_kwargs)
        u.sim.step = capture_physics_step
    if args.diagnostic_hold_legs_after_seconds is not None:
        from rmuc_eval_commands import HeldLegReference
        held_legs = HeldLegReference(count, device, args.diagnostic_hold_legs_after_seconds)
    # Per-control-step exposure, rather than the fresh post-reset final snapshot.
    delay_exposure = torch.zeros((count, 2), dtype=torch.long, device=device)
    delay_maximum = torch.zeros_like(delay_exposure)
    try:
        distill_buffer = []
        distill_shard = 0
        if args.distill_dataset_dir:
            args.distill_dataset_dir.mkdir(parents=True, exist_ok=False)
        for step in range(args.steps):
            with torch.inference_mode():
                if critic is not None:
                    saved_observation = obs['policy'].detach().clone()
                    predicted_value = critic(obs).squeeze(-1).detach().clone()
                if trace_ring is not None or args.collect_wheel_teacher:
                    action_input_observation.copy_(obs['policy'])
                action = actor(obs, stochastic_output=args.audit_critic)
                if held_legs is not None:
                    action = held_legs.apply(action, u._curriculum_step_counter, u.step_dt)
                if args.action_bias_start_seconds or args.action_bias_ramp_seconds:
                    from rmuc_eval_commands import diagnostic_bias_fraction
                    fraction = diagnostic_bias_fraction(u._curriculum_step_counter, u.step_dt,
                        args.action_bias_start_seconds, args.action_bias_ramp_seconds)
                    action = action + action_bias * fraction[:, None]
                else:
                    action = action + action_bias
                if args.distill_dataset_dir:
                    if args.audit_critic or args.action_bias_sweep:
                        raise ValueError('Distillation requires deterministic unbiased teacher')
                    teacher_label = distill_teacher(obs, stochastic_output=False) if distill_teacher is not None else action
                    distill_buffer.append((obs['policy'].detach().cpu().clone(), teacher_label.detach().cpu().clone()))
                    if len(distill_buffer) >= 256 or step == args.steps-1:
                        torch.save(dict(observations=torch.cat([v[0] for v in distill_buffer]),
                                        actions=torch.cat([v[1] for v in distill_buffer]),
                                        checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                                        rollout_student_sha256=student_sha256, deterministic_teacher=True),
                                   args.distill_dataset_dir / f'shard_{distill_shard:05d}.pt')
                        distill_buffer.clear(); distill_shard += 1
                current_raw_action.copy_(action)
                if args.record_leg_tracking:
                    leg_tracking_samples.append(torch.cat((
                        u._curriculum_step_counter[:,None],
                        u._heading_forward_velocity(u.robot.data.root_lin_vel_w.torch)[:,None],
                        u.robot.data.joint_pos.torch[:,u._control_dof_ids]),dim=-1).detach().clone())
                if (len(startup_samples) < args.capture_startup_steps
                        and step % args.capture_observation_stride == 0):
                    startup_closure.append(closure_audit.sample())
                    startup_samples.append((obs['policy'].clone(), action.clone(), u._curriculum_step_counter.clone()))
                if u._sensor_model_enabled:
                    active_delays = torch.stack((u._sensor_delay_steps, u._communication_delay_steps), -1)
                    delay_exposure.add_((active_delays > 0).long())
                    delay_maximum.copy_(torch.maximum(delay_maximum, active_delays))
                if args.record_world_trajectory:
                    world_pre = world_state()
                obs, reward, _, _ = env.step(action)
                if critic is not None:
                    critic_samples.append((saved_observation, predicted_value, reward.clone(),
                                           last_terminated.clone(), last_truncated.clone()))
            if (step + 1) % 1000 == 0:
                counts = torch.zeros(len(commands_list), dtype=torch.long, device=device).scatter_add_(0, bucket_ids, completed)
                print(f'[RMUC_GRID] steps={step + 1} minimum_completed={int(counts.min())}', flush=True)
            if (step + 1) % 100 == 0:
                counts = torch.zeros(len(commands_list), dtype=torch.long, device=device).scatter_add_(0, bucket_ids, completed)
                if bool(torch.all(counts >= args.min_episodes)):
                    break
        rows = []
        # A min-episode early break must not discard the final partial shard.
        if args.distill_dataset_dir and distill_buffer:
            torch.save(dict(observations=torch.cat([v[0] for v in distill_buffer]),
                            actions=torch.cat([v[1] for v in distill_buffer]),
                            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                            rollout_student_sha256=student_sha256, deterministic_teacher=True),
                       args.distill_dataset_dir / f'shard_{distill_shard:05d}.pt')
            distill_buffer.clear()
        for bucket, (vx, yaw) in enumerate(commands_list):
            selection = bucket_ids == bucket
            n = int(completed[selection].sum())
            sample_count = float(samples[selection].sum())
            denom = max(sample_count, 1)
            row = {'vx': vx, 'yaw': yaw, 'completed_episodes': n,
                   'action_bias_case': bias_labels[bucket],
                   'survival': float(survived[selection].sum()) / max(n, 1),
                   'steady_samples': sample_count,
                   'all_replicas_observed': bool(torch.all(samples[selection] > 0)),
                   'linear_error': float(linear_sum[selection].sum()) / denom,
                   'linear_limit': linear_limits[bucket],
                   'yaw_error': float(yaw_sum[selection].sum()) / denom,
                   'steady_signed_yaw_error_rad_s': float(signed_yaw_sum[selection].sum()) / denom,
                   'steady_signed_speed_error_mps': float(signed_speed_sum[selection].sum()) / denom,
                   'stable_fraction': float(stable_sum[selection].sum()) / denom,
                   'raw_action_mean': (raw_action_sum[selection].sum(0) / denom).tolist(),
                   'raw_action_saturation_fraction': (raw_action_saturated[selection].sum(0) / denom).tolist(),
                   'executed_action_mean': (executed_action_sum[selection].sum(0) / denom).tolist(),
                   'leg_joint_position_mean': (joint_position_sum[selection].sum(0) / denom).tolist(),
                   'bank_error_deg': float(bank_sum[selection].sum()) / denom,
                   'roll_mean_deg': float(roll_sum[selection].sum()) / denom,
                   'pitch_abs_max_deg': float(pitch_abs_max[selection].max()),
                   'steady_wheel_slip_speed_mps': float(slip_sum[selection].sum()) / denom,
                   'root_min': float(root_min[selection].min()),
                   'root_mean': float(root_sum[selection].mean()) / observed_steps,
                   'mean_wheel_effort_nm': float(torque_sum[selection].mean()) / observed_steps,
                   'applied_torque_saturation': float(torque_clip_sum[selection].mean()) / observed_steps,
                   'preclip_request_saturation_fraction_by_joint': (demand_hits[selection].mean(0)/observed_steps).tolist(),
                   'longest_sampled_saturation_s_by_joint': demand_longest[selection].amax(0).tolist(),
                   'max_sampled_preclip_request_nm_by_joint': demand_peak[selection].amax(0).tolist(),
                   'termination_counts': {k: int(v[selection].sum()) for k, v in reason_counts.items()}}
            if sample_count == 0:
                for key in ('linear_error', 'yaw_error', 'steady_signed_yaw_error_rad_s',
                            'steady_signed_speed_error_mps', 'bank_error_deg', 'roll_mean_deg',
                            'steady_wheel_slip_speed_mps'):
                    row[key] = None
            checks = {'episodes': n >= args.min_episodes,
                      'coverage': row['all_replicas_observed'],
                      'survival': row['survival'] >= cfg.command_min_survival_env_fraction,
                      'linear': sample_count > 0 and row['linear_error'] <= row['linear_limit'],
                      'yaw': sample_count > 0 and row['yaw_error'] <= cfg.command_advance_yaw_error_rad_s,
                      'stable': row['stable_fraction'] >= cfg.command_min_stable_env_fraction,
                      'height': row['root_min'] >= cfg.command_min_operating_root_height_m,
                      'bank': not args.require_analytic_bank or abs(yaw) < 1e-5 or (sample_count > 0 and row['bank_error_deg'] <= args.max_bank_error_deg)}
            row.update(checks=checks, passed=all(checks.values()))
            if turn_dynamics_sum is not None:
                row['turn_dynamics_extrema'] = turn_extrema.report(selection)
                row['turn_dynamics_steady'] = dict(zip(turn_columns,
                    (turn_dynamics_sum[selection].sum(0)/denom).tolist())) if sample_count>0 else None
            rows.append(row)
        report = {'checkpoint': str(args.checkpoint.resolve()), 'seed': args.seed,
                  'distilled_student': str(args.distilled_student) if args.distilled_student else None,
                  'student_sha256': student_sha256,
                  'diagnostic_wheel_velocity_profile': applied_velocity_profile,
                  'wheel_velocity_target_audit': None if target_velocity_audit is None else {
                      'compute_calls':target_velocity_audit['compute_calls'],
                      'max_applied_effort_nm':float(target_velocity_audit['max_applied_effort_nm']),
                      'max_reference_rad_s':float(target_velocity_audit['max_reference_rad_s']),
                      'saturated_joint_step_fraction':float(target_velocity_audit['saturated_joint_steps'])/max(target_velocity_audit['compute_calls']*count*2,1),
                      'controller_gain_kp':target_velocity_controller.kp,
                      'controller_gain_ki':target_velocity_controller.ki,
                      'controller_period_s':target_velocity_controller.dt,
                      'output_shaft_reference_scale_rad_s':reference_scale,
                      'old_actuator_D_used_as_drag':False},
                  'diagnostic_wheel_velocity_equivalence': args.diagnostic_wheel_velocity_equivalence,
                  'wheel_velocity_equivalence_audit': None if velocity_bridge_audit is None else {
                      'compute_calls': velocity_bridge_audit['compute_calls'],
                      'max_effort_error_nm': float(velocity_bridge_audit['max_effort_error']),
                      'max_reference_rad_s': float(velocity_bridge_audit['max_reference']),
                      'controller_dt_s': velocity_bridge.dt,
                      'reference_input': 'Old clipped normalized-action torque transformed by original D',
                      'scope': 'Per-physics-step P-loop equivalence only, not target3Nm or real motor response'},
                  'provenance_version': 2, 'bank_contract': BANK_CONTRACT, 'source_sha256_at_start': source_hashes,
                  'turn_acceptance_contract': 'analytic_bank_legacy' if args.require_analytic_bank else 'operator_tracking_v1',
                  'analytic_bank_required': args.require_analytic_bank,
                  'inward_diagnostic_contract': 'signed_actual_heading_speed_times_world_yaw_v2',
                  'turn_dynamics_audit_enabled': args.audit_turn_dynamics,
                  'turn_dynamics_scope': 'Steady ready samples only; wheel-center slip projected on root heading; net_forces_w contains NORMAL forces only: its horizontal/vertical ratio is NOT friction utilization. Quasistatic support margin is not exact ZMP or dynamic feasibility. COM is mass-weighted body COM, not root height.',
                  'runtime_contract': runtime_contract,
                  'checkpoint_observation_contract': checkpoint_observation,
                  'command_leg_scale_profile': leg_scale_profile_record,
                  'actual_leg_scale_knots': {'speed_abs_mps':u._manual_leg_speed_knots.detach().cpu().tolist(), 'scale_rad':u._manual_leg_scale_knots.detach().cpu().tolist()},
                  'diagnostic_hardware_profile': applied_hardware_profile,
                  'actual_wheel_actuator_damping_range': [float(u._wheel_actuator.damping.min()),float(u._wheel_actuator.damping.max())],
                  'actual_wheel_actuator_effort_limit_range_nm': [float(u._wheel_actuator.effort_limit.min()),float(u._wheel_actuator.effort_limit.max())],
                  'terrain_type': cfg.terrain.terrain_type,
                  'flat_mesh_subdivisions': getattr(cfg.terrain, 'flat_mesh_subdivisions', 0),
                  'diagnostic_plane_terrain': args.diagnostic_plane_terrain,
                  'diagnostic_mesh_subdivisions': args.diagnostic_mesh_subdivisions,
                  'terrain_prim_audit': terrain_prim_audit,
                  'initial_state_audit': initial_state_audit,
                  'initial_material_readback': initial_material_readback,
                  'initial_mass_inertia_readback': initial_mass_inertia_readback,
                  'final_material_readback': material_readback(),
                  'layout_seed': args.layout_seed, 'environment_bucket_ids': bucket_ids.tolist(),
                  'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  'dr_strength': args.dr_strength, 'steps': observed_steps, 'replicas': args.replicas,
                  'diagnostic_speed_ramp_seconds': args.diagnostic_speed_ramp_seconds,
                  'steady_warmup_control_steps': warmup,
                  'min_episodes_required': args.min_episodes,
                  'min_operating_root_height_m': cfg.command_min_operating_root_height_m,
                  'max_bank_error_deg': args.max_bank_error_deg,
                  'wheel_torque_scale_nm': cfg.wheel_torque_scale_nm,
                  'wheel_effort_limit_nm': cfg.robot_cfg.actuators['wheel_motors'].effort_limit,
                  'wheel_damping': cfg.robot_cfg.actuators['wheel_motors'].damping,
                  'wheel_pitch_rate_feedback_action_per_rad_s': cfg.wheel_pitch_rate_feedback_action_per_rad_s,
                  'diagnostic_feedback_override': args.pitch_rate_feedback is not None,
                  'diagnostic_disabled_high_speed_feedback': args.disable_high_speed_feedback,
                  'policy_outer_feedback_enabled': cfg.policy_outer_feedback_enabled,
                  'diagnostic_total_residual_budget': args.diagnostic_total_residual_budget,
                  'diagnostic_base_wheel_clip': args.base_wheel_clip,
                  'diagnostic_disabled_stage_head': args.disable_stage_head,
                  'diagnostic_stage_head_output_scale': args.stage_head_output_scale if args.disable_stage_head is not None else None,
                  'control_contract': 'command-based', 'rows': rows,
                  'diagnostic_action_bias': args.action_bias_sweep is not None,
                  'position_authority_consumer': args.position_authority_consumer,
                  'diagnostic_leg_target_offsets': args.diagnostic_leg_target_offsets,
                  'diagnostic_action_bias_start_seconds': args.action_bias_start_seconds,
                  'diagnostic_hold_legs_after_seconds': args.diagnostic_hold_legs_after_seconds,
                  'diagnostic_action_bias_ramp_seconds': args.action_bias_ramp_seconds,
                  'gas_spring_assist_torque_nm': cfg.gas_spring_assist_torque_nm,
                  'diagnostic_gas_spring_override': args.gas_spring_torque is not None,
                  'diagnostic_radial_spring_override': args.radial_spring_force is not None,
                  'gas_spring_radial_force_n': cfg.gas_spring_radial_force_n,
                  'gas_spring_radial_dr_fraction': cfg.gas_spring_radial_dr_fraction,
                  'final_radial_spring_force_per_env_n': u._radial_spring_force_n.tolist() if cfg.gas_spring_radial_force_n > 0 else None,
                  'dr_ablation': args.dr_ablation,
                  'command_sequence_policy_steps': sequence_table,
                  'fixed_grid_metrics_valid': sequence_table is None,
                  'diagnostic_wheel_peak_nm': args.diagnostic_wheel_peak_nm,
                  'preclip_saturation_scope': 'Last physics substep sampled each8ms; per-policy contiguous runs, not full physics-substep statistics',
                  'wheel_dr_sampling': 'per_environment_reset_v1' if wheel_dr_streams is not None else 'global_rng_legacy',
                  'wheel_dr_reset_budget': None if wheel_dr_streams is None else wheel_dr_streams.scales.shape[1],
                  'wheel_dr_reset_counts': None if wheel_dr_streams is None else wheel_dr_streams.indices.tolist(),
                  'sensor_model': cfg.domain_randomization_sensor_model,
                  'leg_transport': u._leg_transport.readback(),
                  'gravity_direction_dr': u._gravity_direction_dr.readback(),
                  'motor_response_enabled': cfg.motor_response_enabled,
                  'motor_response_time_constants_s': {name: response.time_constant.tolist()
                      for name,response in u._motor_responses.items()},
                  'motor_response_effort_audit': {name: {
                      'state_matches_applied_effort': bool(torch.equal(response.effort, u.robot.actuators[name].applied_effort)),
                      'applied_within_current_limit': bool((response.effort.abs() <= u.robot.actuators[name].effort_limit+1e-6).all()),
                      'last_response_gap_abs_max_nm': float((response.effort-torch.maximum(torch.minimum(
                          u.robot.actuators[name].computed_effort,u.robot.actuators[name].effort_limit),
                          -u.robot.actuators[name].effort_limit)).abs().max())}
                      for name,response in u._motor_responses.items()},
                  'diagnostic_motor_response_override': args.motor_response_time_constants_ms is not None,
                  'observation_delay_steps_per_env': u._sensor_delay_steps.tolist(),
                  'communication_delay_steps_per_env': u._communication_delay_steps.tolist(),
                  'sensor_delay_exposure': {
                      'columns': ['sensor_observation', 'action_communication'],
                      'nonzero_control_steps_per_env': delay_exposure.tolist(),
                      'maximum_delay_cycles_per_env': delay_maximum.tolist(),
                      'environment_bucket_ids': bucket_ids.tolist(),
                      'scope': 'Pre-step active sensor/communication delay per evaluated control input; legacy observation delay excluded'},
                  'diagnostic_dr_ablation': args.dr_ablation != 'none',
                  'dr_parameter_maxima': {name: getattr(cfg, 'domain_randomization_max_' + name) for name in
                      ('base_mass_fraction', 'base_inertia_fraction', 'friction_fraction', 'wheel_damping_fraction', 'wheel_torque_gain_fraction',
                       'observation_noise_std', 'observation_delay_steps', 'observation_delay_probability')},
                  'separated_wheel_dr_readback': ({
                      'mechanical_viscous_range_nm_s_rad': list(cfg.separated_wheel_mechanical_viscous_range),
                      'coulomb_range_nm': list(cfg.separated_wheel_coulomb_range),
                      'feedback_damping_range_nm_s_rad': list(cfg.separated_wheel_feedback_damping_range),
                      'mechanical_viscous_per_env_nm_s_rad': u._wheel_mechanical_viscous_per_env.tolist(),
                      'coulomb_per_env_nm': u._wheel_coulomb_per_env.tolist(),
                      'feedback_damping_per_env_nm_s_rad': u._wheel_feedback_damping_per_env.tolist(),
                      'feedback_damping_mode': getattr(cfg, 'wheel_feedback_damping_mode', 'postclip_loss_v1'),
                      'legacy_actuator_wheel_damping_dr_active': False,
                      'feedback_damping_hardware_identified': False,
                  } if u._separated_wheel_passive else None),
                  'mass_inertia_readback': {
                      'root_body_id': u._root_body_id,
                      'mass_kg': u.robot.data.body_mass.torch[:, u._root_body_id].tolist(),
                      'inertia_tensor_kg_m2': u.robot.data.body_inertia.torch[:, u._root_body_id, :].tolist(),
                      'mass_scale': u._domain_base_mass_scale.tolist(),
                      'independent_inertia_shape_scale': u._domain_base_inertia_shape_scale.tolist(),
                      'total_inertia_scale': u._domain_base_inertia_scale.tolist(),
                      'scope': 'Root body only; scalar inertia shape variation, no COM or principal-axis ratio randomization'},
                  'leg_dr_readback': {
                      'enabled': cfg.leg_actuator_dr_enabled,
                      'joint_ids': list(u._leg_control_ids),
                      'velocity_limit_rad_s': u.robot.data.joint_vel_limits.torch[:, u._leg_control_ids].tolist(),
                      'armature_kg_m2': u.robot.data.joint_armature.torch[:, u._leg_control_ids].tolist()},
                  'passed': sequence_table is None and args.diagnostic_wheel_peak_nm is None and getattr(args,'diagnostic_wheel_velocity_profile',None) is None and not getattr(args,'diagnostic_wheel_velocity_equivalence',False) and not args.diagnostic_plane_terrain and args.diagnostic_total_residual_budget is None and not args.disable_high_speed_feedback and args.disable_stage_head is None and args.base_wheel_clip is None and args.radial_spring_force is None and args.gas_spring_torque is None and args.dr_ablation == 'none' and not args.audit_critic and args.action_bias_sweep is None and args.pitch_rate_feedback is None
                            and getattr(args, 'motor_response_time_constants_ms', None) is None
                            and not getattr(args, 'diagnostic_disable_outer_feedback', False)
                            and not getattr(args, 'collect_wheel_teacher', False)
                            and not getattr(args, 'diagnostic_speed_ramp_seconds', 0.)
                            and getattr(args, 'diagnostic_hardware_profile', None) is None
                            and all(row['passed'] for row in rows)}
        if args.diagnostic_turn_entry:
            report['diagnostic_turn_entry'] = {'straight_seconds': args.diagnostic_turn_entry_hold_s,
                                              'yaw_ramp_seconds': args.diagnostic_turn_entry_ramp_s}
            report['passed'] = False
        if args.diagnostic_mesh_subdivisions:
            report['passed'] = False
        if args.arena_ground_severity is not None:
            report['arena_ground_material']=u.arena_ground_audit
            report['arena_ground_material']['legacy_friction_scaling_disabled']=True
            report['passed']=False
        if args.arena_ground_progress is not None:
            report['arena_ground_curriculum']=u._arena_ground_randomizer.readback()
            report['passed']=False
        if args.base_com_dr_strength is not None:
            report['base_com_randomization']=u._base_com_randomizer.readback()
            report['passed']=False
        if applied_separated_wheel_profile is not None:
            report['diagnostic_separated_wheel_profile'] = applied_separated_wheel_profile
            report['passed'] = False
        if args.real2sim_wheel_profile:
            report['real2sim_wheel'] = u._real2sim_wheel.readback()
            report['real2sim_wheel']['runtime_joint_armature_by_policy_wheel'] = u.robot.data.joint_armature.torch[:, u._wheel_control_ids].cpu().tolist()
            report['real2sim_wheel']['runtime_actuator_joint_order'] = list(u._wheel_actuator.joint_names)
            report['real2sim_wheel']['policy_wheel_order'] = [u.robot.joint_names[i] for i in u._wheel_control_ids]
            report['real2sim_wheel']['end_to_end_delay_mode'] = bool(cfg.real2sim_end_to_end_delay)
            report['real2sim_wheel']['legacy_fixed_observation_delay_steps'] = int(u._obs_delay_steps)
            report['real2sim_wheel']['profile_path'] = str(args.real2sim_wheel_profile.resolve())
            report['real2sim_wheel']['profile_sha256'] = hashlib.sha256(args.real2sim_wheel_profile.read_bytes()).hexdigest()
            report['real2sim_wheel']['source_estimates_validated'] = False
            report['passed'] = False
        if args.diagnostic_joint_dr_profile:
            report['diagnostic_joint_dr_profile'] = str(args.diagnostic_joint_dr_profile.resolve())
            report['passed'] = False
        report['solver_external_forces_every_iteration'] = cfg.sim.physics.enable_external_forces_every_iteration
        if args.diagnostic_tgs_external_forces_every_iteration:
            report['passed'] = False
        if args.diagnostic_leg_velocity_limit is not None or args.diagnostic_leg_armature is not None or args.diagnostic_leg_profile is not None:
            report['passed'] = False
        if leg_audit is not None:
            report['leg_actuator_audit'] = {k: v.tolist() if torch.is_tensor(v) else v for k, v in leg_audit.items()}
            report['leg_actuator_audit']['joint_order'] = list(u.robot.actuators['leg_motors'].joint_names)
            report['leg_actuator_audit']['final_solver_velocity_limits'] = u.robot.data.joint_vel_limits.torch[:, leg_joint_ids].cpu().tolist()
            report['leg_actuator_audit']['final_armature'] = u.robot.data.joint_armature.torch[:, leg_joint_ids].cpu().tolist()
            denominator = leg_audit['steady_physics_samples_per_env'].clamp_min(1)[:, None]
            report['leg_actuator_audit']['steady_control_means_per_env'] = {
                name: (value / denominator).cpu().tolist() for name, value in leg_sums.items()}
            report['leg_actuator_audit']['control_statistics_scope'] = 'Physics inputs age>=1200; applied effort is actuator output, not net constraint reaction torque. Zero sample count means unavailable.'
        if args.record_world_trajectory:
            import numpy as np
            from rmuc_trajectory_trace import analyze_trace, COLUMNS
            frames = torch.stack(world_frames).numpy()
            trace_path = args.output.with_suffix('.world_trace.npz')
            with trace_path.open('xb') as trace_file:
                np.savez_compressed(trace_file, frames=frames, dt=float(u.step_dt),
                    columns=np.asarray(COLUMNS), environment_bucket_ids=bucket_ids.cpu().numpy())
            report['world_trajectory'] = analyze_trace(frames, float(u.step_dt))
            report['world_trajectory']['trace_path'] = str(trace_path)
            report['world_trajectory']['trace_sha256'] = hashlib.sha256(trace_path.read_bytes()).hexdigest()
        if args.record_leg_tracking:
            import numpy as np
            tracking_path=args.output.with_suffix('.leg_tracking.npz')
            if tracking_path.exists():
                raise RuntimeError(f'Refusing to overwrite {tracking_path}')
            np.savez_compressed(tracking_path,frames=torch.stack(leg_tracking_samples).cpu().numpy(),
                                dt=float(u.step_dt),bucket_ids=bucket_ids.cpu().numpy(),
                                columns=np.array(['episode_step','actual_vx','q0','q1','q2','q3','q4','q5']))
            report['leg_tracking_trace']=dict(path=str(tracking_path),sha256=hashlib.sha256(tracking_path.read_bytes()).hexdigest(),
                scope='Pre-action simulator state each policy step; final-checkpoint evaluation, not training rollouts')
        if startup_samples:
            report['startup_physics_trace'] = startup_physics
            report['startup_closure_trace_m'] = startup_closure
            report['startup_physics_scope'] = 'First environment, first16 pre-physics actuator outputs; not net constraint/contact torque'
            dataset = args.output.with_suffix('.startup.pt')
            if dataset.exists():
                raise RuntimeError(f'Refusing to overwrite {dataset}')
            startup_obs, startup_actions, startup_age = [torch.stack(items).cpu() for items in zip(*startup_samples)]
            torch.save(dict(observations=startup_obs, actions=startup_actions, episode_age=startup_age,
                            checkpoint_sha256=report['checkpoint_sha256'], diagnostic=True), dataset)
            report['startup_dataset'] = str(dataset)
            report['capture_observation_stride'] = args.capture_observation_stride
        if args.collect_wheel_teacher:
            dataset = args.output.with_suffix('.wheel_teacher.pt')
            if dataset.exists():
                raise RuntimeError(f'Refusing to overwrite {dataset}')
            observations, raw_actions, wheel_targets, dones, ages = [torch.stack(items).cpu()
                for items in zip(*wheel_teacher_samples)]
            assert observations.shape[-1] == 25 and wheel_targets.shape[-1] == 2
            assert torch.isfinite(observations).all() and torch.isfinite(wheel_targets).all()
            assert bool((wheel_targets.abs() <= cfg.action_clip).all())
            assert not bool(u._communication_delay_steps.any())
            dataset.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'observations': observations, 'raw_actions': raw_actions,
                'wheel_targets': wheel_targets, 'done': dones, 'episode_step': ages,
                'command_per_env': command.cpu(), 'checkpoint': str(args.checkpoint),
                'source_sha256': source_hashes, 'sample_stride': 8,
                'target_semantics': 'Delivered normalized wheel action before reset; no leg imitation targets',
                'observation_dim': 25, 'wheel_action_indices': [2, 5]}, dataset)
            report['wheel_teacher_dataset'] = {'path': str(dataset),
                'samples': int(observations.shape[0] * observations.shape[1]),
                'scope': 'Teacher-assisted wheel initialization only; no inward-bank or E2E certificate'}
        if spring_wrench_audit is not None:
            report['spring_wrench_audit'] = {key: float(value) if torch.is_tensor(value) else value
                for key, value in spring_wrench_audit.items()}
        if critic is not None:
            from rmuc_critic_audit import returns_and_tail_bounds, value_metrics
            observations, values, rewards, terminated, truncated = [torch.stack(items).cpu()
                for items in zip(*critic_samples)]
            targets, tail_bounds = returns_and_tail_bounds(rewards, terminated, truncated,
                args.critic_gamma, max(abs(cfg.reward_clip_min), abs(cfg.reward_clip_max)))
            eligible = tail_bounds <= 5.
            rows_critic = []
            for bucket, (vx, yaw) in enumerate(commands_list):
                mask = eligible & (bucket_ids.cpu()[None, :] == bucket)
                rows_critic.append({'vx': vx, 'yaw': yaw, **value_metrics(values[mask], targets[mask])})
            dataset = args.output.with_suffix('.critic_samples.pt')
            if dataset.exists():
                raise RuntimeError(f'Refusing to overwrite {dataset}')
            dataset.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'observations': observations[eligible], 'predicted_values': values[eligible],
                        'returns': targets[eligible], 'tail_error_bounds': tail_bounds[eligible],
                        'checkpoint': str(args.checkpoint), 'gamma': args.critic_gamma}, dataset)
            report['critic_audit'] = {'diagnostic_only': True, 'stochastic_policy': True,
                'gamma': args.critic_gamma, 'max_unobserved_tail_error': 5., 'rows': rows_critic,
                'reward_overrides': audit_reward_overrides,
                'command_reward_contract_by_command': cfg.command_reward_contract_by_command,
                'dataset': str(dataset), 'overall': value_metrics(values[eligible], targets[eligible])}
        if sequence_table is not None:
            for row in report['rows']:
                row['fixed_command_checks_diagnostic_only']=row['checks']
                row['checks']={'fixed_command_not_applicable':False}
                row['passed']=False
            report['passed']=False
        if args.record_policy_trace:
            import numpy as np
            trace_path=args.output.with_suffix('.policy_trace.npz')
            with trace_path.open('xb') as trace_file:
                np.savez_compressed(trace_file,frames=torch.stack(full_policy_frames).numpy(),columns=np.asarray(trace_columns),bucket_ids=bucket_ids.cpu().numpy(),dt=float(u.step_dt))
            report['policy_trace']=dict(path=str(trace_path),sha256=hashlib.sha256(trace_path.read_bytes()).hexdigest(),scope='Input/raw action pre-step; physical state sampled post-step before reset, every policy step; not physics-substep trace')
        if args.failure_traces:
            report.update(failure_trace_columns=trace_columns, failure_traces=failure_traces,
                          failure_trace_motor_scope='Policy slot order; post-step final physics substep motor output, excludes passive spring/constraint reactions and intermediate substep peaks',
                          failure_trace_limit=16, failure_trace_sample_stride=10)
            tracking_probes = []
            # At most two replicas per command bucket; snapshots are evidence
            # samples, never a replacement for all-replica gate statistics.
            for bucket in range(len(commands_list)):
                ids = ((bucket_ids == bucket) & (tracking_probe_counts > 0)).nonzero().flatten()[:2]
                for env_id in ids.tolist():
                    size = int(tracking_probe_counts[env_id])
                    tracking_probes.append({'env_id': env_id, 'bucket': bucket,
                        'command': command[env_id].tolist(),
                        'samples': tracking_probe_frames[:size, env_id].tolist()})
            report.update(steady_tracking_probe_columns=tracking_probe_columns,
                          steady_tracking_probes=tracking_probes,
                          steady_tracking_probe_limit_per_env=tracking_probe_limit,
                          steady_tracking_probe_scope='First sampled post-warmup nonterminal tracking failures; at most2replicas per bucket; no acceptance effect')
        if args.audit_reward_contract:
            saved = {name: getattr(u, name).clone() for name in (
                '_rehearsal_mask', '_previous_actions', '_previous_command_aligned_speed',
                '_previous_command_aligned_speed_valid', '_estop')}
            saved_level, saved_flag, saved_extras = u._command_level, cfg.command_reward_contract_by_command, u.extras.copy()
            saved_state = u._state
            audit = {}
            try:
                for consistent in (False, True):
                    values = {}
                    for level in (26, 31):
                        for rehearsal in (False, True):
                            for name, value in saved.items():
                                getattr(u, name).copy_(value)
                            u._command_level = level
                            cfg.command_reward_contract_by_command = consistent
                            u._rehearsal_mask.fill_(rehearsal)
                            values[f'level{level}_rehearsal{int(rehearsal)}'] = u._get_rewards().detach().clone()
                    reference_reward = values['level26_rehearsal0']
                    audit['consistent' if consistent else 'legacy'] = {
                        key: float((value - reference_reward).abs().max()) for key, value in values.items()}
                # Counterfactual reward scan at the final physical pose. Only
                # the yaw sensor value changes; no simulation step is taken.
                gravity, velocity, angular = saved_state()
                audit['yaw_error_sweep'] = []
                for error in (0., .025, .05, .06, .075, .1):
                    for name, value in saved.items():
                        getattr(u, name).copy_(value)
                    u._command_level = 26
                    cfg.command_reward_contract_by_command = True
                    u._rehearsal_mask.fill_(False)
                    changed_angular = angular.clone()
                    changed_angular[:, 2] = command[:, 2] + error
                    u._state = lambda: (gravity, velocity, changed_angular)
                    clipped = u._get_rewards()
                    raw = u.extras['episode']['reward_unclipped']
                    audit['yaw_error_sweep'].append({'yaw_error': error, 'buckets': [
                        {'bucket': bucket, 'raw_mean': float(raw[bucket_ids == bucket].mean()),
                         'clipped_mean': float(clipped[bucket_ids == bucket].mean()),
                         'high_clip_fraction': float((raw[bucket_ids == bucket] > cfg.reward_clip_max).float().mean())}
                        for bucket in range(len(commands_list))]})
            finally:
                for name, value in saved.items():
                    getattr(u, name).copy_(value)
                u._command_level, cfg.command_reward_contract_by_command, u.extras = saved_level, saved_flag, saved_extras
                u._state = saved_state
            report['reward_contract_audit'] = audit
            report['audit_reward_overrides'] = audit_reward_overrides
        args.output.parent.mkdir(parents=True, exist_ok=True)
        from rmuc_strict_acceptance import audit as strict_acceptance_audit
        report['legacy_stage_passed'] = report['passed']
        report['strict_acceptance'] = strict_acceptance_audit(report)
        report['passed'] = report['strict_acceptance']['passed']
        report['row_passed_semantics'] = 'Legacy stage diagnostic only; strict_acceptance controls qualification'
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print('[RMUC_GRID_RESULT] ' + json.dumps(report), flush=True)
        return 0 if report['passed'] else 2
    finally:
        env.close()


try:
    code = main()
except BaseException:
    # Kit shutdown can exit before Python emits the pending exception.
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    raise
finally:
    app.close()
raise SystemExit(code)
