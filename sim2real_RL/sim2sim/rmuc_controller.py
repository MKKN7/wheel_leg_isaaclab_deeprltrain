"""CPU ONNX replay controller. No task feedback or hardware current conversion."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from leg_pd import leg_pd_torque
from rmuc_policy_interface import LEG_INDICES, consume_action
from wheel_passive import (validate_wheel_contract, feedback_damping_is_active_motor,
                           feedback_damping_torque)


def separated_physics_dt(contract):
    diagnostic = contract.get('diagnostic_physics_dt_s')
    if diagnostic is None:
        return .001
    if diagnostic != .002 or contract.get('certified') is not False:
        raise ValueError('2ms diagnostic requires explicit uncertified contract')
    return .002


def load_contract(path):
    path = Path(path).resolve()
    contract = json.loads(path.read_text())
    separated = validate_wheel_contract(contract)
    if separated and contract.get('leg_pd_dt') != .002:
        raise ValueError('Separated contract requires leg_pd_dt=.002')
    expected = {
        'observation_dim': 25, 'action_dim': 6, 'policy_dt': .008,
        'physics_dt': separated_physics_dt(contract) if separated else .002, 'wheel_peak_nm': 5.5, 'wheel_continuous_nm': 3.,
        'wheel_action_scale_nm': 5.5, 'leg_kp': 60., 'leg_kd': 2.,
        'leg_limit_nm': 40., 'leg_scale_rad': .22, 'outer_feedback': False,
        'radial_force_n': 0., 'wheel_gear': 1.,
    }
    for key, value in expected.items():
        if key not in contract or contract[key] != value:
            raise ValueError(f'Unsupported controller contract: {key}')
    for key in (('gas_spring_torque_nm',) if separated else ('wheel_damping', 'gas_spring_torque_nm')):
        value = contract[key]
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'Invalid {key}')
    if contract['gas_spring_torque_nm'] > 10.775:
        raise ValueError('Unvalidated spring magnitude')
    if contract['wheel_action_sign'] != [-1., -1.]:
        raise ValueError('Wheel signs differ from current RMUC training contract')
    observation = contract.get('observation_contract')
    if observation is None:
        # Pre-v458 exports had no metadata and always used the historical clip=2.
        observation = dict(observation_dim=25,
                           command_observation_mode='tracking_error_with_target',
                           command_yaw_observation_error_clip=2.)
        contract['observation_contract_source'] = 'legacy_pre_v458_default_2rad_s'
    else:
        contract['observation_contract_source'] = 'explicit_export_metadata'
    if (observation.get('observation_dim') != 25
            or observation.get('command_observation_mode') != 'tracking_error_with_target'
            or observation.get('command_observation_scale', [2.1, .5, 1.]) != [2.1, .5, 1.]
            or observation.get('command_observation_error_clip', 2.) != 2.):
        raise ValueError('Controller observation layout mismatch')
    yaw_clip = observation.get('command_yaw_observation_error_clip')
    if isinstance(yaw_clip, bool) or not isinstance(yaw_clip, (int, float)) or not np.isfinite(yaw_clip) or yaw_clip < 2.:
        raise ValueError('Controller observation yaw clip must be finite and >=2 rad/s')
    contract['command_yaw_observation_error_clip'] = float(yaw_clip)
    action_contract = path.parent / 'export' / 'action_contract.json'
    if action_contract.exists():
        exported_observation = json.loads(action_contract.read_text()).get('observation_contract')
        if exported_observation is not None and exported_observation != observation:
            raise ValueError('Controller and exported policy observation contracts differ')
    for key in ('checkpoint', 'onnx', 'jit', 'mjcf'):
        artifact = Path(contract[key])
        if not artifact.is_absolute():
            artifact = path.parent / artifact
        artifact = artifact.resolve()
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != contract[key + '_sha256']:
            raise ValueError(f'{key} hash mismatch')
        contract[key] = str(artifact)
    if contract.get('controller_contract_version', 1) >= 2:
        if 'manual_leg_speed_knots' not in contract or 'manual_leg_scale_knots' not in contract:
            raise ValueError('Controller v2 requires command leg scale knots')
        for key in ('training_env', 'evaluation_report', 'export_action_contract'):
            if key not in contract or key + '_sha256' not in contract:
                raise ValueError(f'Controller v2 requires {key} and hash')
            artifact = Path(contract[key])
            if not artifact.is_absolute():
                artifact = path.parent / artifact
            artifact = artifact.resolve()
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != contract[key + '_sha256']:
                raise ValueError(f'{key} hash mismatch')
            contract[key] = str(artifact)
        exported_observation = json.loads(Path(contract['export_action_contract']).read_text()).get('observation_contract')
        if exported_observation != observation:
            raise ValueError('Controller and exported policy observation contracts differ')
        command_leg_scale(contract, 1.5)
    command_leg_scale(contract, 0.)
    return contract



def manual_leg_knots(params):
    """Match env.py dictionary construction, including last duplicate winning."""
    speeds = [float(value) for value in params['command_stage_linear_x_mps']]
    scales = [float(value) for value in params['command_stage_leg_scale_rad']]
    stand = float(params['stand_leg_position_scale_rad'])
    if (len(speeds) != len(scales) or not speeds or
        not np.isfinite([stand, *speeds, *scales]).all() or
        min([stand, *scales]) < 0.0):
        raise ValueError('Invalid command leg scale source')
    values = {0.0: stand}
    values.update(zip(speeds, scales))
    speeds = sorted(values)
    return dict(manual_leg_speed_knots=speeds,
                manual_leg_scale_knots=[values[speed] for speed in speeds])


def command_leg_scale(contract, command_vx=None):
    """CPU float32 equivalent of training interpolate_leg_scale; legacy fallback."""
    if 'manual_leg_speed_knots' not in contract:
        if 'manual_leg_scale_knots' in contract:
            raise ValueError('Incomplete manual leg knots')
        return float(contract['leg_scale_rad'])
    knots = np.asarray(contract['manual_leg_speed_knots'], dtype=np.float32)
    scales = np.asarray(contract.get('manual_leg_scale_knots', []), dtype=np.float32)
    if (knots.ndim != 1 or len(knots) < 2 or scales.shape != knots.shape
            or not np.isfinite(knots).all() or not np.isfinite(scales).all()
            or np.any(np.diff(knots) <= 0) or np.any(scales < 0)):
        raise ValueError('Invalid manual leg knots')
    if command_vx is None or not np.isfinite(command_vx):
        raise ValueError('Finite command_vx required with manual leg knots')
    value = np.clip(np.abs(np.float32(command_vx)), knots[0], knots[-1])
    upper = int(np.clip(np.searchsorted(knots, value), 1, len(knots) - 1))
    lower = upper - 1
    alpha = (value - knots[lower]) / (knots[upper] - knots[lower])
    return float(scales[lower] + alpha * (scales[upper] - scales[lower]))

def compute_effort(contract, action, position, velocity, default_position, joint_ranges, *, command_vx=None, _held_leg_output=None):
    """Return electromagnetic motor effort; mechanical losses remain external.

    Stateless candidate. LowLevelEffortScheduler holds leg effort between 2ms ticks.
    """
    separated = validate_wheel_contract(contract)
    action, position, velocity, default_position = [
        np.asarray(x, dtype=np.float64) for x in (action, position, velocity, default_position)
    ]
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    if any(x.shape != (6,) or not np.isfinite(x).all()
           for x in (action, position, velocity, default_position)):
        raise ValueError('Controller vectors must be finite and six dimensional')
    if ranges.shape != (6, 2) or not np.isfinite(ranges).all():
        raise ValueError('Joint ranges must be finite (6,2)')
    legs = LEG_INDICES
    if np.any(ranges[legs, 1] - ranges[legs, 0] <= .04):
        raise ValueError('Leg ranges cannot accommodate the joint margin')
    effort = np.zeros(6, dtype=np.float64)
    if _held_leg_output is None:
        targets = np.clip(default_position[legs] + command_leg_scale(contract, command_vx) * action[legs],
                          ranges[legs, 0] + .02, ranges[legs, 1] - .02)
        effort[legs] = leg_pd_torque(targets, position[legs], velocity[legs])
    else:
        held_effort, held_targets = _held_leg_output
        effort[legs] = held_effort
        targets = held_targets.copy()
    peak = contract['wheel_peak_nm']
    if separated:
        wheel_request = contract['wheel_action_scale_nm'] * np.asarray(
            contract['wheel_action_sign']) * action[[2, 5]]
        if feedback_damping_is_active_motor(contract):
            # The only clamp is the electromagnetic 5.5 Nm boundary. Passive
            # mechanical b/C remains in qfrc_applied in the MuJoCo consumer.
            wheel_request = contract['wheel_action_scale_nm'] * np.asarray(
                contract['wheel_action_sign']) * np.clip(action[[2, 5]], -1., 1.)
            wheel_request = wheel_request + feedback_damping_torque(
                contract, velocity[[2, 5]])
        effort[[2, 5]] = np.clip(wheel_request, -peak, peak)
        return effort, targets, wheel_request
    wheel_target = np.clip(contract['wheel_action_scale_nm'] *
                           np.asarray(contract['wheel_action_sign']) * action[[2, 5]], -peak, peak)
    wheel_unclipped = wheel_target - contract['wheel_damping'] * velocity[[2, 5]]
    effort[[2, 5]] = np.clip(wheel_unclipped, -peak, peak)
    return effort, targets, wheel_unclipped


class LowLevelEffortScheduler:
    """Wheel feedback at 1ms; leg electromagnetic effort held for 2ms."""

    def __init__(self, contract):
        self.contract = contract
        separated = validate_wheel_contract(contract)
        if separated and contract.get('leg_pd_dt') != .002:
            raise ValueError('Separated contract requires leg_pd_dt=.002')
        self.leg_stride = round(.002 / separated_physics_dt(contract)) if separated else 1
        self.tick = 0
        self.leg_effort = None
        self.leg_targets = None

    def compute(self, *args, **kwargs):
        hold = ((self.leg_effort, self.leg_targets) if self.tick % self.leg_stride else None)
        effort, targets, unclipped = compute_effort(
            self.contract, *args, _held_leg_output=hold, **kwargs)
        if self.tick % self.leg_stride == 0:
            self.leg_effort = effort[LEG_INDICES].copy()
            self.leg_targets = targets.copy()
        else:
            effort[LEG_INDICES] = self.leg_effort
            targets = self.leg_targets.copy()
        self.tick += 1
        return effort, targets, unclipped


class OnnxReplayController:
    """Policy/consumer replay; estimator, hardware transport and FSM remain external."""

    def __init__(self, contract_path):
        import onnxruntime as ort
        self.contract = load_contract(contract_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(self.contract['onnx'], sess_options=options,
                                            providers=['CPUExecutionProvider'])

    def infer(self, observation):
        obs = np.asarray(observation, dtype=np.float32)
        if obs.shape != (25,) or not np.isfinite(obs).all():
            raise ValueError('Finite 25D observation required')
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError('Expected one observation input and one action output')
        raw = self.session.run([outputs[0].name], {inputs[0].name: obs[None]})[0][0]
        if raw.shape != (6,) or not np.isfinite(raw).all():
            raise ValueError('Finite6D policy action required')
        return raw, consume_action(obs, raw)
