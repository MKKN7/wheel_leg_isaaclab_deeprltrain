"""Explicit output-shaft components; priors are not hardware measurements."""
import numpy as np
import json
from functools import lru_cache

SEPARATED = 'separated_passive_v1'
POSTCLIP_LOSS_V1 = 'postclip_loss_v1'
ACTIVE_MOTOR_V1 = 'active_motor_v1'
FIELDS = ('wheel_feedback_damping_nm_s_rad', 'wheel_mechanical_viscous_nm_s_rad',
          'wheel_coulomb_friction_nm', 'wheel_coulomb_omega_eps_rad_s')


def fitted_profile(contract):
    profile = contract.get('wheel_fit_profile')
    if profile is None:
        return None
    _validate_fitted_profile(json.dumps(profile, sort_keys=True), contract.get('wheel_feedback_damping_mode'))
    return profile


@lru_cache(maxsize=32)
def _validate_fitted_profile(serialized, feedback_mode):
    profile = json.loads(serialized)
    # Validate with the same schema as Isaac; profile is embedded and hashed
    # as part of the controller contract, not silently loaded from a default.
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/real2sim_wheel.py'
    spec = importlib.util.spec_from_file_location('rmuc_fit_schema', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    module.validate_profile(profile)
    if profile['delay_mode'] != 'none' or any(profile['delay_s']):
        raise ValueError('MuJoCo fitted profile currently supports zero added delay only')
    if not profile['compensation_enabled'] or feedback_mode != ACTIVE_MOTOR_V1:
        raise ValueError('Fitted compensation requires active_motor_v1')
    return profile


def validate_wheel_contract(contract):
    mode = contract.get('wheel_torque_contract', 'legacy')
    if mode not in ('legacy', SEPARATED):
        raise ValueError('Unknown wheel_torque_contract')
    if mode == 'legacy':
        if any(k in contract for k in FIELDS):
            raise ValueError('Separated wheel fields require separated_passive_v1')
        return False
    if 'wheel_damping' in contract:
        raise ValueError('Ambiguous wheel_damping in separated contract')
    for key in FIELDS:
        if key not in contract or not np.isfinite(contract[key]) or contract[key] < 0:
            raise ValueError(f'Invalid {key}')
    if contract[FIELDS[-1]] <= 0:
        raise ValueError('wheel_coulomb_omega_eps_rad_s must be positive')
    if contract.get('hardware_sign_verified') is not False:
        raise ValueError('hardware_sign_verified must remain false')
    if contract.get('wheel_feedback_damping_mode', POSTCLIP_LOSS_V1) not in (
            POSTCLIP_LOSS_V1, ACTIVE_MOTOR_V1):
        raise ValueError('Invalid wheel_feedback_damping_mode')
    fitted_profile(contract)
    return True


def feedback_damping_is_active_motor(contract):
    """True when feedback D contributes before the electromagnetic peak clip."""
    if not validate_wheel_contract(contract):
        return False
    return contract.get('wheel_feedback_damping_mode', POSTCLIP_LOSS_V1) == ACTIVE_MOTOR_V1


def passive_components(contract, omega):
    omega = np.asarray(omega, dtype=np.float64)
    if not np.isfinite(omega).all():
        raise ValueError('Finite wheel velocity required')
    if not validate_wheel_contract(contract):
        return np.zeros_like(omega), np.zeros_like(omega)
    profile = contract.get('wheel_fit_profile')
    if profile is not None:
        # Isaac model-right wheel is physical Left; model-left is physical Right.
        return (-np.asarray(profile['mechanical_viscous_nm_s_rad']) * omega,
                -np.asarray(profile['coulomb_friction_nm']) * np.tanh(omega/profile['friction_epsilon_rad_s']))
    viscous = -contract[FIELDS[1]] * omega
    coulomb = -contract[FIELDS[2]] * np.tanh(omega / contract[FIELDS[3]])
    return viscous, coulomb


def feedback_damping_torque(contract, omega):
    omega = np.asarray(omega, dtype=np.float64)
    if not np.isfinite(omega).all():
        raise ValueError('Finite wheel velocity required')
    if not validate_wheel_contract(contract):
        return np.zeros_like(omega)
    profile = contract.get('wheel_fit_profile')
    if profile is not None:
        return (-np.asarray(profile['damping_compensation_nm_s_rad']) * omega
                + np.asarray(profile['coulomb_compensation_nm']) * np.tanh(omega/profile['friction_epsilon_rad_s']))
    return -contract['wheel_feedback_damping_nm_s_rad'] * omega


def configure_passive_runtime(model, wheel_dofs):
    """Remove native terms only in memory; qfrc_applied owns wheel passivity."""
    old = dict(damping=model.dof_damping[wheel_dofs].tolist(),
               frictionloss=model.dof_frictionloss[wheel_dofs].tolist())
    model.dof_damping[wheel_dofs] = 0.
    model.dof_frictionloss[wheel_dofs] = 0.
    return old


def apply_passive(data, wheel_dofs, viscous, coulomb):
    # Hinge generalized torque acts between child and parent, not world torque.
    data.qfrc_applied[wheel_dofs] += viscous + coulomb
