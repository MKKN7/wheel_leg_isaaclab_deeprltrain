"""Explicit output-shaft components; priors are not hardware measurements."""
import numpy as np

SEPARATED = 'separated_passive_v1'
POSTCLIP_LOSS_V1 = 'postclip_loss_v1'
ACTIVE_MOTOR_V1 = 'active_motor_v1'
FIELDS = ('wheel_feedback_damping_nm_s_rad', 'wheel_mechanical_viscous_nm_s_rad',
          'wheel_coulomb_friction_nm', 'wheel_coulomb_omega_eps_rad_s')


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
    viscous = -contract[FIELDS[1]] * omega
    coulomb = -contract[FIELDS[2]] * np.tanh(omega / contract[FIELDS[3]])
    return viscous, coulomb


def feedback_damping_torque(contract, omega):
    omega = np.asarray(omega, dtype=np.float64)
    if not np.isfinite(omega).all():
        raise ValueError('Finite wheel velocity required')
    if not validate_wheel_contract(contract):
        return np.zeros_like(omega)
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
