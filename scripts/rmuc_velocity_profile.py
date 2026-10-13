"""Validate and apply only the wheel velocity-loop diagnostic contract."""
import math
def apply_velocity_profile(cfg, profile, migration_fraction=1.):
    if not math.isfinite(migration_fraction) or not 0<=migration_fraction<=1:
        raise ValueError('Wheel migration fraction must be finite within0..1')
    p=profile['target_controller_proposal']
    keys=('period_s','proportional_gain_nm_per_rad_s','integral_gain_nm_per_rad',
          'velocity_filter_tau_s','output_effort_limit_nm','speed_reference_limit_rad_s',
          'legacy_action_reference_scale_rad_s')
    values={k:float(p[k]) for k in keys}
    if not all(math.isfinite(v) for v in values.values()):
        raise ValueError('Nonfinite wheel controller profile')
    if (not math.isclose(values['period_s'],cfg.sim.dt,abs_tol=1e-12)
        or values['proportional_gain_nm_per_rad_s']<=0 or values['integral_gain_nm_per_rad']<0
        or values['velocity_filter_tau_s']<0 or values['output_effort_limit_nm']!=3.
        or not 0<values['legacy_action_reference_scale_rad_s']<=values['speed_reference_limit_rad_s']):
        raise ValueError('Invalid controller timing, gains,3Nm limit or velocity reference scale')
    if profile['units']!={'wheel_velocity':'output_shaft_rad_per_s','wheel_effort':'output_shaft_N_m'}:
        raise ValueError('Controller must use gearbox output-shaft SI units')
    base=cfg.robot_cfg.actuators['wheel_motors']
    base_gain=float(base.damping)
    base_effort=float(base.effort_limit)
    base_action_scale=float(cfg.wheel_torque_scale_nm)
    if not all(math.isfinite(v) and v>0 for v in (base_gain,base_effort,base_action_scale)):
        raise ValueError('Migration requires positive finite legacy gain, effort and action scale')
    f=migration_fraction
    target_values=dict(values)
    values['proportional_gain_nm_per_rad_s']=(1-f)*base_gain+f*values['proportional_gain_nm_per_rad_s']
    values['integral_gain_nm_per_rad']*=f
    values['velocity_filter_tau_s']*=f
    values['output_effort_limit_nm']=(1-f)*base_effort+f*3.
    values['legacy_action_reference_scale_rad_s']=(1-f)*(base_action_scale/base_gain)+f*values['legacy_action_reference_scale_rad_s']
    if values['legacy_action_reference_scale_rad_s']>values['speed_reference_limit_rad_s']:
        raise ValueError('Legacy-equivalent reference would clip; choose a sufficient reference limit')
    values.update(migration_fraction=f,target_controller=target_values,
                  initial_gain=base_gain,initial_effort_limit_nm=base_effort)
    # Preserve the normalized actor/feedback action. Its old 5.5Nm scale is
    # legacy bookkeeping here: the explicit controller uses a separate rad/s scale.
    cfg.robot_cfg.actuators['wheel_motors'].stiffness=0.
    cfg.robot_cfg.actuators['wheel_motors'].damping=0.
    cfg.robot_cfg.actuators['wheel_motors'].effort_limit=values['output_effort_limit_nm']
    cfg.wheel_damping_anneal_target=-1.
    cfg.wheel_damping_compensation_ratio=0.
    return dict(values,scope='Only wheel controller migration: gas spring/passive losses unchanged; outer feedback retained; no hardware calibration',
                source_provenance=p['provenance'])
