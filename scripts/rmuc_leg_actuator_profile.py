"""Explicit leg-only target; does not alter USD, wheel torque or leg PD gains."""
import math
def apply_leg_profile(cfg, profile, enable_dr=False):
    if profile['parameter_frame'] != 'gearbox_output_joint':
        raise ValueError('Leg profile must use gearbox output joint units')
    velocity=float(profile['nominal_velocity_limit_rad_s'])
    armature=float(profile['nominal_armature_kg_m2'])
    if not all(math.isfinite(x) and x>0 for x in [velocity,armature]):
        raise ValueError('Positive finite nominal leg parameters required')
    vr=tuple(float(x) for x in profile['dr']['velocity_limit_rad_s'])
    ar=tuple(float(x) for x in profile['dr']['armature_kg_m2'])
    for nominal,bounds in [(velocity,vr),(armature,ar)]:
        if len(bounds)!=2 or not all(math.isfinite(x) for x in bounds) or not 0<bounds[0]<=nominal<=bounds[1]:
            raise ValueError('DR ranges must contain positive nominal parameters')
    actuator=cfg.robot_cfg.actuators['leg_motors']
    actuator.velocity_limit=velocity
    actuator.velocity_limit_sim=velocity
    actuator.armature=armature
    cfg.leg_actuator_dr_enabled=bool(enable_dr)
    cfg.leg_actuator_velocity_dr_range=vr
    cfg.leg_actuator_armature_dr_range=ar
    return {'nominal_velocity_limit_rad_s':velocity,'nominal_armature_kg_m2':armature,
            'dr_enabled':bool(enable_dr),'measured_calibration':profile['measured_calibration']}
