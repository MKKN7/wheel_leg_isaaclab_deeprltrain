"""Apply an explicit user-supplied target model for migration diagnostics."""
import math


def apply_hardware_profile(cfg, profile, migration_fraction=1.0):
    if not math.isfinite(migration_fraction) or not 0<=migration_fraction<=1:
        raise ValueError('Migration fraction must be within0..1')
    schema = profile.get('schema_version', 1)
    if schema not in (1, 2, 3):
        raise ValueError('Unsupported hardware profile schema')
    if schema == 3:
        # This is a target specification, not an executable legacy damping profile.
        # Refuse even a zero-fraction application: accepting it would conceal an
        # unresolved contract in a run labelled as using the hardware target.
        feedback = profile['wheel'].get('feedback_damping', {})
        if feedback.get('value') is None or not feedback.get('identified', False):
            raise ValueError('Schema3 hardware target has unidentified feedback damping; '
                             'mechanical viscous damping must not map to IdealPD damping')
        raise ValueError('Schema3 separated passive target requires an explicit validated '
                         'multirate runtime adapter; legacy hardware-profile application is forbidden')
    wheel=profile['wheel'];damping=wheel['damping'];spring=profile['gas_spring'];response=profile['motor_response']
    continuous=float(wheel['max_continuous_output_torque_nm'])
    # Archived schema1 profiles used the continuous rating as a hard limit.
    # Preserve explicit historical diagnostics without reinterpreting them.
    torque=float(wheel.get('max_peak_output_torque_nm', continuous))
    action_scale=float(wheel['target_action_torque_scale_nm'])
    low,high=map(float,damping['range']);center=float(damping['proposed_center'])
    fraction=float(damping['full_dr_fraction']);equivalent_torque=float(spring['equivalent_torque_nm_per_leg'])
    force_fraction=float(spring['proposed_full_dr_fraction'])
    tmin,tmax=map(float,response['time_constant_range_s'])
    values=[torque,continuous,action_scale,low,high,center,fraction,force_fraction,tmin,tmax,equivalent_torque]
    if not all(math.isfinite(v) for v in values):raise ValueError('Hardware profile contains non-finite values')
    if (wheel['torque_reference']!='gearbox_output_shaft' or wheel['multiply_by_gear_ratio']
            or continuous!=3. or not continuous<=torque<=5.5 or action_scale<=0
            or wheel['target_actuator_effort_limit_nm']!=torque):
        raise ValueError('Require separate 3Nm continuous reference and declared output-shaft peak up to5.5Nm, without gear multiplication')
    if not (0<low<=center<=high and 0<=fraction<1
            and math.isclose(center*(1-fraction),low,abs_tol=1e-9)
            and math.isclose(center*(1+fraction),high,abs_tol=1e-9)):
        raise ValueError('Damping center/fraction must reproduce the supplied interval')
    if (spring.get('model') != 'equivalent_assist_joint_torque'
            or spring.get('radial_force_model_available', False)
            or spring.get('radial_force_per_leg_n') is not None
            or equivalent_torque<0 or not 0<=force_fraction<=1):
        raise ValueError('Equivalent gas-spring torque must be nonnegative')
    if not 0<tmin<=tmax:raise ValueError('Invalid response time constants')
    if migration_fraction==0:
        return {'migration_fraction':0.,'applied_overrides':{},'measured_calibration':profile['measured_calibration']}
    f=migration_fraction
    base=cfg.robot_cfg.actuators['wheel_motors']
    base_damping=float(base.damping)
    base_fraction=cfg.domain_randomization_max_wheel_damping_fraction
    mixed_low=(1-f)*base_damping*(1-base_fraction)+f*low
    mixed_high=(1-f)*base_damping*(1+base_fraction)+f*high
    center=(mixed_low+mixed_high)/2.
    fraction=(mixed_high-mixed_low)/(mixed_high+mixed_low)
    # Action units are independent from the physical output-shaft limit.
    # Legacy3Nm derating diagnostics remain readable; current target peak is5.5Nm.
    scale=(1-f)*cfg.wheel_torque_scale_nm+f*action_scale
    effort_limit=(1-f)*float(base.effort_limit)+f*torque
    force=0.0
    spring_torque=(1-f)*cfg.gas_spring_assist_torque_nm+f*equivalent_torque
    response_low=(1-f)*cfg.motor_response_time_constant_min_s+f*tmin if cfg.motor_response_enabled else f*tmin
    response_high=(1-f)*cfg.motor_response_time_constant_max_s+f*tmax if cfg.motor_response_enabled else f*tmax
    overrides=dict(wheel_torque_scale_nm=scale,wheel_damping_anneal_target=-1.,wheel_damping_compensation_ratio=0.,
        domain_randomization_max_wheel_damping_fraction=fraction,gas_spring_assist_torque_nm=spring_torque,
        gas_spring_radial_force_n=force,gas_spring_radial_dr_fraction=0.,
        gas_spring_torque_common_dr_fraction=force_fraction/2.,
        gas_spring_torque_independent_dr_fraction=force_fraction/2.,
        motor_response_enabled=True,motor_response_time_constant_min_s=response_low,motor_response_time_constant_max_s=response_high,
        torque_limit_ratio_sim=1.)
    for name,value in overrides.items():setattr(cfg,name,value)
    cfg.robot_cfg.actuators['wheel_motors'].damping=center
    cfg.robot_cfg.actuators['wheel_motors'].effort_limit=effort_limit
    cfg.wheel_authored_effort_limit_nm=torque
    return dict(overrides,wheel_actuator_damping=center,wheel_actuator_effort_limit_nm=effort_limit,
                migration_fraction=f,damping_full_DR_range=[mixed_low,mixed_high],
                continuous_torque_reference_nm=continuous, peak_torque_limit_nm=torque,
                legacy_continuous_as_hard_limit='max_peak_output_torque_nm' not in wheel,
                measured_calibration=profile['measured_calibration'])
