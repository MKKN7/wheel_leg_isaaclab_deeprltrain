"""CPU arithmetic checks for supplied hardware target; not a physics certificate."""
import argparse
import json
import math
from pathlib import Path
import yaml


def audit_legacy(config):
    wheel=config['wheel'];damping=wheel['damping'];spring=config['gas_spring']
    low,high=damping['range'];center=damping['proposed_center'];fraction=damping['full_dr_fraction']
    assert 0 <= low < high and abs(center*(1-fraction)-low)<1e-10
    assert abs(center*(1+fraction)-high)<1e-10
    assert wheel['max_continuous_output_torque_nm']==3 and not wheel['multiply_by_gear_ratio']
    assert wheel['max_peak_output_torque_nm']==5.5
    assert wheel['target_actuator_effort_limit_nm']==wheel['max_peak_output_torque_nm']
    assert spring['equivalent_torque_nm_per_leg']==10.775
    assert spring['use_legacy_constant_active_joint_torque']
    assert spring['joint_order']==['left_active_link_pitch','right_active_link_pitch']
    assert spring['joint_torque_sign']==[-1.,-1.] and spring['equal_opposite_chassis_reaction']
    evidence=wheel['speed_evidence']
    radius=evidence['reference_wheel_radius_m'];speed_anchor=evidence['user_reported_max_speed_mps']
    torque=wheel['max_continuous_output_torque_nm']
    bound=torque*radius/speed_anchor
    assert abs(bound-evidence['conditional_viscous_damping_upper_bound'])<1e-10
    rows=[]
    for speed in [2.0,2.5,2.9,3.0,3.2,speed_anchor]:
        for value in [low,center,high]:
            drag=value*speed/radius
            rows.append({'speed_mps':speed,'damping_nm_s_rad':value,'viscous_torque_nm':drag,
                         'ideal_wheel_only_torque_margin_nm':3.-drag})
    return {'supplied_parameters_checked':True,'measured_calibration':False,
        'peak_torque_limit_nm':wheel['max_peak_output_torque_nm'],
        'continuous_torque_reference_nm':torque, 'thermal_certified':False,
        'nominal_equivalent_torque_per_leg_nm':10.775,
        'spring_proposed_DR_range_per_leg_nm':[10.775*(1-spring['proposed_full_dr_fraction']),10.775*(1+spring['proposed_full_dr_fraction'])],
        'damping_reconstructed_range':[center*(1-fraction),center*(1+fraction)],
        'wheel_only_screen':rows,
        'high_damping_no_load_speed_mps':torque/high*radius,
        'speed_evidence':evidence,
        'speed_anchor_wheel_rad_s':speed_anchor/radius,
        'conditional_damping_upper_bound_nm_s_rad':bound,
        'previous_upper_endpoint_drag_at_anchor_nm':damping['original_user_estimated_range'][1]*speed_anchor/radius,
        'assumptions':['Wheel radius0.06m from existing config; steady rolling without slip.',
            'Damping treated as uncancelled viscous loss; no leg propulsion or transient energy.',
            'Explicit actuator D, passive joint damping, and compensated torque control are distinct; runtime mapping must be audited.'],
        'conclusion':'User-reported4m/s gives a conditional0.045 upper bound. Current user-selected engineering DR range0.002..0.04 leaves0.333Nm at its upper endpoint; neither the range nor its midpoint is measured. Old0.08 would require5.333Nm at4m/s under these assumptions.',
        'scope':'Analytic screening only, not proof that the complete robot cannot reach a speed; no new GPU run or physics change'}


def audit(config):
    if config.get('schema_version', 1) in (1, 2):
        return audit_legacy(config)
    assert config['schema_version'] == 3
    w=config['wheel']; spring=config['gas_spring']; timing=config['timing']
    assert w['torque_contract']=='separated_passive_v1'
    assert w['torque_reference']=='gearbox_output_shaft' and not w['multiply_by_gear_ratio']
    assert w['max_peak_output_torque_nm']==w['target_action_torque_scale_nm']==w['target_actuator_effort_limit_nm']==5.5
    assert w['max_continuous_output_torque_nm']==3.
    assert 'damping' not in w and not w['combined_torque_outer_clip']
    viscous=w['mechanical_viscous']; coulomb=w['coulomb_friction']; feedback=w['feedback_damping']
    low,high=viscous['range']; nominal=viscous['nominal']; fc=coulomb['nominal']
    assert all(math.isfinite(v) for v in (low,nominal,high,fc))
    assert [low,high]==[.001,.005] and nominal==.002 and fc==.001
    assert viscous['application']==coulomb['application']=='passive_dynamics_after_electromagnetic_clip'
    assert feedback['application_if_present']=='after_electromagnetic_clip_per_current_simulation_contract'
    assert feedback['firmware_application_order_verified'] is False
    assert timing['wheel_execution_period_s']==.001
    assert timing['leg_pd_period_s']==.002 and timing['policy_period_s']==.008
    assert timing['leg_pd_requires_sample_and_hold_at_1ms_physics']
    assert spring['model']=='equivalent_assist_joint_torque' and spring['equivalent_torque_nm_per_leg']==10.775
    assert not spring['radial_force_model_available'] and spring['radial_force_per_leg_n'] is None
    blockers=[]
    if feedback.get('value') is None or not feedback.get('identified', False):
        blockers.append('feedback_damping_absolute_value_unidentified')
    if not w['installation_signs']['hardware_verified']:
        blockers.append('wheel_installation_signs_not_hardware_verified')
    if coulomb.get('smoothing_omega_eps_rad_s') is None:
        blockers.append('numerical_coulomb_regularization_requires_explicit_experiment_choice')
    return {
        'supplied_parameters_checked': True, 'schema_version': 3,
        'measured_calibration': False, 'hardware_certified': False,
        'training_launch_ready': False,
        'runtime_adapter_status': 'legacy_profile_helper_refuses_schema3',
        'unresolved_items': blockers,
        'peak_torque_limit_nm': 5.5, 'continuous_torque_reference_nm': 3.,
        'action_scale_nm': 5.5, 'thermal_certified': False,
        'mechanical_viscous_nominal_nm_s_rad': nominal,
        'mechanical_viscous_range_nm_s_rad': [low,high],
        'coulomb_friction_prior_nm': fc,
        'feedback_damping': feedback,
        'timing': timing, 'nominal_equivalent_torque_per_leg_nm': 10.775,
        'passive_total_outer_clip': False,
        'conclusion': 'Component priors and distinct execution periods are explicit. '
                      'Mechanical viscous and Coulomb losses follow electromagnetic clipping. '
                      'Feedback damping is unidentified; no automatic mapping into IdealPD is allowed.',
        'scope': 'CPU contract consistency only; no runtime, behavior, identification or deployment certification',
    }


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(yaml.safe_load((root/'configs/rmuc_hardware_target.yaml').read_text()))
    with args.output.open('x') as f:f.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
