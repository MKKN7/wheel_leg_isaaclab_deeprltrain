"""Explicit joint-DR overrides shared by experiment preparation and evaluation."""
import copy
import math

FRACTIONS = (
    'domain_randomization_max_base_mass_fraction',
    'domain_randomization_max_base_inertia_fraction',
    'domain_randomization_max_friction_fraction',
    'domain_randomization_max_wheel_torque_gain_fraction',
    'domain_randomization_max_observation_delay_probability',
)
BOOLEANS = ('motor_response_enabled', 'leg_actuator_dr_enabled')
POSITIVE = ('motor_response_time_constant_min_s', 'motor_response_time_constant_max_s')
ALLOWED = set(FRACTIONS + BOOLEANS + POSITIVE) | {
    'domain_randomization_max_observation_noise_std',
    'domain_randomization_max_observation_delay_steps',
    'domain_randomization_sensor_model',
}
SEPARATED_WHEEL_RANGES = {
    'separated_wheel_mechanical_viscous_range': 'wheel_mechanical_viscous_nm_s_rad',
    'separated_wheel_coulomb_range': 'wheel_coulomb_friction_nm',
    'separated_wheel_feedback_damping_range': 'wheel_feedback_damping_nm_s_rad',
}


def validate_joint_dr_profile(profile):
    values = copy.deepcopy(profile['overrides'])
    if not ALLOWED <= set(values) <= ALLOWED | set(SEPARATED_WHEEL_RANGES):
        raise ValueError('Joint DR profile requires all standard fields and only supported optional wheel ranges')
    for key in SEPARATED_WHEEL_RANGES.keys() & values.keys():
        bounds = values[key]
        if (not isinstance(bounds, (list, tuple)) or len(bounds) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) for v in bounds)
                or not 0 <= bounds[0] <= bounds[1]):
            raise ValueError(f'Invalid separated wheel range: {key}')
    for key in FRACTIONS:
        if type(values[key]) not in (int, float) or not math.isfinite(values[key]) or not 0 <= values[key] < 1:
            raise ValueError(f'Invalid fraction: {key}')
    for key in BOOLEANS:
        if type(values[key]) is not bool:
            raise ValueError(f'Boolean required: {key}')
    for key in POSITIVE:
        if type(values[key]) not in (int, float) or not math.isfinite(values[key]) or values[key] <= 0:
            raise ValueError(f'Invalid response constant: {key}')
    if values[POSITIVE[0]] > values[POSITIVE[1]]:
        raise ValueError('Reversed motor response bounds')
    # Avoid accidental stacking of legacy uniform observation noise/delay.
    if (values['domain_randomization_max_observation_noise_std'] != 0
            or values['domain_randomization_max_observation_delay_steps'] != 0
            or values['domain_randomization_max_observation_delay_probability'] != 0):
        raise ValueError('Joint sensor model requires legacy noise/delay to be disabled')
    sensor = values['domain_randomization_sensor_model']
    keys = {'enabled', 'action_delay_enabled', 'delay_min_steps', 'delay_max_steps',
            'gyro_bias_max_rad_s', 'gyro_bias_walk_rad_s_sqrt_s', 'gyro_noise_std_rad_s',
            'accel_bias_max_mps2', 'accel_noise_std_mps2', 'encoder_angle_std_rad',
            'encoder_velocity_std_rad_s', 'forward_velocity_std_mps'}
    probability_keys = {'observation_delay_probability', 'communication_delay_probability'}
    if not isinstance(sensor, dict) or not keys <= set(sensor) <= keys | probability_keys:
        raise ValueError('Complete explicit sensor configuration required')
    for key in probability_keys & sensor.keys():
        value = sensor[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'Invalid independent delay probability: {key}')
    for key in ('enabled', 'action_delay_enabled'):
        if type(sensor[key]) is not bool:
            raise ValueError(f'Boolean sensor switch required: {key}')
    for key in ('delay_min_steps', 'delay_max_steps'):
        if type(sensor[key]) is not int or sensor[key] < 0:
            raise ValueError('Delay bounds must be nonnegative integer cycles')
    if sensor['delay_min_steps'] > sensor['delay_max_steps']:
        raise ValueError('Reversed sensor delay bounds')
    for key in keys - {'enabled', 'action_delay_enabled', 'delay_min_steps', 'delay_max_steps'}:
        if type(sensor[key]) not in (int, float) or not math.isfinite(sensor[key]) or sensor[key] < 0:
            raise ValueError(f'Invalid sensor bound: {key}')
    return values


def apply_joint_dr_profile(cfg, profile):
    values = validate_joint_dr_profile(profile)
    for key, nominal_name in SEPARATED_WHEEL_RANGES.items():
        if key in values and not values[key][0] <= getattr(cfg, nominal_name) <= values[key][1]:
            raise ValueError(f'Separated wheel range excludes nominal value: {key}')
    for key, value in values.items():
        setattr(cfg, key, value)
    return values
