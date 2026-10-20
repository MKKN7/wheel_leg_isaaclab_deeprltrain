"""Check actual wheel-ground and root-COM readbacks, never certify behavior."""
import numpy as np


def audit(report, progress=.1, strength=.1):
    ground = report['arena_ground_curriculum']
    com = report['base_com_randomization']
    offsets = np.asarray(com['offset_flu_m_per_env'])
    count = len(offsets)
    left = np.asarray(ground['wheels_readback']['left_wheel']['effective_pair_per_env'])
    right = np.asarray(ground['wheels_readback']['right_wheel']['effective_pair_per_env'])
    lo = np.array([.5, .5, 0]) + progress * (np.array([.45, .38, .02]) - [.5, .5, 0])
    hi = np.array([.5, .5, 0]) + progress * (np.array([.65, .55, .02]) - [.5, .5, 0])
    checks = dict(
        all_clones_bound=ground['cloned_colliders_bound'] == 2*count,
        material_runtime_verified=ground['runtime_verified'] is True,
        material_readback=ground['write_readback_max_abs_error'] < 1e-6
            and ground['max_abs_error_vs_expected'] < 1e-6,
        ground_progress=abs(ground['migration_progress']-progress) < 1e-7,
        both_wheels_same_ground=bool(np.array_equal(left, right)),
        effective_pair_bounds=bool(np.isfinite(left).all() and np.all(left >= lo-1e-6)
                                  and np.all(left <= hi+1e-6)),
        ground_reset_all=all(v >= 1 for v in ground['reset_counts']),
        com_enabled=com['enabled'] is True,
        com_matches=com['actual_matches_requested'] is True,
        com_quaternion=com['com_quaternion_exactly_preserved'] is True,
        com_strength=all(abs(v-strength) < 1e-7 for v in com['strength_per_env']),
        com_bounds=bool(np.isfinite(offsets).all() and np.all(np.abs(offsets)
            <= np.asarray(com['max_offset_flu_m_at_strength1'])*strength+1e-8)),
        com_no_other_writes=com['mass_inertia_geometry_written_by_helper'] is False,
        com_reset=com['reset_write_calls'] >= 1,
    )
    if 'domain_friction_scale' in report:
        checks['no_duplicate_friction']=all(v == 1 for v in report['domain_friction_scale'])
    else:
        checks['no_duplicate_friction']=all(v == 1 for name in ['initial_material_readback', 'final_material_readback']
                                           for v in report[name]['friction_scale_per_env'])
    return dict(passed=all(checks.values()), checks=checks,
                scope='Actual properties only; not behavior or robustness certification')
