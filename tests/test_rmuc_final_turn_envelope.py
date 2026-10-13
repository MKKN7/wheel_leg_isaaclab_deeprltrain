import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_final_turn_envelope_radii_and_minimum_one_rad_s_below_three():
    cfg = yaml.safe_load((ROOT / 'configs/rmuc_wheel_leg.yaml').read_text())
    envelope = cfg['command']['final_turn_envelope']
    assert [row['speed_mps'] for row in envelope] == [1.0, 1.5, 2.0, 2.5, 3.0]
    for row in envelope:
        assert row['yaw_min_rad_s'] > 0
        assert row['yaw_max_rad_s'] >= row['yaw_min_rad_s']
        # User-provided radii are rounded to 0.1 m (0.83 m at 1.5 m/s).
        assert math.isclose(row['speed_mps'] / row['yaw_max_rad_s'], row['radius_min_m'], abs_tol=.051)
        assert math.isclose(row['speed_mps'] / row['yaw_min_rad_s'], row['radius_max_m'], rel_tol=.02)
    assert all(row['yaw_max_rad_s'] >= 1.0 for row in envelope if row['speed_mps'] < 3.0)


def test_extended_stages_are_appended_after_legacy_47_levels_and_cover_both_range_ends():
    cfg = yaml.safe_load((ROOT / 'configs/rmuc_wheel_leg.yaml').read_text())
    stages = cfg['command']['stages']
    assert len(stages) > 62
    assert stages[46]['name'] == 'terminal_3p2_turn_certified'
    ext = stages[47:]
    assert ext[0]['yaw_rate_rad_s'] == 0.6
    assert ext[-1]['linear_x_mps'] == 3.0 and ext[-1]['yaw_rate_rad_s'] == 1.0
    assert all(stage['yaw_rate_rad_s'] >= 0.4 for stage in ext)
    # Every speed starts from its actual legacy maximum, without large jumps.
    maxima={}
    for stage in stages[:47]:
        speed=stage['linear_x_mps']
        maxima[speed]=max(maxima.get(speed,0),stage['yaw_rate_rad_s'])
    for stage in ext:
        speed=stage['linear_x_mps'];yaw=stage['yaw_rate_rad_s']
        assert 0 < yaw-maxima[speed] <= .25+1e-9
        maxima[speed]=yaw
    for row in cfg['command']['final_turn_envelope']:
        assert maxima[row['speed_mps']] == row['yaw_max_rad_s']
