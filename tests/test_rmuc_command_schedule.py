import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim2real_RL/sim2sim'))
from command_schedule import CommandSchedule


def test_command_changes_exactly_on_policy_boundary_and_holds():
    schedule = CommandSchedule([
        dict(policy_step=0, vx=1, yaw=1),
        dict(policy_step=25, vx=0, yaw=0),
        dict(policy_step=50, vx=-1, yaw=-1),
    ])
    assert schedule.at(24)[1]['vx'] == 1
    assert schedule.at(25) == (1, dict(policy_step=25, vx=0., yaw=0.))
    assert schedule.at(1000)[1]['vx'] == -1


@pytest.mark.parametrize('entries', [
    [], [dict(policy_step=1, vx=1, yaw=0)],
    [dict(policy_step=0.5, vx=1, yaw=0)],
    [dict(policy_step=True, vx=1, yaw=0)],
    [dict(policy_step=0, vx=float('nan'), yaw=0)],
    [dict(policy_step=0, vx=1, yaw=0, feedback=1)],
    [dict(policy_step=0, vx=1, yaw=0), dict(policy_step=0, vx=0, yaw=0)],
    [dict(policy_step=0, vx=1, yaw=0), dict(policy_step=-1, vx=0, yaw=0)],
])
def test_invalid_schedule_rejected(entries):
    with pytest.raises(ValueError):
        CommandSchedule(entries)
