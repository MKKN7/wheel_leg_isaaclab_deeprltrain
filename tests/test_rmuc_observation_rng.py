import runpy
from pathlib import Path
import torch

m = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/observation_noise.py'))


def test_private_noise_sequence_ignores_unrelated_reset_random_draws():
    obs = torch.zeros(8, 25)
    obs[:, 7] = .5
    obs[:, 19:] = .3
    generators = [torch.Generator().manual_seed(91) for _ in range(2)]
    for _ in range(10):
        a = m['add_command_consistent_noise'](obs, torch.ones(8) * .01, torch.ones(8) * .8, generator=generators[0])
        torch.rand(137)  # Another environment resets and consumes global RNG.
        b = m['add_command_consistent_noise'](obs, torch.ones(8) * .01, torch.ones(8) * .8, generator=generators[1])
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        torch.testing.assert_close(a[:, 7], obs[:, 7], atol=0, rtol=0)
        torch.testing.assert_close(a[:, 19:], obs[:, 19:], atol=0, rtol=0)
        torch.testing.assert_close(a[:, 2] / .25 + a[:, 8], torch.ones(8) * .8)


def test_default_draw_preserves_legacy_rng_and_private_draw_does_not_consume_it():
    x = torch.zeros(6, 25)
    torch.manual_seed(51)
    expected = torch.randn_like(x)
    torch.manual_seed(51)
    torch.testing.assert_close(m['observation_randn_like'](x), expected, atol=0, rtol=0)
    before = torch.random.get_rng_state()
    m['observation_randn_like'](x, torch.Generator().manual_seed(7))
    assert torch.equal(before, torch.random.get_rng_state())


def test_physical_sensor_channels_share_private_stream_without_corrupting_history():
    obs = torch.zeros(4, 25)
    obs[:, 5] = -1
    obs[:, 7] = .25
    obs[:, 19:] = .2
    cfg = dict(gyro_noise_std_rad_s=.02, accel_noise_std_mps2=.05,
               encoder_angle_std_rad=.005, encoder_velocity_std_rad_s=.01,
               forward_velocity_std_mps=.01)
    generators = [torch.Generator().manual_seed(31) for _ in range(2)]
    args = (obs, torch.ones(4), torch.zeros(4, 3), torch.zeros(4, 3), torch.zeros(4), cfg)
    a = m['sensor_noise'](*args, generator=generators[0])
    torch.randn(153)
    b = m['sensor_noise'](*args, generator=generators[1])
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a[:, 3:6].norm(dim=-1), torch.ones(4))
    torch.testing.assert_close(a[:, 19:], obs[:, 19:], atol=0, rtol=0)
    torch.testing.assert_close(a[:, 7], obs[:, 7], atol=0, rtol=0)
