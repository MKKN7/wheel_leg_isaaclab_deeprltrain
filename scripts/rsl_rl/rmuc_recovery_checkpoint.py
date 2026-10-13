"""Persist recovery slew progress without restoring unrelated environment clocks."""
from pathlib import Path
import math
import torch
import yaml

KEY = 'rmuc_recovery_slew'


def configure_recovery_slew_checkpoint(runner, env, checkpoint=None, resume=False):
    env = getattr(env, 'unwrapped', env)
    cfg = env.cfg
    duration = getattr(cfg, 'recovery_slew_anneal_steps', 0)
    if not duration:
        return False
    contract = dict(version=1, duration_samples=duration,
                    start_rad_s=cfg.recovery_slew_start_rad_s,
                    end_rad_s=cfg.recovery_slew_end_rad_s)
    if type(duration) is not int or duration < 1 or any(
            not math.isfinite(contract[k]) or contract[k] <= 0
            for k in ('start_rad_s', 'end_rad_s')):
        raise ValueError('Invalid recovery slew schedule')
    offset = 0
    if resume and checkpoint:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=False)
        state = (payload.get('infos') or {}).get(KEY)
        if state is None:
            # An explicitly fixed-rate source can start a new anneal. A legacy
            # active anneal has no recoverable clock: never infer it from iter.
            source_cfg = yaml.load((Path(checkpoint).parent/'params/env.yaml').read_text(),
                                   Loader=yaml.BaseLoader)
            if int(source_cfg.get('recovery_slew_anneal_steps', 0)) != 0:
                raise ValueError('Active recovery anneal checkpoint lacks saved sample progress')
        else:
            if any(state.get(k) != v for k, v in contract.items()):
                raise ValueError('Recovery slew schedule differs from checkpoint')
            offset = state.get('elapsed_samples')
            if type(offset) is not int or offset < 0:
                raise ValueError('Invalid saved recovery slew sample progress')
    env._recovery_slew_sample_offset = offset
    original_save = runner.save

    def save(path, infos=None):
        merged = dict(infos or {})
        merged[KEY] = dict(contract, elapsed_samples=(
            env._recovery_slew_sample_offset + int(env._global_step_counter)))
        return original_save(path, infos=merged)

    runner.save = save
    print(f'[RECOVERY_SLEW_RESUME] elapsed_samples={offset} duration_samples={duration}', flush=True)
    return True
