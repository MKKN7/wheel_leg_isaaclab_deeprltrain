"""Summarize the last complete PPO metrics block without loading a large log."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def summarize(path: Path) -> dict:
    with path.open('rb') as stream:
        stream.seek(max(path.stat().st_size - 300_000, 0))
        tail = stream.read().decode(errors='replace')
    blocks = re.split(r'\[INFO\] Learning iteration ', tail)
    complete = [block for block in blocks[1:] if 'Iteration time:' in block]
    if not complete:
        return {'log': str(path), 'status': 'initializing', 'tail': tail[-700:]}
    block = complete[-1]
    values = {k: float(v) for k, v in re.findall(r'Mean episode ([\w]+):\s*([-\d.]+)', block)}
    level = int(values.get('command_level', -1))
    active_names = ['command_active_actual_vx', 'command_linear_error_ema',
                    'command_yaw_error_ema', 'command_survival_fraction_ema',
                    'command_retention_ready_fraction', 'domain_randomization_strength']
    report = {'log': str(path), 'iteration': block.splitlines()[0], 'level': level,
              'active': {k: values.get(k) for k in active_names},
              'blocked': [k.removeprefix('command_gate_blocked_') for k, v in values.items()
                          if k.startswith('command_gate_blocked_') and v > 0.5],
              'retention': {}}
    for stage in [8, 15, 18, 23, 26, 28, 30, 32, 34, 36, 38, 43, 45]:
        if stage >= level:
            continue
        report['retention'][stage] = {}
        for sign in ['positive', 'negative']:
            prefix = f'command_retention_level{stage}_{sign}_'
            report['retention'][stage][sign] = {
                short: values.get(prefix + key) for short, key in
                [('vx', 'linear_error_ema'), ('yaw', 'yaw_error_ema'),
                 ('stable', 'stable_fraction_ema'), ('survival', 'survival_fraction_ema'),
                 ('episodes', 'survival_episode_count')]
            }
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.log), indent=2))
