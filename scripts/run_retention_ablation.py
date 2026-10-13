"""Run sequential frozen-actor nominal/DR and feedback-parity comparisons."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--iterations', type=int, default=35)
    parser.add_argument('--cases', nargs='+', choices=['nominal', 'dr', 'parity_nominal', 'parity_dr'],
                        default=['nominal', 'dr', 'parity_nominal', 'parity_dr'])
    parser.add_argument('--prefix', default='level29_stage23_frozen')
    parser.add_argument('--low-speed-yaw-feedback-clip', type=float, default=None)
    parser.add_argument('--straight-feedback-only', action='store_true')
    parser.add_argument('--command-consistent-noise', action='store_true')
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--command-level', type=int, default=29)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        parser.error(f'Checkpoint not found: {checkpoint}')
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = '0'
    env['PYTHONPATH'] = os.pathsep.join(
        [str(root / 'source' / name) for name in ['agent_rl', 'agent_tasks', 'agent_world']]
        + ([env['PYTHONPATH']] if env.get('PYTHONPATH') else [])
    )
    summaries = {}
    for case in args.cases:
        name = f'{args.prefix}_{case}'
        log = root / f'train_{name}.log'
        # Refuse to overwrite evidence from a previous comparison.
        if log.exists():
            raise FileExistsError(log)
        snapshot = log.with_suffix('.sources.tar.gz')
        with tarfile.open(snapshot, 'x:gz') as archive:
            for relative in ['scripts/rsl_rl/train.py', 'scripts/rsl_rl/rmuc_command_adapter.py',
                             'configs/rmuc_wheel_leg.yaml',
                             'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg']:
                archive.add(root / relative, arcname=relative,
                            filter=lambda info: None if '__pycache__' in info.name else info)
        command = [sys.executable, 'scripts/rsl_rl/train.py', '--task',
                   'Robotics-Wheelbipe-V14-RMUC-Command-v0', '--num_envs', str(args.num_envs),
                   '--max_iterations', str(args.iterations), '--num_steps_per_env_override', '128',
                   '--command_level', str(args.command_level), '--start_command_at_target', '--checkpoint', str(checkpoint),
                   '--load_critic_for_finetune', '--action_std_override',
                   '0.004,0.004,0.006,0.004,0.004,0.006', '--freeze_action_std',
                   '--freeze_actor_until_command_level', '46', '--learning_rate_override', '2e-6',
                   '--num_learning_epochs_override', '1', '--allow_curriculum_override',
                   '--run_name', name, '--viz', 'none']
        if case.endswith('nominal'):
            command.append('--disable_progressive_flat_dr')
        if case.startswith('parity_'):
            command.append('--rehearsal_feedback_parity')
        if args.low_speed_yaw_feedback_clip is not None:
            command.extend(['--low_speed_yaw_feedback_clip', str(args.low_speed_yaw_feedback_clip)])
        if args.straight_feedback_only:
            command.append('--rehearsal_straight_feedback_only')
        if args.command_consistent_noise:
            command.append('--command_consistent_dr_noise')
        print(json.dumps({'case': case, 'command': command, 'log': str(log)}), flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, cwd=root, env=env, stdout=stream, stderr=subprocess.STDOUT)
        content = log.read_text(errors='replace')
        blocks = re.split(r'\[INFO\] Learning iteration ', content)
        metrics = dict(re.findall(r'Mean episode ([\w]+):\s*([-\d.]+)', blocks[-1]))
        timestamps = re.findall(r'Exact experiment name requested from command line: (\S+)', content)
        unchanged = False
        if result.returncode == 0 and timestamps:
            run = root / 'logs/rsl_rl/rmuc_wheel_leg_course' / f'{timestamps[-1]}_{name}'
            initial = torch.load(run / 'model_0.pt', map_location='cpu', weights_only=False)['actor_state_dict']
            final = torch.load(run / f'model_{args.iterations - 1}.pt', map_location='cpu', weights_only=False)['actor_state_dict']
            unchanged = all(torch.equal(value, final[key]) for key, value in initial.items())
            if not unchanged:
                raise RuntimeError(f'Frozen actor changed during {case}; comparison invalid')
        summaries[case] = {
            'returncode': result.returncode, 'checkpoint': str(checkpoint),
            'actor_unchanged': unchanged, 'source_snapshot': str(snapshot),
            'metrics': {k: float(v) for k, v in metrics.items()
                        if k.startswith(('level23_rehearsal_', 'command_retention_level',
                                         'domain_randomization_strength', 'command_retention_ready_fraction'))},
        }
        (root / f'{args.prefix}_summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
        print(json.dumps(summaries[case]), flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
