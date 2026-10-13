"""Build a hash-checked controller contract with Isaac's command leg scales."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'sim2real_RL' / 'sim2sim'))
from rmuc_controller import load_contract, manual_leg_knots, command_leg_scale


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_contract(source_path, training_env_path, evaluation_path):
    source_path = Path(source_path).resolve()
    training_env_path = Path(training_env_path).resolve()
    evaluation_path = Path(evaluation_path).resolve()
    contract = load_contract(source_path)
    training_env = yaml.load(training_env_path.read_text(), Loader=yaml.BaseLoader)
    evaluation = json.loads(evaluation_path.read_text())
    if evaluation.get('checkpoint_sha256') != contract['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint hash differs from controller source')
    runtime = evaluation['runtime_contract']
    params = {
        'stand_leg_position_scale_rad': float(training_env['stand_leg_position_scale_rad']),
        'command_stage_linear_x_mps': [float(v) for v in training_env['command_stage_linear_x_mps']],
        'command_stage_leg_scale_rad': [float(v) for v in training_env['command_stage_leg_scale_rad']],
    }
    for key in ('command_stage_linear_x_mps', 'command_stage_leg_scale_rad'):
        if not np.array_equal(np.asarray(params[key], dtype=np.float32),
                              np.asarray(runtime[key], dtype=np.float32)):
            raise ValueError(f'Training environment and evaluation differ: {key}')
    contract.update(manual_leg_knots(params))
    if abs(command_leg_scale(contract, 1.0) - contract['leg_scale_rad']) > 1e-6:
        raise ValueError('Reference 1 m/s leg scale differs from contract')
    export_contract = source_path.parent / 'export' / 'action_contract.json'
    if not export_contract.is_file():
        raise ValueError('Missing exported action contract')
    contract.update(controller_contract_version=2,
                    training_env=str(training_env_path), training_env_sha256=sha256(training_env_path),
                    evaluation_report=str(evaluation_path), evaluation_report_sha256=sha256(evaluation_path),
                    export_action_contract=str(export_contract.resolve()),
                    export_action_contract_sha256=sha256(export_contract))
    return contract


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-contract', required=True, type=Path)
    parser.add_argument('--training-env', required=True, type=Path)
    parser.add_argument('--evaluation', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite controller contract')
    contract = build_contract(args.source_contract, args.training_env, args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(contract, stream, indent=2)
        stream.write('\n')
    load_contract(args.output)


if __name__ == '__main__':
    main()
