"""Fail before simulator startup if a candidate evaluation points at another run."""
import argparse
import hashlib
import json
from pathlib import Path


def verify(experiment):
    experiment = experiment.resolve()
    checkpoint = experiment / 'final_checkpoint.pt'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    status = json.loads((experiment / 'status.json').read_text())
    if digest != status['candidate_sha256']:
        raise ValueError('Candidate checkpoint differs from completed training hash')
    runs = json.loads((experiment / 'evaluation_command.json').read_text())['runs']
    checked = []
    for name in runs:
        directory = experiment / name
        if directory.is_symlink() or directory.resolve().parent != experiment:
            raise ValueError(f'{name}: scheduled evaluation must use its own real directory')
        spec = json.loads((directory / 'command.json').read_text())
        command = spec['command']
        output = Path(command[command.index('--output') + 1]).resolve()
        if output != directory / 'evaluation.json':
            raise ValueError(f'{name}: output points outside its evaluation directory')
        actual = Path(command[command.index('--checkpoint') + 1]).resolve()
        if actual != Path(spec['source_checkpoint']).resolve():
            raise ValueError(f'{name}: command checkpoint differs from recorded source')
        if hashlib.sha256(actual.read_bytes()).hexdigest() != spec['source_sha256']:
            raise ValueError(f'{name}: recorded checkpoint hash mismatch')
        if not name.startswith('candidate_'):
            continue
        if actual != checkpoint or Path(spec['source_checkpoint']).resolve() != checkpoint:
            raise ValueError(f'{name}: candidate checkpoint path mismatch')
        if output != directory / 'evaluation.json':
            raise ValueError(f'{name}: output points outside its candidate directory')
        if spec['source_sha256'] != digest:
            raise ValueError(f'{name}: candidate hash mismatch')
        checked.append(name)
    if not checked:
        raise ValueError('No candidate evaluations found')
    return dict(passed=True, checkpoint_sha256=digest, checked=checked)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=Path)
    print(json.dumps(verify(parser.parse_args().experiment), indent=2))
