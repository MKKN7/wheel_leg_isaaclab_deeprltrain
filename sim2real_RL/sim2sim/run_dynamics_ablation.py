"""Reproduce nominal timestep/passive damping contrasts with one fixed ONNX."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    here = Path(__file__).resolve().parent
    rows = []
    for name, flags in [('reference', []), ('dt002', ['--physics-dt', '.002']),
                        ('passive001', ['--passive-damping', '.01']),
                        ('combined', ['--physics-dt', '.002', '--passive-damping', '.01'])]:
        output = here/'probes'/f'v261_ablation_{name}.json'
        command = [sys.executable, str(here/'run_mujoco_probe.py'), '--xml',
            str(here/'wheeled_infantry.xml'), '--seconds', '10',
            '--speed', '1', '--yaw', '-1.4', '--wheel-effort-limit','5.5',
            '--output', str(output), *flags]
        subprocess.run(command, check=True)
        report = json.loads(output.read_text())
        rows.append(dict(name=name, path=str(output), command=command,
            stop_reason=report['stop_reason'], metrics=report['second_half_metrics'],
            max_leg_velocity=report['max_leg_velocity']))
    (here/'probes/dynamics_ablation.json').write_text(json.dumps(rows, indent=2)+'\n')


if __name__ == '__main__':
    main()
