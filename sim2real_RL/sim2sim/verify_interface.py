"""Compare backend-neutral observations against the actual training AST prefix."""
import ast
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import onnxruntime as ort
import torch

from rmuc_policy_interface import build_observation, consume_action

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/rsl_rl'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--export',type=Path,default=ROOT/'artifacts/continuation_v261/export_final')
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('interface_verification.json'))
    args=parser.parse_args()
    source = ROOT / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == '_get_observations')
    # Execute the real source up to obs concatenation, before sensor noise/delay.
    end = next(i for i, n in enumerate(method.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'obs' for t in n.targets))
    method.body = method.body[:end+1] + [ast.Return(value=ast.Name(id='obs', ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {'torch': torch}
    exec(compile(module, str(source), 'exec'), namespace)
    consumer = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                    and n.name == '_compute_policy_action')
    assert isinstance(consumer.body[0], ast.Expr)  # docstring
    assert isinstance(consumer.body[1], ast.Assign)
    assert isinstance(consumer.body[2], ast.If)
    assert 'position_authority_enabled' in ast.unparse(consumer.body[2].test)
    consumer.body = consumer.body[:3] + [ast.Return(value=ast.Name(id='policy_actions', ctx=ast.Load()))]
    exec(compile(ast.fix_missing_locations(ast.Module(body=[consumer], type_ignores=[])),
                 str(source), 'exec'), namespace)
    rng = np.random.default_rng(261)
    count = 1024
    omega = rng.normal(size=(count, 3)).astype('float32')
    gravity = rng.normal(size=(count, 3)).astype('float32')
    gravity /= np.linalg.norm(gravity, axis=1, keepdims=True)
    pos, vel, default, history = [rng.normal(size=(count, 6)).astype('float32') for _ in range(4)]
    forward = rng.uniform(-4, 4, count).astype('float32')
    commands = rng.uniform(-4, 4, (count, 3)).astype('float32')
    tensor = torch.from_numpy
    env = NS(_current_command=lambda: tensor(commands),
             _state=lambda: (tensor(gravity), torch.zeros(count, 3), tensor(omega)),
             _heading_forward_velocity=lambda _: tensor(forward),
             robot=NS(data=NS(joint_pos=NS(torch=tensor(pos)), joint_vel=NS(torch=tensor(vel)),
                              root_lin_vel_w=NS(torch=torch.zeros(count, 3)))),
             _control_dof_ids=list(range(6)), _default_control_pos=tensor(default),
             _policy_actions=tensor(history), _command_scale=torch.tensor([2.1, .5, 1.]),
             cfg=NS(command_observation_mode='tracking_error_with_target',
                    command_observation_error_clip=2., command_yaw_observation_error_clip=2.))
    expected = namespace['_get_observations'](env).numpy()
    actual = np.stack([build_observation(angular_velocity_body=omega[i], gravity_body=gravity[i],
        heading_forward_velocity=forward[i], command_vx=commands[i, 0], command_yaw=commands[i, 2],
        joint_position=pos[i], joint_velocity=vel[i], default_joint_position=default[i],
        previous_executed_action=history[i]) for i in range(count)])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    probes = actual.copy()
    probes[:512, 7] = rng.uniform(.39, .61, 512)
    probes[:512, 8] = rng.uniform(-1.5, 1.5, 512)
    for i, speed in enumerate([.8-1e-4, .8, .8+1e-4, 1., 1.2-1e-4, 1.2, 1.2+1e-4]):
        for j, yaw in enumerate([-.6, -.4501-1e-4, -.4501, 0., .4501, .4501+1e-4, .6]):
            k = i*7+j
            probes[k, 2] = 0
            probes[k, 7] = speed*.5
            probes[k, 8] = yaw
    for i, boundary in enumerate([.8, 1.2]):
        for j, direction in enumerate([-np.inf, np.inf]):
            k = 60 + i*2+j
            probes[k, 2] = 0
            probes[k, 7] = np.nextafter(np.float32(boundary), np.float32(direction))*.5
            probes[k, 8] = 1.
    raw = rng.uniform(-3, 3, (count, 6)).astype('float32')
    consumed = np.stack([consume_action(o, a) for o, a in zip(probes, raw)])
    env.cfg.position_authority_enabled = True
    env.cfg.policy_outer_feedback_enabled = False
    env.cfg.action_clip = 1.
    env._position_authority_observation = tensor(probes)
    reference = namespace['_compute_policy_action'](env, tensor(raw)).numpy()
    np.testing.assert_allclose(consumed, reference, rtol=0, atol=1e-6)
    exported = args.export
    session = ort.InferenceSession(str(exported/'policy.onnx'), providers=['CPUExecutionProvider'])
    jit = torch.jit.load(str(exported/'policy.pt')).eval()
    input_name, output_name = session.get_inputs()[0].name, session.get_outputs()[0].name
    if session.get_inputs()[0].shape[0] == 1:
        actions=np.concatenate([session.run([output_name], {input_name: row[None]})[0] for row in actual])
    else:
        actions=session.run([output_name], {input_name: actual})[0]
    with torch.no_grad():
        expected_actions = jit(tensor(actual)).numpy()
    assert actions.shape == (count, 6) and np.isfinite(actions).all()
    np.testing.assert_allclose(actions, expected_actions, rtol=1e-4, atol=1e-5)
    report = dict(passed=True, samples=count, observation_max_abs_error=float(abs(actual-expected).max()),
                  consumer_max_abs_error=float(abs(consumed-reference).max()),
                  onnx_jit_max_abs_error=float(abs(actions-expected_actions).max()),
                  training_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  onnx_sha256=hashlib.sha256((exported/'policy.onnx').read_bytes()).hexdigest(),
                  scope='Nominal observation, authority consumer and full ONNX/JIT parity only; no MuJoCo closed-loop, state estimator, actuator or DR certification.')
    output = args.output
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
