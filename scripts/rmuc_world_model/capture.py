"""Opt-in replay capture with terminal state before reset. Never mutates env tensors."""
from pathlib import Path
import json,hashlib
import torch

def snapshot(env):
    def tensor(x):return x if torch.is_tensor(x) else x.torch
    data=env.robot.data
    state={
        'episode_step':env._curriculum_step_counter,
        'root_state_w':tensor(data.root_state_w),
        'joint_pos':tensor(data.joint_pos),'joint_vel':tensor(data.joint_vel),
        'applied_torque':tensor(data.applied_torque),
        'policy_actions':env._policy_actions,
        'obs_history':env._obs_history,'action_history':env._command_action_history,
        'sensor_delay_cycles':env._sensor_delay_steps,'action_delay_cycles':env._communication_delay_steps,
        'obs_history_valid':env._obs_history_valid,
        'leg_history':env._leg_transport.history.movedim(1,0),
        'leg_delay_active':env._leg_transport.active,
        'contact_forces_w':tensor(env.contact_sensor.data.net_forces_w),
    }
    for group,response in env._motor_responses.items():
        state[f'motor_{group}_effort']=response.effort
        state[f'motor_{group}_tau']=response.time_constant
    wheel=env._real2sim_wheel
    if wheel is not None:
        state['wheel_delay_history']=wheel._delay.history
        state['wheel_transport_active']=wheel._transport_active
        state['wheel_motor_output']=wheel.last_motor_output
    for name in ['_sensor_gyro_bias','_sensor_accel_bias','_wheel_feedback_damping_per_env','_wheel_mechanical_viscous_per_env','_wheel_coulomb_per_env','_default_control_pos']:
        if hasattr(env,name):state[name]=getattr(env,name)
    result={k:tensor(v).detach().clone() for k,v in state.items()}
    result['root_state_w'][:,:3]-=env.scene.env_origins
    return result

class ReplayCapture:
    def __init__(self,directory,metadata,shard_steps=128):
        self.path=Path(directory);self.path.mkdir(parents=True,exist_ok=False)
        self.metadata=metadata;self.limit=shard_steps;self.pending=None;self.rows=[];self.shards=[]
    def begin(self,state,observation,action):
        if self.pending is not None:raise RuntimeError('Unfinished transition')
        self.pending={'pre':state,'observation':observation.detach().clone(),'raw_action':action.detach().clone()}
    def terminal(self,state,terminated,truncated):
        if self.pending is None or 'post' in self.pending:raise RuntimeError('Terminal snapshot out of order')
        self.pending.update(post=state,terminated=terminated.detach().clone(),truncated=truncated.detach().clone())
    def finish(self,reward,next_observation):
        if self.pending is None or 'post' not in self.pending:raise RuntimeError('Missing pre-reset state')
        row=self.pending;row['reward']=reward.detach().clone();row['next_observation_returned']=next_observation.detach().clone()
        self.rows.append(row);self.pending=None
        if len(self.rows)>=self.limit:self.flush()
    def flush(self):
        if not self.rows:return
        def stack(values):
            if isinstance(values[0],dict):return {k:stack([v[k] for v in values]) for k in values[0]}
            return torch.stack(values).cpu()
        out=self.path/f'shard_{len(self.shards):05d}.pt';torch.save(stack(self.rows),out)
        self.shards.append(dict(path=str(out),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),steps=len(self.rows)));self.rows=[]
    def close(self):
        if self.pending is not None:raise RuntimeError('Capture ended with incomplete transition')
        self.flush();manifest=dict(**self.metadata,shards=self.shards,complete=True,scope='pre and terminal-post physical state; next_observation_returned is reset observation on done, never terminal target; stochastic RNG and PhysX solver hidden state are not serialized')
        (self.path/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest
