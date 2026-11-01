"""Replay privileged terminal snapshots without giving them to the actor."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from replay import load_report

DYNAMIC_FIELDS = ['root_state_w','joint_pos','joint_vel','applied_torque',
                  'contact_forces_w','policy_actions','obs_history','action_history',
                  'leg_history','wheel_delay_history','wheel_motor_output',
                  'motor_leg_motors_effort','motor_wheel_motors_effort']
CONTEXT_FIELDS = ['sensor_delay_cycles','action_delay_cycles','leg_delay_active','wheel_transport_active',
                  'motor_leg_motors_tau','motor_wheel_motors_tau','_wheel_feedback_damping_per_env',
                  '_wheel_mechanical_viscous_per_env','_wheel_coulomb_per_env','_default_control_pos']

def pack_post(post):
    """Keep first16 common metrics elsewhere. Here add physical/controller latents."""
    parts=[];names=[]
    for k in DYNAMIC_FIELDS:
        v=post[k].float()
        if k=='root_state_w':v=v[...,3:] # world x/y and duplicate height are not needed
        if k=='joint_pos':v=torch.cat([v.sin(),v.cos()],-1) # do not regress wrapped/unbounded wheel angles
        v=v.flatten(2);parts.append(v);names.extend([f'{k}:{i}' for i in range(v.shape[-1])])
    return torch.cat(parts,-1),names

def load_capture(directory,history=4,horizon=32,rich=True,condition_commands=False):
    directory=Path(directory);seq,meta=load_report(directory/'evaluation.json',history,horizon)
    manifest=json.loads((directory/'replay/manifest.json').read_text())
    if not manifest['complete']:raise ValueError('Incomplete capture')
    chunks=[];contexts=[];ages=[];acts=[];obs=[];names=None
    for shard in manifest['shards']:
        p=Path(shard['path'])
        if hashlib.sha256(p.read_bytes()).hexdigest()!=shard['sha256']:raise ValueError('Replay hash mismatch')
        s=torch.load(p,map_location='cpu',weights_only=True)
        if not torch.equal(s['post']['episode_step'],s['pre']['episode_step']+1):raise ValueError('State timing mismatch')
        packed,cols=pack_post(s['post'])
        if names is not None and cols!=names:raise ValueError('Schema drift')
        names=cols;chunks.append(packed);ages.append(s['post']['episode_step']);acts.append(s['raw_action']);obs.append(s['observation'])
        contexts.append(torch.cat([s['post'][k].float().reshape(*s['post']['episode_step'].shape,-1) for k in CONTEXT_FIELDS],-1))
    x=torch.cat(chunks);age=torch.cat(ages);a=torch.cat(acts);c=torch.cat(contexts);o=torch.cat(obs)
    if not torch.isfinite(x).all():raise ValueError('Nonfinite privileged replay')
    dynamic=None
    if condition_commands:
        report=json.loads((directory/'evaluation.json').read_text());world=report['world_trajectory'];path=Path(world['trace_path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=world['trace_sha256']:raise ValueError('World command trace hash mismatch')
        data=np.load(path);cols={str(k):i for i,k in enumerate(data['columns'])}
        command=torch.tensor(data['frames'][:,:,[cols['vx_command'],cols['yaw_command']]],dtype=torch.float32)
        spec=json.loads((directory/'command.json').read_text());argv=spec.get('command',spec.get('argv'))
        profile_path=Path(argv[argv.index('--command-leg-scale-profile')+1]);profile=json.loads(profile_path.read_text())
        scale=np.interp(command[...,0].abs().numpy(),profile['speed_abs_mps'],profile['scale_rad'])
        dynamic=torch.cat([command,torch.tensor(scale,dtype=torch.float32)[...,None]],-1)
        meta['command_conditioning']={'fields':['current_vx','current_yaw','leg_scale_rad'],'world_trace_sha256':world['trace_sha256'],'leg_scale_profile_sha256':hashlib.sha256(profile_path.read_bytes()).hexdigest(),'scope':'Training-only dynamics inputs, current command at action execution; actor interface unchanged'}
    for q in seq:
        env=q['env'];n=len(q['state'])
        if not torch.equal(a[:n,env],q['action']):raise ValueError('Action alignment error')
        # DR readback and queues may be redrawn at reset. Truncate context at same first-episode boundary.
        if not torch.allclose(c[:n,env],c[0,env].expand(n,-1),atol=0,rtol=0):raise ValueError('Unexpected changing per-episode context')
        if rich:q['state']=torch.cat([q['state'],x[:n,env]],-1)
        q['context']=torch.cat([q['context'],c[0,env]])
        q['observations']=o[:n,env]
        if dynamic is not None:q['context']=torch.cat([q['context'][None].expand(n,-1),dynamic[:n,env]],-1)
    meta.update(capture_manifest=str((directory/'replay/manifest.json').resolve()),capture_manifest_sha256=hashlib.sha256((directory/'replay/manifest.json').read_bytes()).hexdigest(),rich=rich,extra_state_names=names if rich else [],context_fields=CONTEXT_FIELDS,scope='Privileged fields from post-before-reset snapshots. Physics solver caches/RNG/context not exhaustive; actor still25D single frame.')
    return seq,meta
