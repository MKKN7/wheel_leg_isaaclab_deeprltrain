import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from rmuc_world_model.replay import valid_starts,sample
from rmuc_world_model.dynamics import Dynamics

def test_windows_do_not_cross_resets_and_actions_align():
    age=np.array([1,2,3,4,5,1,2,3,4,5])
    assert valid_starts(age,2,2).tolist()==[2,3,7,8]
    q=dict(state=torch.arange(10.).view(-1,1),action=torch.arange(10.).view(-1,1).expand(-1,6),context=torch.zeros(2),starts=torch.tensor([3]))
    s,ah,a,target,c=sample([q],1,2,2,torch.Generator().manual_seed(1))
    assert s.flatten().tolist()==[1,2] and a[0,:,0].tolist()==[3,4] and target.flatten().tolist()==[3,4]
    assert ah[0,:,0].tolist()==[1,2]

def test_rollout_gradients_and_no_teacher_forcing():
    torch.manual_seed(3);m=Dynamics(torch.zeros(2),torch.ones(2),torch.ones(2),torch.zeros(1),torch.ones(1),history=2,width=16)
    torch.nn.init.normal_(m.net[-1].weight,std=.1)
    a=torch.randn(3,4,6,requires_grad=True);s=torch.zeros(3,2,2);ah=torch.zeros(3,2,6);c=torch.zeros(3,1)
    y=m.rollout(s,ah,a,c);y[:,-1].sum().backward()
    assert torch.isfinite(a.grad).all() and a.grad[:,0].abs().sum()>0
    assert y.shape==(3,4,2)

def test_capture_keeps_terminal_and_reset_observations_distinct(tmp_path):
    from rmuc_world_model.capture import ReplayCapture
    writer=ReplayCapture(tmp_path/'replay',{'obs':25},shard_steps=1)
    obs=torch.zeros(2,25);action=torch.zeros(2,6);state={'q':torch.ones(2,3)}
    writer.begin(state,obs,action);obs.fill_(8)
    writer.terminal({'q':torch.full((2,3),2.)},torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool))
    writer.finish(torch.ones(2),torch.full((2,25),9.));manifest=writer.close()
    row=torch.load(manifest['shards'][0]['path'],weights_only=True)
    assert row['observation'].max()==0
    assert row['post']['q'].min()==2
    assert row['next_observation_returned'].min()==9
    assert row['terminated'][0,0]

def test_rich_packing_uses_post_fields_without_episode_time_or_world_xy():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
    from rich_replay import DYNAMIC_FIELDS,pack_post
    post={k:torch.zeros(2,3,2) for k in DYNAMIC_FIELDS}
    post['root_state_w']=torch.zeros(2,3,13)
    post['root_state_w'][...,0]=123;post['root_state_w'][...,1]=-456
    post['joint_pos']=torch.zeros(2,3,4);post['joint_pos'][...,0]=torch.pi/2
    post['episode_step']=torch.full((2,3),9876)
    result,names=pack_post(post)
    assert result.shape[:2]==(2,3) and len(names)==result.shape[-1]
    assert not any('episode_step' in s for s in names)
    assert result.abs().max()<=1
    torch.testing.assert_close(result[...,names.index('joint_pos:0')],torch.ones(2,3))

def test_time_varying_context_uses_current_action_time():
    q=dict(state=torch.arange(10.).view(-1,1),action=torch.zeros(10,6),context=torch.arange(10.).view(-1,1),starts=torch.tensor([3]))
    *_,context=sample([q],1,2,2,torch.Generator().manual_seed(1))
    assert context.flatten().tolist()==[3.,4.]
    class ContextDynamics(Dynamics):
        def forward(self,states,history,action,ctx):return states[:,-1]+ctx
    m=ContextDynamics(torch.zeros(1),torch.ones(1),torch.ones(1),torch.zeros(1),torch.ones(1),2,8)
    out=m.rollout(torch.zeros(1,2,1),torch.zeros(1,2,6),torch.zeros(1,2,6),context)
    assert out.flatten().tolist()==[3.,7.]
