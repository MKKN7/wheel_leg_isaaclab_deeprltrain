import ast,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_world_model.observation import observe

def matrix(q):
 w,x,y,z=q.unbind(-1)
 return torch.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],-1).reshape(-1,3,3)

def test_against_current_env_source_noise_free_prefix():
 path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/env.py'
 tree=ast.parse(path.read_text());methods={n.name:n for cls in tree.body if isinstance(cls,ast.ClassDef) for n in cls.body if isinstance(n,ast.FunctionDef)}
 fn=methods['_get_observations'];cut=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='obs' for t in n.targets))
 fn.body=fn.body[:cut+1]+[ast.Return(ast.Name('obs',ast.Load()))]
 module=ast.fix_missing_locations(ast.Module(body=[methods['_state'],methods['_heading_forward_velocity'],fn],type_ignores=[]))
 scope={'torch':torch,'quat_apply':lambda q,v:(matrix(q)@v[...,None]).squeeze(-1),'quat_apply_inverse':lambda q,v:(matrix(q).transpose(-1,-2)@v[...,None]).squeeze(-1)}
 exec(compile(module,str(path),'exec'),scope)
 torch.manual_seed(33);n=128;root=torch.randn(n,13);root[:,3:7]/=root[:,3:7].norm(dim=-1,keepdim=True);pos=torch.randn(n,6);vel=torch.randn(n,6);zero=torch.randn(n,6);last=torch.randn(n,6);cmd=torch.randn(n,3);scale=torch.tensor([2.1,.5,1.]);g=torch.zeros(n,3);g[:,2]=-1
 wrap=lambda x:SimpleNamespace(torch=x)
 e=SimpleNamespace(robot=SimpleNamespace(data=SimpleNamespace(root_quat_w=wrap(root[:,3:7]),root_lin_vel_w=wrap(root[:,7:10]),root_ang_vel_w=wrap(root[:,10:13]),joint_pos=wrap(pos),joint_vel=wrap(vel))),_gravity_w=g,_current_command=lambda:cmd,_control_dof_ids=list(range(6)),_default_control_pos=zero,_policy_actions=last,_command_scale=scale,cfg=SimpleNamespace(command_observation_mode='tracking_error_with_target',command_observation_error_clip=2.,command_yaw_observation_error_clip=2.))
 e._state=lambda:scope['_state'](e);e._heading_forward_velocity=lambda v:scope['_heading_forward_velocity'](e,v)
 expected=scope['_get_observations'](e);actual=observe(root,pos,vel,zero,last,cmd,scale,2.,2.)
 torch.testing.assert_close(actual,expected,atol=3e-6,rtol=3e-6)
 assert actual.shape==(128,25)

def test_gradients_and_last_action_are_preserved():
 root=torch.zeros(2,13);root[:,3]=1;root.requires_grad_();last=torch.randn(2,6,requires_grad=True);pos=torch.randn(2,6,requires_grad=True)
 out=observe(root,pos,torch.ones(2,6),torch.zeros(2,6),last,torch.zeros(2,3),torch.tensor([2.1,.5,1.]),2.,2.)
 torch.testing.assert_close(out[:,-6:],last)
 out.sum().backward();assert torch.isfinite(root.grad).all();assert last.grad.eq(1).all();assert pos.grad[:,[0,1,3,4]].eq(1).all()

def test_sensor_delay_preserves_current_command_and_last_action():
    from rmuc_world_model.sensor import push_sensor,push_action,delayed_sensor_observation
    h=torch.randn(3,2,25,requires_grad=True);current=torch.randn(3,25,requires_grad=True);valid=torch.tensor([True,True,False]);delay=torch.tensor([0,1,1]);yaw=torch.tensor([0.,.2,-.3])
    out,updated=push_sensor(h,current,valid,delay,yaw)
    torch.testing.assert_close(out,delayed_sensor_observation(updated,delay,current,yaw))
    torch.testing.assert_close(out[:,7],current[:,7]);torch.testing.assert_close(out[:,19:],current[:,19:])
    torch.testing.assert_close(updated[2,0],current[2]);torch.testing.assert_close(updated[2,1],current[2])
    out[:,0].sum().backward();assert h.grad[1,0,0]==1 and current.grad[1,0]==0 and current.grad[0,0]==1
    ah=torch.zeros(3,2,6);a=torch.ones(3,6);delivered,queue=push_action(ah,a,delay)
    torch.testing.assert_close(delivered[0],a[0]);assert delivered[1:].count_nonzero()==0


def test_native_root_quaternion_order():
    from rmuc_world_model.observation import native_root_to_wxyz, rotate
    r=torch.zeros(2,13);r[:,3]=2**-.5;r[:,6]=2**-.5
    r[:,7:]=torch.arange(6)
    converted=native_root_to_wxyz(r)
    torch.testing.assert_close(converted[:,3:7],torch.tensor([[2**-.5,2**-.5,0.,0.]]).expand(2,-1))
    torch.testing.assert_close(converted[:,7:],r[:,7:])
    f=torch.tensor([[0.,0.,-1.]]).expand(2,-1)
    torch.testing.assert_close(rotate(converted[:,3:7],f),torch.tensor([[0.,1.,0.]]).expand(2,-1),atol=2e-7,rtol=0)
