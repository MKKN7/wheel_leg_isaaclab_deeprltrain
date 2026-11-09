from pathlib import Path
import runpy,torch,pytest
from rmuc_target_policy_gradient import target_gate_from_commands
m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/active_command_cohort.py'))
def test_signed_targets_and_rehearsal_untouched():
 t=m['validate_targets']([[1.5,-.45],[.225,0]])
 v=torch.ones(4);w=torch.ones(4);r=torch.tensor([0,0,1,0],dtype=torch.bool);c=torch.tensor([0,0,0,1],dtype=torch.bool)
 a,b=m['assign'](v,w,r,c,t,True);torch.testing.assert_close(a,torch.tensor([1.5,.225,1,1]));torch.testing.assert_close(b,torch.tensor([-.45,0,1,1]))
 obs=torch.zeros(4,25);obs[:,7]=a*.5;obs[:,2]=.2;obs[:,8]=b-.8
 assert target_gate_from_commands(obs,t).tolist()==[1.,1.,0.,0.]
 assert m['assign'](v,w,r,c,())[0] is v
 with pytest.raises(ValueError):m['validate_targets']([[1,0],[1,0]])
