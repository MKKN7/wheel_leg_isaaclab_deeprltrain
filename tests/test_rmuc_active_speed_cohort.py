import runpy
from pathlib import Path
import torch,pytest
m=runpy.run_path(str(Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/active_speed_cohort.py'))
def test_cohort_preserves_other_groups_and_stable_assignment():
 speeds=m['validate_speeds']([1.7,1.8,2]);v=torch.ones(12);y=torch.ones(12);r=torch.zeros(12,dtype=torch.bool);c=r.clone();r[0]=True;c[1]=True
 a,b=m['apply_cohort'](v,y,r,c,speeds);assert torch.equal(a[:2],v[:2]);assert torch.equal(b[:2],y[:2]);assert not b[2:].any()
 x,z=m['apply_cohort'](v,y,torch.zeros_like(r),torch.zeros_like(c),speeds);assert torch.equal(a[2:],x[2:]);assert all(int((x==s).sum())==4 for s in speeds)
def test_invalid_or_disabled():
 for s in [[1.5,2],[1.7,1.7],[float('nan')],[True]]:
  with pytest.raises(ValueError):m['validate_speeds'](s)
 v=torch.ones(2);y=v.clone();r=torch.zeros(2,dtype=torch.bool);a,b=m['apply_cohort'](v,y,r,r,());assert a is v and b is y
