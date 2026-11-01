import sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from stop_phase_sampling import stop_phase_labels,StopPhaseSampler


def test_stop_phases_and_reset_boundaries():
    c=torch.tensor([.75]*10+[0.]*300+[.25]*20+[0.]*300)
    p=stop_phase_labels(c)
    assert p[10]==3 and p[60]==4 and p[160]==5 and p[260]==-1
    assert p[330]==0 and p[380]==1 and p[480]==2
    assert (stop_phase_labels(torch.zeros(100))==-1).all()
    assert (stop_phase_labels(torch.linspace(.5,0,100))==-1).all()


def test_strata_cover_delays_and_rehearsal_without_invalid_starts():
    labels=torch.tensor([-1]*10+[0]*10+[1]*10+[2]*10+[3]*10+[4]*10+[5]*10)*1
    labels=labels.repeat(2);ctx=torch.zeros(140,14);ctx[70:,12:14]=1
    starts=torch.arange(4,136)
    r=SimpleNamespace(state=torch.zeros(140,2),context=ctx,starts=starts)
    s=StopPhaseSampler(r,labels);out=s.sample(256,torch.Generator().manual_seed(12))
    assert len(out)==256 and torch.isin(out,starts).all()
    assert (out<70).sum()==128 and (out>=70).sum()==128
    assert set(labels[out].tolist())==set(range(-1,6))
