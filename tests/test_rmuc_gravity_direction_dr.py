import importlib.util
from pathlib import Path
import torch
import pytest
path=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/gravity_direction_dr.py'
s=importlib.util.spec_from_file_location('gravity_dr',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);DR=m.GravityDirectionDR

def test_disabled_identity_and_rng():
    x=torch.randn(12,25);rng=torch.random.get_rng_state().clone();d=DR(12,'cpu');d.reset(torch.arange(12));assert d.apply(x,.008) is x;assert torch.equal(rng,torch.random.get_rng_state())

def test_rotation_norm_clean_channels_and_repeatability():
    x=torch.randn(2048,25);x[:,3:6]=torch.nn.functional.normalize(x[:,3:6],dim=-1)
    a=DR(len(x),'cpu',strength=1);b=DR(len(x),'cpu',strength=1);ids=torch.arange(len(x));a.reset(ids);b.reset(ids)
    for _ in range(8):
        y=a.apply(x,.008);z=b.apply(x,.008);assert torch.equal(y,z)
    assert 800<int(a.active.sum())<1200
    assert torch.equal(x[~a.active],y[~a.active]);assert torch.equal(x[:,:3],y[:,:3]);assert torch.equal(x[:,6:],y[:,6:])
    assert torch.allclose(y[:,3:6].norm(dim=-1),x[:,3:6].norm(dim=-1),atol=2e-7)
    assert (y[a.active,3:6]-x[a.active,3:6]).abs().max()>0
    a.reset(ids[:20]);assert not a.drift[:20].any()
    assert a.readback()['calls']==8

def test_actual_tilt_retained_without_measurement_error():
    d=DR(2,'cpu',strength=1,probability=1,noise_deg=0,bias_deg=0,walk_deg_sqrt_s=0);d.reset(torch.arange(2));x=torch.zeros(2,25);x[:,3:6]=torch.tensor([[0,0,-1],[.5,0,-.8660254]]);assert torch.equal(d.apply(x,.008),x)
    with pytest.raises(ValueError):DR(2,'cpu',strength=2)
