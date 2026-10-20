import importlib.util
from pathlib import Path
import torch
import pytest
p=Path(__file__).resolve().parents[1]/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/random_speed_sequence.py'
s=importlib.util.spec_from_file_location('random_speed_schedule_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
RandomSpeedSequence=m.RandomSpeedSequence


def test_seeded_bounded_random_schedules_and_partial_reset():
    torch.manual_seed(15)
    a=RandomSpeedSequence(256,'cpu',.008,16,(0,3.2));a.reset(torch.arange(256))
    assert (a.targets>=0).all() and (a.targets<=3.2).all()
    assert torch.diff(a.targets,dim=1).abs().max()<=1.500001
    assert (a.ramps==0).any() and (a.ramps>0).any()
    saved=a.targets.clone();a.reset(torch.tensor([2,4]));assert torch.equal(saved[5:],a.targets[5:])
    torch.manual_seed(15)
    b=RandomSpeedSequence(256,'cpu',.008,16,(0,3.2));b.reset(torch.arange(256))
    assert torch.equal(saved,b.targets)


def test_reads_are_pure_and_ramp_endpoint_exact():
    a=RandomSpeedSequence(1,'cpu',.008,16,(0,.5))
    a.starts[0]=torch.arange(a.slots)*250;a.targets[0]=.5;a.targets[0,0]=0;a.ramps[0,1]=100
    rng=torch.get_rng_state().clone()
    for age,want in [(249,0),(250,0),(300,.25),(350,.5),(450,.5)]:
        cmd=a.commands(torch.tensor([age]));assert abs(float(cmd[0,0])-want)<1e-6
        assert not cmd[:,1:].any()
    assert torch.equal(rng,torch.get_rng_state())


def test_step_boundary_and_reset_age():
    a=RandomSpeedSequence(1,'cpu',.008,16,(0,.5));a.starts[0]=torch.arange(a.slots)*250
    a.targets[0]=.5;a.targets[0,0]=0
    assert a.commands(torch.tensor([249]))[0,0]==0
    assert a.commands(torch.tensor([250]))[0,0]==.5
    assert a.commands(torch.tensor([0]))[0,0]==0


@pytest.mark.parametrize('kwargs',[{'max_delta':1.6},{'hold_s':(0,1)},{'ramp_s':(2,3)},{'step_fraction':1.1},{'speed_range':(-1.6,1)}])
def test_invalid_schedule_rejected(kwargs):
    options=dict(count=1,device='cpu',dt=.008,episode_s=16,speed_range=(0,.5));options.update(kwargs)
    with pytest.raises(ValueError):RandomSpeedSequence(**options)


def test_snapshot_replays_and_is_not_mutated_by_reset():
    a=RandomSpeedSequence(8,'cpu',.008,16,(0,.5));a.reset(torch.arange(8))
    saved=a.snapshot();copy={k:v.clone() for k,v in saved.items()}
    b=RandomSpeedSequence(8,'cpu',.008,16,(0,.5))
    for k,v in saved.items():setattr(b,k,v)
    ages=torch.tensor([0,100,300,500,800,1100,1500,1900])
    torch.testing.assert_close(a.commands(ages),b.commands(ages))
    a.reset(torch.arange(8))
    for k in saved:assert torch.equal(saved[k],copy[k])


def test_isolated_schedule_does_not_perturb_global_rng_across_lengths():
    torch.manual_seed(66)
    before = torch.get_rng_state().clone()
    for hold in [(2., 4.), (1.6, 2.4)]:
        a = RandomSpeedSequence(32, 'cpu', .008, 16, (0, .5), hold_s=hold, isolated_rng=True)
        a.reset(torch.arange(32)); a.reset(torch.tensor([1, 5]))
        assert torch.equal(before, torch.get_rng_state())


def test_isolated_stream_reproducible_despite_unrelated_draws():
    torch.manual_seed(66)
    a = RandomSpeedSequence(32, 'cpu', .008, 16, (0, .5), isolated_rng=True)
    b = RandomSpeedSequence(32, 'cpu', .008, 16, (0, .5), isolated_rng=True)
    a.reset(torch.arange(32)); torch.rand(1000); b.reset(torch.arange(32))
    for k, v in a.snapshot().items():
        assert torch.equal(v, b.snapshot()[k])


def test_signed_course_contains_stops_and_bounded_smooth_reversals():
    torch.manual_seed(57)
    a=RandomSpeedSequence(128,'cpu',.008,16,(-.1,.3),step_fraction=0,max_delta=.4,isolated_rng=True)
    a.reset(torch.arange(128))
    assert (a.targets>=-.100001).all() and (a.targets<=.300001).all()
    assert (a.targets<0).any() and (a.targets==0).any() and (a.targets>0).any()
    assert torch.diff(a.targets,dim=1).abs().max()<=.400001
    previous=a.commands(torch.zeros(128,dtype=torch.long))
    for step in range(1,2000):
        current=a.commands(torch.full((128,),step,dtype=torch.long))
        assert not current[:,1:].any()
        assert (current[:,0]-previous[:,0]).abs().max()<=.4/50+1e-6
        previous=current


def test_optional_stop_sampling_preserves_rng_and_forward_courses():
    torch.manual_seed(58)
    for bounds in [(0,.3),(-.25,.3)]:
        a=RandomSpeedSequence(512,'cpu',.008,16,bounds,max_delta=.6,isolated_rng=True)
        b=RandomSpeedSequence(512,'cpu',.008,16,bounds,max_delta=.6,isolated_rng=True,stop_probability=.3)
        rng=torch.get_rng_state().clone();ids=torch.arange(512);a.reset(ids);b.reset(ids)
        assert torch.equal(rng,torch.get_rng_state())
        assert torch.equal(a.starts,b.starts) and torch.equal(a.ramps,b.ramps)
        if bounds[0]==0:assert torch.equal(a.targets,b.targets)
        else:
            assert (b.targets==0).float().mean()>(a.targets==0).float().mean()+.15
            assert torch.diff(b.targets,dim=1).abs().max()<=.600001
    with pytest.raises(ValueError):RandomSpeedSequence(1,'cpu',.008,16,(-.1,.3),stop_probability=.7)
