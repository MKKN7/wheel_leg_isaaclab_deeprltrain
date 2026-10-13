import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from rmuc_turn_extrema import TurnExtrema
from rmuc_turn_diagnostics import COLUMNS

def test_extrema_do_not_hide_single_sample_and_streak_does_not_cross_reset():
    a=TurnExtrema(2,'cpu',.008)
    m=torch.zeros(2,len(COLUMNS));m[:,COLUMNS.index('min_normal_load_fraction')]=.4
    m[:,COLUMNS.index('quasistatic_support_margin_proxy_m')]=.1
    ready=torch.tensor([True,False]);forces=torch.full((2,2),20.)
    a.update(m,forces,ready,torch.zeros(2,dtype=torch.bool))
    m[0,COLUMNS.index('wheel_center_longitudinal_slip_mps')]=.3
    forces[0,0]=0
    a.update(m,forces,ready,torch.tensor([True,False]))
    a.update(m,forces,ready,torch.zeros(2,dtype=torch.bool))
    r=a.report(torch.tensor([True,False]))['steady']
    assert r['extrema']['wheel_center_longitudinal_slip_mps']['max']>.29
    assert abs(r['events']['not_both_wheels_above_2N']['sample_fraction']-2/3)<1e-6
    assert abs(r['events']['not_both_wheels_above_2N']['longest_observed_duration_s']-.008)<1e-7
    assert a.report(torch.tensor([False,True]))['steady']['extrema'] is None

def test_invalid_samples_not_reported_as_zero_and_support_requires_contacts():
    a=TurnExtrema(1,'cpu',.01)
    m=torch.zeros(1,len(COLUMNS));m[:,COLUMNS.index('quasistatic_support_margin_proxy_m')]=-.1
    a.update(m,torch.zeros(1,2),torch.ones(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
    r=a.report(torch.ones(1,dtype=torch.bool))['all_steps']
    assert r['events']['negative_support_proxy_with_two_contacts']['sample_fraction']==0
    m[:]=float('nan')
    a.update(m,torch.zeros(1,2),torch.ones(1,dtype=torch.bool),torch.zeros(1,dtype=torch.bool))
    assert a.report(torch.ones(1,dtype=torch.bool))['all_steps']['samples']==1
