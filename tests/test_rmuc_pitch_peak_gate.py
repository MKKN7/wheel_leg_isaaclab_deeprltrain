import numpy as np
from scripts.rmuc_world_model.audit_pitch_peak_gate import pitch_peak_gate


def test_peak_spike_cannot_hide_behind_percentile_or_offset():
    t=np.arange(1500)*.008;x=np.full(len(t),5.)
    x[900]=8.01
    assert np.percentile(x[750:],95)-np.percentile(x[750:],5)==0
    assert not pitch_peak_gate(t,x,complete=True)['passed']
    assert not pitch_peak_gate(t,x-5,complete=True)['passed']


def test_inclusive_three_degree_limit_and_trend_not_removed():
    t=np.arange(1500)*.008;x=np.zeros(len(t));x[1000]=3.
    assert pitch_peak_gate(t,x,complete=True)['passed']
    x[750:]=np.linspace(0,3.1,750)
    assert not pitch_peak_gate(t,x,complete=True)['passed']


def test_missing_or_fallen_windows_never_pass_as_small_amplitude():
    t=np.arange(1500)*.008;x=np.zeros(len(t))
    assert not pitch_peak_gate(t,x,complete=False)['passed']
    assert not pitch_peak_gate(t[:1400],x[:1400],complete=True)['passed']
