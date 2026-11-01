import importlib.util
from pathlib import Path
import numpy as np
p=Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model/audit_periodic_motion.py'
s=importlib.util.spec_from_file_location('periodic',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
def test_sustained_frequency_and_decay_are_distinguished():
 t=np.arange(750)*.008;x=np.sin(2*np.pi*2*t)
 a=m.periodic_metrics(x,.008);b=m.periodic_metrics(x*np.exp(-t),.008)
 assert abs(a['dominant_hz']-2)<.18
 assert a['band_power_fraction_near_peak']>.9
 assert .95<a['last_first_amplitude_ratio']<1.05
 assert b['last_first_amplitude_ratio']<.1

def test_trend_and_invalid_not_oscillation():
 t=np.arange(750)*.008
 a=m.periodic_metrics(2*t+3,.008)
 assert a['rms']<1e-10
 assert a['band_power_fraction_near_peak']==0
 assert not m.periodic_metrics([1,2],.008)['valid']
 assert not m.periodic_metrics(np.full(750,np.nan),.008)['valid']
