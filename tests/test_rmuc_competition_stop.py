import numpy as np
from rmuc_world_model.competition_stop import stop_metrics

def test_hold_and_reverse_drift_are_not_replaced_by_survival():
 t=np.arange(375)*.008;v=.5*np.exp(-8*t)
 def check(x,terminated=False):return stop_metrics(t,x,.5,.008,True,terminated,[1,1],.01,0)
 assert check(v)['passed']
 assert not check(v,True)['passed']
 v2=v.copy();v2[125:]=-0.06
 assert not check(v2)['checks']['no_sustained_reverse']
 v3=v.copy();v3[80]=-.16
 assert not check(v3)['checks']['reverse_speed']
 v4=v.copy();v4[:240]=.08
 assert not check(v4)['checks']['arrival']
