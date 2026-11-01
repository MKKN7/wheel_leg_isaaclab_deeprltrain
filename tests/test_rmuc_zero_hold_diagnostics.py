"""Synthetic fixtures verify statistics, never serve as behavioral evidence."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/'artifacts/teacher_reward_audit_20261017/start_stop53_overshoot_20261024/audit_zero_hold.py'
spec=importlib.util.spec_from_file_location('zero_hold_audit_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

@pytest.mark.parametrize('speed,reset,command,passed',[(.004,False,0.,True),(.006,False,0.,False),(.004,True,0.,False),(.004,False,.1,False)])
def test_full_hold_rejects_drift_reset_and_nonzero_command(tmp_path,speed,reset,command,passed):
    age=np.arange(3000,dtype=float);dt=.008
    if reset:age[2500:]-=2500
    leg=np.zeros((3000,1,8));leg[:,0,0]=age;leg[:,0,1]=speed
    world=np.zeros((3000,1,5));world[:,0,0]=np.arange(3000)*dt*speed;world[:,0,3]=command
    np.savez(tmp_path/'leg.npz',frames=leg,dt=dt)
    np.savez(tmp_path/'world.npz',frames=world,columns=np.array(['pre_x','pre_y','terminated','vx_command','yaw_command']))
    (tmp_path/'sequence.json').write_text(json.dumps([[0,.35,0],[12,0,0]]))
    (tmp_path/'evaluation.json').write_text(json.dumps(dict(runtime_contract={'episode_length_s':24},leg_tracking_trace={'path':str(tmp_path/'leg.npz')},world_trajectory={'trace_path':str(tmp_path/'world.npz')})))
    result=module.audit(tmp_path)
    assert result['passed']==int(passed)
