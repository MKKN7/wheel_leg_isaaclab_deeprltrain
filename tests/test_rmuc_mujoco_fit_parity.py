import importlib.util,json,sys
from pathlib import Path
import numpy as np
import torch
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'sim2real_RL/sim2sim'))
from wheel_passive import passive_components,feedback_damping_torque,validate_wheel_contract
spec=importlib.util.spec_from_file_location('isaac_fit_test',ROOT/'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/real2sim_wheel.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

def contract():
 return dict(wheel_torque_contract='separated_passive_v1',wheel_feedback_damping_mode='active_motor_v1',hardware_sign_verified=False,wheel_feedback_damping_nm_s_rad=.05,wheel_mechanical_viscous_nm_s_rad=.002,wheel_coulomb_friction_nm=.001,wheel_coulomb_omega_eps_rad_s=.1,wheel_fit_profile=json.loads((ROOT/'artifacts/real2sim_wheel_20261014/fit_compensated.json').read_text()))

def test_fit_feedback_and_passive_parity_both_sides_and_saturation():
 c=contract();profile=c['wheel_fit_profile'];omega=torch.tensor([[a,b] for a in [-60,-.1,0,.1,60] for b in [-60,-.1,0,.1,60]],dtype=torch.float64)
 plant=mod.Real2SimWheel(torch.zeros_like(omega),.002,profile)
 np.testing.assert_allclose(feedback_damping_torque(c,omega.numpy()),plant.feedback(omega).numpy(),atol=1e-12)
 v,f=passive_components(c,omega.numpy());expected=-omega*torch.tensor(profile['mechanical_viscous_nm_s_rad'])-torch.tanh(omega/.1)*torch.tensor(profile['coulomb_friction_nm'])
 np.testing.assert_allclose(v+f,expected.numpy(),atol=1e-8)
 for a in [-1.7,-1,0,1,1.7]:
  motor=np.clip(-5.5*np.clip(a,-1,1)+feedback_damping_torque(c,omega.numpy()),-4.89,4.89)
  reference=torch.clamp(-5.5*max(-1,min(a,1))+plant.feedback(omega),-4.89,4.89)
  np.testing.assert_allclose(motor,reference.numpy(),atol=1e-12)
 assert np.max(np.abs(motor+v+f))>4.89 # Mechanical drag must not be outer-clipped.

def test_nonzero_delay_rejected_until_implemented():
 c=contract();c['wheel_fit_profile']['delay_s']=[.002,.002];c['wheel_fit_profile']['delay_mode']='motor_transport'
 with pytest.raises(ValueError,match='zero added delay'):validate_wheel_contract(c)
