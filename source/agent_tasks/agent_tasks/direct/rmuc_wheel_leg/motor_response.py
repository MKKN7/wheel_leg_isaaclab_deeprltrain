"""First-order applied-effort response, separate from command transport delay."""
import math
import torch


class EffortResponse:
    def __init__(self, template, dt, minimum_s, maximum_s):
        if not all(math.isfinite(v) for v in [dt,minimum_s,maximum_s]) or not (dt>0 and 0<minimum_s<=maximum_s):
            raise ValueError('Finite positive physicsdt and ordered motor time constants required')
        self.dt=dt;self.minimum_s=minimum_s;self.maximum_s=maximum_s
        self.time_constant=torch.zeros_like(template)
        self.effort=torch.zeros_like(template)

    def sample(self, env_ids, strength):
        if not math.isfinite(strength) or not 0<=strength<=1:raise ValueError('Motor response DR strength must be in[0,1]')
        if strength==0:
            self.time_constant[env_ids]=0
        else:
            shape=self.time_constant[env_ids].shape
            self.time_constant[env_ids]=torch.empty(shape,device=self.effort.device,dtype=self.effort.dtype).uniform_(
                self.minimum_s,self.maximum_s)*strength

    def reset(self, env_ids):
        self.effort[env_ids]=0

    def step(self, requested, effort_limit):
        tau=self.time_constant
        # Exact discretization for piecewise-constant desired effort. Zero tau
        # bypasses subtraction/addition so the legacy path is bitwise identical.
        alpha=-torch.expm1(-self.dt/tau.clamp_min(1e-12))
        filtered=self.effort+alpha*(requested-self.effort)
        output=torch.where(tau>0,filtered,requested)
        output=torch.maximum(torch.minimum(output,effort_limit),-effort_limit)
        self.effort.copy_(output)
        return self.effort


def install_effort_response(actuator, dt, minimum_s, maximum_s):
    """Wrap only explicit motor compute/reset; preserve its PD and hard limits."""
    if hasattr(actuator,'_rmuc_effort_response'):raise ValueError('Motor response already installed')
    response=EffortResponse(actuator.damping,dt,minimum_s,maximum_s)
    original_compute=actuator.compute;original_reset=actuator.reset
    def compute(control_action,joint_pos,joint_vel):
        action=original_compute(control_action,joint_pos,joint_vel)
        actuator.applied_effort=response.step(action.joint_efforts,actuator.effort_limit)
        action.joint_efforts=actuator.applied_effort
        return action
    def reset(env_ids):
        original_reset(env_ids)
        response.reset(env_ids)
    actuator.compute=compute;actuator.reset=reset
    actuator._rmuc_effort_response=response
    return response
