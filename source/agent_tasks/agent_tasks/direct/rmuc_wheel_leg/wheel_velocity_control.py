"""Discrete output-shaft wheel velocity controller and explicit legacy-P diagnostic bridge.

Kp is a control gain (N m / (rad/s)), not passive mechanical damping.
No controller is installed unless the diagnostic installer is explicitly called.
"""
import math
import torch

class WheelVelocityController:
    def __init__(self, template, dt, kp, ki=0., filter_tau_s=0., reference_limit_rad_s=80.):
        scalars=[dt,kp,ki,filter_tau_s,reference_limit_rad_s]
        if not all(math.isfinite(v) for v in scalars) or not (dt>0 and kp>0 and ki>=0 and filter_tau_s>=0 and reference_limit_rad_s>0):
            raise ValueError('Invalid finite SI wheel velocity controller parameters')
        self.dt=dt;self.kp=kp;self.ki=ki;self.filter_tau_s=filter_tau_s;self.reference_limit=reference_limit_rad_s
        self.integral_effort=torch.zeros_like(template)
        self.filtered_velocity=torch.zeros_like(template)
        self.initialized=torch.zeros_like(template,dtype=torch.bool)
        self.reference=torch.zeros_like(template)
        self.unsaturated_effort=torch.zeros_like(template)
        self.applied_effort=torch.zeros_like(template)

    def reset(self, ids):
        self.integral_effort[ids]=0
        self.filtered_velocity[ids]=0
        self.initialized[ids]=False
        self.reference[ids]=0
        self.unsaturated_effort[ids]=0
        self.applied_effort[ids]=0

    def step(self, reference, velocity, effort_limit, enabled=None, kp_override=None):
        if not torch.isfinite(effort_limit).all() or (effort_limit<0).any():
            raise ValueError('Output-shaft effort limits must be finite and nonnegative')
        if not torch.isfinite(reference).all() or not torch.isfinite(velocity).all():
            raise ValueError('Non-finite velocity or reference')
        kp=self.kp if kp_override is None else kp_override
        if isinstance(kp,torch.Tensor) and (not torch.isfinite(kp).all() or (kp<=0).any()):
            raise ValueError('Invalid sampled controller gain')
        active=torch.ones_like(self.initialized) if enabled is None else enabled.expand_as(self.initialized)
        target=reference.clamp(-self.reference_limit,self.reference_limit)
        alpha=1. if self.filter_tau_s==0 else -math.expm1(-self.dt/self.filter_tau_s)
        filtered=velocity if self.filter_tau_s==0 else self.filtered_velocity+alpha*(velocity-self.filtered_velocity)
        filtered=torch.where(self.initialized,filtered,velocity)
        error=target-filtered
        old_integral=self.integral_effort.clamp(-effort_limit,effort_limit)
        proposed=(old_integral+self.ki*self.dt*error).clamp(-effort_limit,effort_limit)
        raw=kp*error+proposed
        # Conditional integration: do not accumulate into saturation; allow unwinding.
        blocked=((raw>effort_limit)&(error>0))|((raw<-effort_limit)&(error<0))
        integral=torch.where(blocked,old_integral,proposed)
        raw=kp*error+integral
        effort=raw.clamp(-effort_limit,effort_limit)
        self.reference.copy_(torch.where(active,target,torch.zeros_like(target)))
        self.integral_effort.copy_(torch.where(active,integral,torch.zeros_like(integral)))
        self.filtered_velocity.copy_(filtered)
        self.initialized.copy_(active)
        self.unsaturated_effort.copy_(torch.where(active,raw,torch.zeros_like(raw)))
        self.applied_effort.copy_(torch.where(active,effort,torch.zeros_like(effort)))
        return self.applied_effort

def install_legacy_velocity_equivalence(actuator, dt):
    """Diagnostic re-expression of old torque feedforward minus D*velocity.

    omega_reference = old_clipped_feedforward / old_D, Kp=old_D, Ki=0.
    Controller runs each physics step. This proves interface equivalence only;
    it is NOT the target3Nm controller, a damping identification, or a new policy.
    """
    if hasattr(actuator,'_rmuc_effort_response') or hasattr(actuator,'_rmuc_velocity_bridge'):
        raise ValueError('Equivalence diagnostic requires unwrapped explicit wheel actuator')
    if (actuator.stiffness!=0).any() or (actuator.damping<=0).any():
        raise ValueError('Equivalence requires zero stiffness and positive D')
    controller=WheelVelocityController(actuator.damping,dt,.1,reference_limit_rad_s=1000.)
    original_reset=actuator.reset
    audit={'max_effort_error':torch.zeros((),device=actuator.damping.device),
           'max_reference':torch.zeros((),device=actuator.damping.device),'compute_calls':0}
    def compute(action,joint_pos,joint_vel):
        if action.joint_velocities is not None and (action.joint_velocities!=0).any():
            raise ValueError('Legacy equivalence requires zero old velocity target')
        gain=actuator.damping
        reference=action.joint_efforts/gain
        if (reference.abs()>controller.reference_limit).any():
            raise ValueError('Reference exceeds diagnostic equivalence domain')
        expected=(action.joint_efforts-gain*joint_vel).clamp(-actuator.effort_limit,actuator.effort_limit)
        output=controller.step(reference,joint_vel,actuator.effort_limit,kp_override=gain)
        audit['max_effort_error'].copy_(torch.maximum(audit['max_effort_error'],(output-expected).abs().max()))
        audit['max_reference'].copy_(torch.maximum(audit['max_reference'],reference.abs().max()))
        audit['compute_calls']+=1
        actuator.computed_effort=controller.unsaturated_effort
        actuator.applied_effort=output
        action.joint_efforts=output
        action.joint_positions=None;action.joint_velocities=None
        return action
    def reset(ids):
        original_reset(ids);controller.reset(ids)
    actuator.compute=compute;actuator.reset=reset
    actuator._rmuc_velocity_bridge=controller
    return controller,audit

def install_wheel_velocity_controller(actuator, dt, reference_provider, enabled_provider,
                                      kp, ki=0., filter_tau_s=0., reference_limit_rad_s=80.):
    """Explicit PI execution of velocity references supplied after transport delay.

    This bypasses IdealPD compute rather than adding PI torque on top of old D.
    Its actuator effort_limit is the actual motor torque cap at each physics step.
    """
    if hasattr(actuator,'_rmuc_effort_response') or hasattr(actuator,'_rmuc_velocity_bridge'):
        raise ValueError('Wheel velocity controller requires an unwrapped actuator')
    if (actuator.stiffness!=0).any() or (actuator.damping!=0).any():
        raise ValueError('Disable old PD gains before installing explicit velocity controller')
    controller=WheelVelocityController(actuator.damping,dt,kp,ki,filter_tau_s,reference_limit_rad_s)
    original_reset=actuator.reset
    audit={'compute_calls':0,'saturated_joint_steps':torch.zeros((),device=actuator.damping.device),
           'max_applied_effort_nm':torch.zeros((),device=actuator.damping.device),
           'max_reference_rad_s':torch.zeros((),device=actuator.damping.device)}
    def compute(action,joint_pos,joint_vel):
        requested=reference_provider()
        output=controller.step(requested,joint_vel,actuator.effort_limit,enabled_provider())
        actuator.computed_effort=controller.unsaturated_effort
        actuator.applied_effort=output
        action.joint_efforts=output;action.joint_positions=None;action.joint_velocities=None
        audit['compute_calls']+=1
        audit['saturated_joint_steps'].add_((output.abs()>=actuator.effort_limit*.999).sum())
        audit['max_applied_effort_nm'].copy_(torch.maximum(audit['max_applied_effort_nm'],output.abs().max()))
        audit['max_reference_rad_s'].copy_(torch.maximum(audit['max_reference_rad_s'],controller.reference.abs().max()))
        return action
    def reset(ids):
        original_reset(ids);controller.reset(ids)
    actuator.compute=compute;actuator.reset=reset
    actuator._rmuc_velocity_bridge=controller
    return controller,audit
