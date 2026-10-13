"""Recovery-specific reward terms, independent of locomotion tracking rewards.

Versioned opt-in; weights are an experimental design, not behavior evidence.
"""
import torch

def linear_slew_schedule(start, end, step, total_steps):
    """Bounded scalar schedule used by recovery target slew annealing."""
    if total_steps <= 0:
        return float(end)
    alpha=min(1.,max(0.,float(step)/float(total_steps)))
    return float(start)+(float(end)-float(start))*alpha

def recovery_phase_progress(up, height, ready, hold_seconds, min_height, upright_cos, required_hold):
    """Measured phase, not a clock: turn upright -> rise -> hold. Can regress."""
    orientation=((up.clamp(-1.,1.)+1.)/(1.+upright_cos)).clamp(0.,1.)
    rise=(height/min_height).clamp(0.,1.)
    progress=.5*orientation
    progress=torch.where(up>upright_cos,.5+.3*rise,progress)
    return torch.where(ready,.8+.2*(hold_seconds/required_hold).clamp(0.,1.),progress).clamp(0.,1.)

def upright_without_side_income(up):
    """No static posture income at >=60 degrees tilt, including side/inverted."""
    return ((up-.5)/.5).clamp(0.,1.)

def supported_rise_shaping(up, height, supported, min_height, shaping_height=None):
    """Continuous approach reward before strict standing; zero at side/airborne."""
    target=min_height if shaping_height is None else shaping_height
    return supported.float()*upright_without_side_income(up)*(height/target).clamp(0.,1.)

def stable_pose_shaping(up, height, supported, planar_speed, angular_speed,
                        min_height, shaping_height, max_planar_speed=.15,
                        max_angular_speed=.3):
    """Continuous approach to the stable pose; strict ready/hold remains separate."""
    upright=upright_without_side_income(up)
    height_score=(height-(min_height-.04))/.08
    height_score=height_score.clamp(0.,1.)
    speed_score=torch.exp(-((planar_speed/max_planar_speed).square()
                            +(angular_speed/max_angular_speed).square()))
    return supported.float()*upright*height_score*speed_score

def joint_limit_barrier(joint_pos, joint_limits, margin=.15):
    """Smooth penalty before the hard recovery joint-limit termination."""
    distance=torch.minimum(joint_pos-joint_limits[...,0],joint_limits[...,1]-joint_pos)
    return ((margin-distance).clamp_min(0.)/margin).square().mean(-1)

def supported_braking_cost(up, any_wheel_contact, planar_speed, speed_scale=.3):
    """Height-independent bounded cost; no positive static side reward."""
    speed_sq=(planar_speed/speed_scale).square()
    return any_wheel_contact.float()*upright_without_side_income(up)*speed_sq/(1.+speed_sq)

def double_contact_acquired(previous_both, current_both):
    """One-shot transition signal; static side contact receives no income."""
    return (~previous_both & current_both).float()


def supported_wheel_gap_progress(previous_gap, gap, previous_valid, previous_supported,
                                supported, previous_up, up, side_case, scale=.3):
    """Signed wheel-height-gap progress, never static or takeoff income."""
    eligible=(previous_valid & previous_supported & supported & side_case
              & (previous_up>0.) & (up>0.))
    delta=(previous_gap-gap)/scale
    return torch.where(eligible,delta,torch.zeros_like(delta))

def contact_impact_penalty(contact_forces, soft_limit_n=300.):
    """Penalize violent contact while retaining ordinary wheel support."""
    magnitude=torch.linalg.vector_norm(contact_forces,dim=-1)
    return ((magnitude-soft_limit_n).clamp_min(0.)/soft_limit_n).mean(-1)

def refall_event(armed, bad_steps, ready_steps, fallen, arm_steps, dwell_steps):
    """One event per recovery; fresh/reset environments start unarmed."""
    armed=armed | (ready_steps>=arm_steps)
    bad_steps=torch.where(armed & fallen,bad_steps+1,torch.zeros_like(bad_steps))
    event=armed & (bad_steps>=dwell_steps)
    return armed & ~event,torch.where(event,torch.zeros_like(bad_steps),bad_steps),event

def standing_recovery_terms(up, height, supported, ready, hold_seconds,
                            min_height, upright_cos, required_hold):
    """Bounded standing scores; caller resets consecutive hold on loss of ready."""
    standing=supported & (up>upright_cos)
    return {
        'supported_height':standing.float()*(height/min_height).clamp(0.,1.),
        'height_reached':(standing & (height>min_height)).float(),
        'stable_hold':ready.float()*(hold_seconds/required_hold).clamp(0.,1.),
    }

DEFAULT_WEIGHTS = {
    'upright':3., 'upright_height':2., 'upright_progress':2.,
    'established':5., 'action_rate':-.1, 'leg_effort':-.05, 'time_cost':-.05,
}


def compute_recovery_rewards(up, height, established, previous_up, previous_valid,
                             action, previous_action, normalized_leg_effort, weights):
    """Per-control-step terms. Progress is zero on each environment's reset."""
    if set(weights)!=set(DEFAULT_WEIGHTS):
        raise ValueError('Recovery reward keys must exactly match declared terms')
    upright=up.clamp(-1.,1.)
    terms={
        'upright':(upright+1.)*.5,
        # Height helps after rotating toward upright; no height reward upside-down
        # or horizontal. This is attitude-conditioned, not a contact certificate.
        'upright_height':(height/.4).clamp(0.,1.)*upright.clamp_min(0.),
        'upright_progress':torch.where(previous_valid,upright-previous_up.clamp(-1.,1.),torch.zeros_like(up)),
        'established':established.float(),
        'action_rate':(action-previous_action).square().mean(-1),
        'leg_effort':normalized_leg_effort.square().mean(-1),
        'time_cost':torch.ones_like(up),
    }
    weighted={key:float(weights[key])*value for key,value in terms.items()}
    reward=torch.stack(list(weighted.values())).sum(0)
    return reward,weighted


def advance_recovery_stage(stage, streak, success, required, ceiling):
    """Training sampling only; ceiling is authorized by independent evaluation."""
    count=torch.where(success,streak+1,torch.zeros_like(streak))
    promote=(count>=required)&(stage<ceiling)
    return (stage+promote.long()).clamp_max(ceiling),torch.where(promote,torch.zeros_like(count),count.clamp_max(required))


def debounced_contact_event(dwell, paid, supported, required_steps):
    """One payment per reset after consecutive upright two-wheel support."""
    dwell=torch.where(supported,(dwell+1).clamp_max(required_steps),torch.zeros_like(dwell))
    event=(dwell>=required_steps)&~paid
    return dwell,paid|event,event.float()


def recovery_cohort_layout(num_envs, cohorts, *, device):
    """Repeat explicit (tilt degrees, direction index) pairs without RNG."""
    if not cohorts or any(len(pair)!=2 or any(type(v) is not int for v in pair)
            or pair[0] not in (0,5,15,30,45,60,75,90) or pair[1] not in (0,1,2,3) for pair in cohorts):
        raise ValueError('Cohorts require audited integer angle and direction pairs')
    table=torch.tensor(cohorts,device=device,dtype=torch.long)
    return table[torch.arange(num_envs,device=device)%len(cohorts)]
