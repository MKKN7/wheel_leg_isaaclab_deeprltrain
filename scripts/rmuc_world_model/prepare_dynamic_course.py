"""Build policy-tick-aligned vx AND yaw ramps using the existing sequence interface."""
import argparse,json
from pathlib import Path

def ramp_sequence(events,dt=.008,ramp_s=.8):
    if ramp_s<=0 or abs(round(ramp_s/dt)*dt-ramp_s)>1e-9:raise ValueError('Ramp must align with policy ticks')
    if not events or events[0][0]!=0:raise ValueError('Initial command required')
    out=[list(events[0])];previous=events[0]
    for event in events[1:]:
        start,v,w=event
        if start<previous[0]+(ramp_s if previous[0]>0 else 0):raise ValueError('Overlapping ramps')
        count=round(ramp_s/dt)
        for j in range(count+1):
            alpha=j/count
            out.append([round(start+j*dt,8),previous[1]+alpha*(v-previous[1]),previous[2]+alpha*(w-previous[2])])
        previous=event
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    cases={
      'yaw_slew_at_vx05':[[0,0,0],[2,.5,0],[5,.5,.3],[8,.5,.6],[11,.5,-.6],[14,.5,-.3],[17,0,0]],
      'joint_vx_yaw':[[0,0,0],[2,.5,.3],[5,1.,.6],[8,1.5,.8],[11,1.,-.6],[14,.5,-.3],[17,0,0]],
      'straight_reverse':[[0,0,0],[2,-.25,0],[5,-.4,0],[8,-.6,0],[11,-.8,0],[14,-.4,0],[17,0,0]],
      'spin_slew':[[0,0,0],[2,0,1.],[5,0,2.],[8,0,3.],[11,0,0.],[14,0,-1.],[17,0,0.]]}
    for name,events in cases.items():
        (a.output/(name+'_events.json')).write_text(json.dumps(events,indent=2))
        (a.output/(name+'_sequence.json')).write_text(json.dumps(ramp_sequence(events),indent=2))
    (a.output/'contract.json').write_text(json.dumps(dict(policy_dt=.008,ramp_s=.8,actor_obs=25,actor_action=6,current_commands_only=True,no_command_filter_added=True,scope='Prepared command trajectories; not run or accepted. Smooth tracking gates remain strict; spin3 is initial stage, final6-8 unfulfilled.',turning_symmetry_penalty=False,promoted=False),indent=2))
