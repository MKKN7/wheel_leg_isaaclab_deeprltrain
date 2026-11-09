import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/rmuc_world_model'))
from audit_dynamic_periodic_motion import analyze

def test_event_window_and_first_episode_failure(tmp_path):
 columns=['episode_step','vx','yaw_rate','pitch_deg','action_input_observation_3','wheel_target_right_nm','wheel_target_left_nm']+[f'joint_position_{i}' for i in range(6)]
 dt=.008;t=np.arange(2500)*dt;f=np.zeros((2500,2,len(columns)));f[:,:,0]=np.arange(2500)[:,None]
 f[:,0,3]=np.sin(2*np.pi*5*t);f[:,1,3]=np.sin(2*np.pi*5*t)
 f[625:,1,0]=np.arange(1875)
 np.savez(tmp_path/'policy.npz',frames=f,columns=columns);np.savez(tmp_path/'leg.npz',dt=dt)
 (tmp_path/'sequence.json').write_text(json.dumps([[0,0,0],[2,-.25,0],[6,0,0],[19,0,0]]))
 (tmp_path/'command.json').write_text(json.dumps({'command':['--command-sequence',str(tmp_path/'sequence.json'),'--episode-length-seconds','20']}))
 (tmp_path/'evaluation.json').write_text(json.dumps({'policy_trace':{'path':str(tmp_path/'policy.npz')},'leg_tracking_trace':{'path':str(tmp_path/'leg.npz')}}))
 rows=analyze(tmp_path)['rows']
 row=next(x for x in rows if x['env']==0 and x['start']==2)
 assert row['valid'] and row['metrics']['pitch_deg']['at_least_three_cycles']
 assert abs(row['metrics']['pitch_deg']['dominant_hz']-5)<.3
 assert not next(x for x in rows if x['env']==1 and x['start']==2)['valid']
 assert not next(x for x in rows if x['env']==0 and x['start']==19)['valid']
