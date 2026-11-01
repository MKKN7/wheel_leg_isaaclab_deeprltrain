"""Compare the actual optimized objective against joint gates without rerunning physics."""
import argparse,json,numpy as np
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('candidate',type=Path);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();results={}
for label,d in [('baseline',a.baseline),('candidate',a.candidate)]:
 r=json.loads((d/'evaluation.json').read_text());z=np.load(r['policy_trace']['path']);c={str(k):i for i,k in enumerate(z['columns'])};f=z['frames'];report=json.loads((d/'dynamic_domain_report.json').read_text());rows=[]
 for env in range(f.shape[1]):
  x=f[2125:2499,env];assert len(x)==374 and (x[:,c['action_input_observation_7']]==0).all()
  v=x[:,c['vx']];pitch=np.deg2rad(x[:,c['pitch_deg']]);active=x[:,c['joint_position_0']]-x[:,c['joint_position_3']];terms=np.stack([v*v,.2*pitch*pitch,.05*active*active,50*np.maximum(-v-.09,0)**2],-1)*.008;discount=.99**np.arange(len(x));cost=(terms*discount[:,None]).sum(0)
  gate=next(q for q in report['rows'] if q['env']==env and q['start']==17);e=gate['vx_event']
  rows.append(dict(env=env,cost=float(cost.sum()),components=cost.tolist(),overshoot_fail=e['overshoot_fraction']>=.2,arrival_fail=e['arrival_s'] is None or e['arrival_s']>e['arrival_limit_s'],speed_tail_fail=e['tail_speed_mae']>=.01,angle_tail_fail=max(e['tail_angle_p95_deg'])>2.,yaw_tail_fail=gate['yaw_tail_mae']>=.05,joint_pass=gate['full_dynamic_pass'],overshoot=e['overshoot_fraction'],arrival=e['arrival_s'],tail_angles=e['tail_angle_p95_deg'],yaw_tail=gate['yaw_tail_mae']))
 results[label]=rows
old=results['baseline'];new=results['candidate'];summary={}
for label,rr in results.items():summary[label]=dict(cost_mean=float(np.mean([r['cost'] for r in rr])),components_mean=np.mean([r['components'] for r in rr],0).tolist(),**{k:sum(bool(r[k]) for r in rr) for k in ['overshoot_fail','arrival_fail','speed_tail_fail','angle_tail_fail','yaw_tail_fail','joint_pass']})
changed=[dict(env=b['env'],old=a,new=b) for a,b in zip(old,new) if a['joint_pass']!=b['joint_pass']]
result=dict(summary=summary,changed_gate_cases=changed,rows=results,cost_component_names=['speed_squared','pitch_squared','active_difference_squared','reverse_rebound'],gamma=.99,discount_at_1p5s=float(.99**(1.5/.008)),discount_at_2p5s=float(.99**(2.5/.008)),objective_omissions=['yaw tracking/rate oscillation','hip difference','explicit joint pass/arrival hold','uniform-weight steady tail'],scope='Actual trajectory objective diagnostic; changes before17s may change stop initial state. Cannot identify model-gradient error solely from objective regression.',promoted=False)
(a.candidate/'objective_diagnosis.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['rows','changed_gate_cases']},indent=2))
