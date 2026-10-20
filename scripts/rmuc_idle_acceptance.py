"""User-confirmed zero-command gate, measured on complete ten-second windows."""
import argparse,json,math
from pathlib import Path
import numpy as np


def audit(report,world,legs,buckets,dt,settling_seconds=2.,duration_seconds=10.):
    world=np.asarray(world);legs=np.asarray(legs);buckets=np.asarray(buckets)
    if (world.ndim!=3 or world.shape[-1]!=14 or legs.shape!=(*world.shape[:2],8)
            or buckets.shape!=(world.shape[1],) or not np.isfinite(world).all()
            or not np.isfinite(legs).all() or not math.isfinite(dt) or dt<=0):
        raise ValueError('Finite synchronized world/leg traces and timebase required')
    if not np.allclose(world[:,:,4],legs[:,:,0],atol=1e-5,rtol=0):
        raise ValueError('World and joint trace episode ages differ')
    # Use synchronized pre-step joint/velocity/position states including both
    # endpoints. N states span (N-1)*dt; never splice across auto-reset.
    needed=math.ceil(duration_seconds/dt-1e-8)+1;settle=math.ceil(settling_seconds/dt-1e-8)
    rows=[]
    for bucket,raw in enumerate(report['rows']):
        if raw['vx']!=0 or raw['yaw']!=0:continue
        env_ids=np.flatnonzero(buckets==bucket);windows=[];missing=[]
        for env in env_ids:
            z=world[:,env];q=legs[:,env]
            cuts=[0]+(np.flatnonzero((z[1:,4]<=z[:-1,4]) | (z[:-1,10]!=0) | (z[:-1,11]!=0))+1).tolist()+[len(z)]
            found=0
            for start,end in zip(cuts[:-1],cuts[1:]):
                ids=np.flatnonzero(z[start:end,4]>=settle)+start
                if len(ids)<needed:continue
                idx=ids[:needed];w=z[idx];joint=q[idx]
                continuous=bool(np.all(np.diff(idx)==1) and np.allclose(np.diff(w[:,4]),1)
                                and np.allclose(w[:,9]-w[:,4],1) and np.all(w[:,12:14]==0))
                xy=w[:,:2];drift=float(np.linalg.norm(xy-xy[0],axis=1).max())
                absolute_speed=np.abs(w[:,3])
                speed=float(((absolute_speed[:-1]+absolute_speed[1:])*.5).mean())
                active=float(np.percentile(np.abs(np.rad2deg(joint[:,2]-joint[:,5])),95))
                hip=float(np.percentile(np.abs(np.rad2deg(joint[:,3]+joint[:,6])),95))
                checks=dict(continuous_window=continuous,no_termination=not bool(z[start:end,10].any()),
                    speed_mae_le_0p01=speed<=.01,max_xy_drift_le_0p05=drift<=.05,
                    active_p95_le_2=active<=2,hip_p95_le_2=hip<=2)
                windows.append(dict(env=int(env),start_step=int(idx[0]),duration_s=(needed-1)*dt,speed_mae_mps=speed,
                    maximum_xy_drift_m=drift,active_abs_p95_deg=active,hip_abs_p95_deg=hip,checks=checks,passed=all(checks.values())))
                found+=1
            if not found:missing.append(int(env))
        safe=raw['survival']==1 and bool(raw.get('termination_counts')) and not any(raw['termination_counts'].values())
        available_pass=bool(windows) and not missing and all(w['passed'] for w in windows) and safe
        checks=dict(all_replicas_have_ten_second_windows=not missing and bool(len(env_ids)),
                    all_windows_pass=bool(windows) and all(w['passed'] for w in windows),no_observed_failure=safe,
                    full_windows_at_least32=len(windows)>=32,completed_episodes_at_least32=raw['completed_episodes']>=32)
        rows.append(dict(vx=0,yaw=0,windows=windows,missing_envs=missing,checks=checks,
                         short_screen_pass=available_pass,passed=all(checks.values())))
    return dict(contract='user_confirmed_idle_strict_20261014',settling_seconds=settling_seconds,
        window_seconds=duration_seconds,rows=rows,passed=bool(rows) and all(r['passed'] for r in rows),
        short_screen_pass=bool(rows) and all(r['short_screen_pass'] for r in rows),promoted=False,
        scope='Each complete episode window; no full-domain/multiseed/deployment certification')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--experiment',type=Path,required=True);args=parser.parse_args();p=args.experiment
    r=json.loads((p/'evaluation.json').read_text());w=np.load(p/'evaluation.world_trace.npz');q=np.load(p/'evaluation.leg_tracking.npz')
    if float(w['dt'])!=float(q['dt']):raise ValueError('Trace dt mismatch')
    result=audit(r,w['frames'],q['frames'],q['bucket_ids'],float(q['dt']));result['checkpoint_sha256']=r['checkpoint_sha256']
    (p/'idle_acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print('Idle formal',result['passed'],'short screen',result['short_screen_pass'])
