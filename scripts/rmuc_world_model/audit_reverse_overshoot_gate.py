"""User-authorized reverse acceptance: underspeed25%, overshoot at most .05m/s."""
import argparse,json
from pathlib import Path
import numpy as np

def reverse_speed_checks(vx,target,steady_mask,complete):
    v=np.asarray(vx,dtype=float);steady=np.asarray(steady_mask,dtype=bool)
    if target>=0 or v.ndim!=1 or steady.shape!=v.shape or not np.isfinite(v).all():
        raise ValueError('Finite reverse trace and matching steady mask required')
    excess=float(np.maximum(-v-abs(target),0).max()) if len(v) else None
    speed=float(-v[steady].mean()) if steady.any() else None
    return dict(maximum_reverse_overshoot_mps=excess,steady_backward_speed_mps=speed,
                overshoot_pass=bool(complete and excess is not None and excess<=.05),
                underspeed_pass=bool(complete and speed is not None and speed>=.75*abs(target)),
                complete=bool(complete))

def reverse_angle_check(angle_p95_deg, limit_deg=4.):
    """Reverse-only engineering tolerance, authorized 2026-11-07.

    This does not change pitch fluctuation, forward alignment, or survival.
    Keep the original strict two-degree result alongside the relaxed result.
    """
    values=np.asarray(angle_p95_deg,dtype=float)
    if not np.isfinite(limit_deg) or limit_deg<=0:
        raise ValueError('Finite positive reverse leg angle limit required')
    return bool(values.shape==(2,) and np.isfinite(values).all()
                and (values>=0).all() and (values<=limit_deg).all())

def analyze(directory, leg_limit_deg=4., output_filename='reverse_overshoot_gate.json'):
    p=Path(directory);report=json.loads((p/'evaluation.json').read_text())
    strict=json.loads((p/'strict_screen.json').read_text())['rows']
    pitch={r['env']:r for r in json.loads((p/'pitch_peak_gate.json').read_text())['rows']}
    z=np.load(report['policy_trace']['path']);f=z['frames'];c={str(v):i for i,v in enumerate(z['columns'])};dt=float(z['dt']);rows=[]
    for r in strict:
        if r['vx']>=0 or abs(r['yaw'])>1e-6:continue
        v=f[:,r['env']];t=v[:,c['episode_step']]*dt;cuts=np.flatnonzero(np.diff(t)<0);end=int(cuts[0]+1) if len(cuts) else len(t);v=v[:end];t=t[:end]
        s=reverse_speed_checks(v[:,c['vx']],r['vx'],t>=10,r['complete'])
        checks=dict(survival=r['complete'],no_runaway=s['overshoot_pass'],underspeed=s['underspeed_pass'],
                    pitch_peak=pitch[r['env']]['passed'],yaw=r['checks']['yaw'],
                    leg_angles=reverse_angle_check(r['angle_p95_deg'],leg_limit_deg))
        rows.append(dict(env=r['env'],command=[r['vx'],r['yaw']],**s,checks=checks,
                         angle_p95_deg=r['angle_p95_deg'],leg_limit_deg=leg_limit_deg,
                         original_2deg_leg_pass=r['checks']['straight_angles'],passed=all(checks.values())))
    result=dict(rows=rows,passed=sum(r['passed'] for r in rows),total=len(rows),
        leg_limit_deg=leg_limit_deg,gate_revision=('20261108_reverse_angle6' if leg_limit_deg==6. else '20261107_reverse_angle4' if leg_limit_deg==4. else 'explicit_reverse_angle_override'),
        scope='Entire first12s maximumreverseovershoot<=.05m/s, last2s meanbackwardspeed>=75%target; survival/yaw/rawpitch3deg unchanged. Reverse-only active/hip P95 angle tolerance explicitly recorded. No command scaling or motor overrides.',promoted=False)
    if Path(output_filename).name!=output_filename:raise ValueError('Output must be a local filename')
    (p/output_filename).write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('directory',type=Path)
    ap.add_argument('--leg-limit-deg',type=float,default=4.)
    ap.add_argument('--output-filename',default='reverse_overshoot_gate.json')
    a=ap.parse_args();r=analyze(a.directory,a.leg_limit_deg,a.output_filename);print(r['passed'],r['total'])
