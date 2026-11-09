"""User-confirmed raw steady pitch peak-to-peak gate, per episode, without filtering."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pitch_peak_gate(time, pitch, *, complete, start=6., end=12., limit=3., dt=.008):
    t=np.asarray(time,dtype=float);x=np.asarray(pitch,dtype=float)
    if t.shape!=x.shape or t.ndim!=1 or not np.isfinite(t).all() or not np.isfinite(x).all():
        raise ValueError('Finite matching one-dimensional traces required')
    selected=(t>=start)&(t<end)
    times=t[selected];values=x[selected]
    valid=bool(complete and len(values)>=int((end-start)/dt)-2
               and times[0]<=start+2*dt and times[-1]>=end-2*dt
               and np.allclose(np.diff(times),dt,atol=dt*.05,rtol=0))
    if not valid:
        return dict(valid=False,passed=False,pitch_peak_to_peak_deg=None,
                    reason='Incomplete or discontinuous steady window')
    span=float(values.max()-values.min())
    center=float(values.mean())
    return dict(valid=True,passed=span<=limit,pitch_peak_to_peak_deg=span,
                min_pitch_deg=float(values.min()),max_pitch_deg=float(values.max()),
                steady_mean_pitch_deg=center,
                single_sided_peak_about_mean_deg=float(np.max(np.abs(values-center))),
                limit_deg=limit)


def analyze(directory):
    p=Path(directory);report=json.loads((p/'evaluation.json').read_text())
    strict=json.loads((p/'strict_screen.json').read_text())['rows']
    functional={r['env']:r for r in json.loads((p/'straight_yaw08_screen.json').read_text())['rows']}
    path=Path(report['policy_trace']['path']);z=np.load(path);frames=z['frames']
    cols={str(k):i for i,k in enumerate(z['columns'])};dt=float(z['dt'])
    rows=[]
    for original in strict:
        env=original['env'];f=frames[:,env];t=f[:,cols['episode_step']]*dt
        resets=np.flatnonzero(np.diff(t)<0);end=int(resets[0]+1) if len(resets) else len(t)
        result=pitch_peak_gate(t[:end],f[:end,cols['pitch_deg']],complete=original['complete'],dt=dt)
        motion=bool(functional[env]['yaw08_component_pass'])
        rows.append(dict(env=env,command=[original['vx'],original['yaw']],
                         **result,functional_screen_pass=motion,
                         combined_screen_pass=motion and result['passed']))
    commands=list(dict.fromkeys(tuple(r['command']) for r in rows));summary=[]
    for command in commands:
        group=[r for r in rows if tuple(r['command'])==command]
        values=[r['pitch_peak_to_peak_deg'] for r in group if r['valid']]
        summary.append(dict(command=list(command),replicas=len(group),valid=sum(r['valid'] for r in group),
                            peak_gate_pass=sum(r['passed'] for r in group),
                            combined_screen_pass=sum(r['combined_screen_pass'] for r in group),
                            worst_pitch_peak_to_peak_deg=max(values) if values else None))
    out=dict(rows=rows,summary=summary,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
             limit_deg=3.,window_seconds=[6,12],method='Raw max-min; constant offset irrelevant. No detrend, low-pass, percentiles, or averaging across replicas.',
             scope='Steady pitch fluctuation component only. Motion uses unchanged existing engineering tracking/yaw/angle screen. Transient reverse overshoot, dynamic transitions, full154 and multiseed remain separate.',
             promoted=False)
    (p/'pitch_peak_gate.json').write_text(json.dumps(out,indent=2))
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    result=analyze(parser.parse_args().directory)
    print(json.dumps(result['summary'],indent=2))
