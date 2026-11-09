"""Reject aggregate screening gains that hide broad tracking deterioration."""
import argparse,json,statistics
from pathlib import Path

def review(source,candidate):
    sm={(x['vx'],x['yaw']):x for x in source['summary']};cm={(x['vx'],x['yaw']):x for x in candidate['summary']}
    if sm.keys()!=cm.keys():raise ValueError('Paired command sets differ')
    rows=[];reasons=[]
    for key,a in sm.items():
        b=cm[key]
        if a['replicas']!=b['replicas']:raise ValueError('Replica counts differ')
        x=[v for v in source['rows'] if (v['vx'],v['yaw'])==key];y=[v for v in candidate['rows'] if (v['vx'],v['yaw'])==key]
        metrics={}
        for field in ['speed_tail_mae','yaw_tail_mae']:
            pairs=[(u[field],v[field]) for u,v in zip(x,y) if u.get(field) is not None and v.get(field) is not None]
            metrics[field]={'source':statistics.mean(u for u,v in pairs),'candidate':statistics.mean(v for u,v in pairs)} if pairs else None
        row=dict(vx=key[0],yaw=key[1],source=a,candidate=b,paired_available_metrics=metrics);rows.append(row)
        if b['complete']<a['complete']:reasons.append(f'survival loss at {key}')
        if a['strict_screen_pass']==a['replicas'] and b['strict_screen_pass']<b['replicas']:reasons.append(f'previous full-pass point lost at {key}')
    gain=sum(x['candidate']['strict_screen_pass']-x['source']['strict_screen_pass'] for x in rows)
    if gain<=0:reasons.append('no net passing-instance gain')
    for field in ['speed_tail_mae','yaw_tail_mae']:
        values=[x['paired_available_metrics'][field] for x in rows]
        if values and all(v is not None and v['candidate']>v['source']+1e-6 for v in values):reasons.append(f'{field} worsened at every paired point')
    return dict(eligible_for_broad_evaluation=not reasons,reasons=reasons,net_pass_gain=gain,rows=rows,promoted=False,scope='Conservative triage only; does not replace per-point gate, authorized engineering angle review, oscillation or full retention')

def engineering_retention(source, candidate, action_steps=None, require_high_delay_gain=False):
    """Apply already-authorized 3deg straight gate; keep every missing tail failed."""
    def indexed(report):
        rows=report['rows']
        if len({x['env'] for x in rows})!=len(rows):raise ValueError('Duplicate environment IDs')
        return {x['env']:x for x in rows}
    a,b=indexed(source),indexed(candidate)
    if a.keys()!=b.keys():raise ValueError('Paired environment IDs differ')
    def passed(x):
        checks=dict(x['checks'])
        if x['vx']>=1.6 and abs(x['yaw'])<=1e-6:
            angles=x.get('angle_p95_deg')
            checks['straight_angles']=angles is not None and max(angles)<=3.
        return all(checks.values())
    groups={'all':[]}
    if action_steps is not None:groups.update(low=[],middle=[],high=[])
    for e,x in a.items():
        y=b[e]
        if (x['vx'],x['yaw'])!=(y['vx'],y['yaw']):raise ValueError('Paired command mismatch')
        pair=(passed(x),passed(y));groups['all'].append(pair)
        if action_steps is not None:
            step=int(action_steps[e])
            if step not in range(5):raise ValueError('Expected0..4 action physics ticks')
            groups['low' if step<=1 else 'middle' if step==2 else 'high'].append(pair)
    counts={k:dict(n=len(v),source=sum(a for a,b in v),candidate=sum(b for a,b in v),lost=sum(a and not b for a,b in v),gained=sum(b and not a for a,b in v)) for k,v in groups.items()}
    reasons=[f'authorized engineering pass count fell in {k} group' for k,x in counts.items() if x['candidate']<x['source']]
    if require_high_delay_gain:
        high=counts.get('high')
        if high is None or not high['n']:
            reasons.append('high-delay objective has no paired high-delay samples')
        elif high['candidate']<=high['source']:
            reasons.append('no high-delay passing-instance gain for high-delay adaptation objective')
    return dict(eligible=not reasons,reasons=reasons,groups=counts,promoted=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('source',type=Path);ap.add_argument('candidate',type=Path);ap.add_argument('output',type=Path);ap.add_argument('--require-high-delay-gain',action='store_true');a=ap.parse_args();src=json.loads((a.source/'strict_screen.json').read_text());dst=json.loads((a.candidate/'strict_screen.json').read_text());d=review(src,dst)
    if all('env' in x and 'checks' in x for x in src['rows']+dst['rows']):
        steps=None
        paths=[x/'evaluation.joint_delay_trace.npz' for x in [a.source,a.candidate]]
        if all(x.exists() for x in paths):
            import numpy as np
            with np.load(paths[0]) as x, np.load(paths[1]) as y:
                if not np.array_equal(x['frames'][0],y['frames'][0]):raise ValueError('Initial delay assignment mismatch')
                steps=x['frames'][0,:,list(x['columns']).index('action_physics_steps')].copy()
        eng=engineering_retention(src,dst,steps,a.require_high_delay_gain);d['engineering_retention']=eng;d['reasons']+=eng['reasons'];d['eligible_for_broad_evaluation']=not d['reasons']
    elif a.require_high_delay_gain:
        d['reasons'].append('high-delay objective requires per-environment acceptance evidence');d['eligible_for_broad_evaluation']=False
    a.output.write_text(json.dumps(d,indent=2));print({k:v for k,v in d.items() if k!='rows'})
