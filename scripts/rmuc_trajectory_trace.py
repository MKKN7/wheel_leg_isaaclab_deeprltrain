"""Offline episode-safe metrics from pre-step / pre-reset physical states.

Commands remain body vx+yaw. Columns are shared with the opt-in grid recorder.
"""
import numpy as np
from rmuc_trajectory_metrics import arc_metrics, step_metrics

COLUMNS = ['pre_x','pre_y','pre_heading','pre_speed','pre_age',
           'post_x','post_y','post_heading','post_speed','post_age',
           'terminated','truncated','vx_command','yaw_command']


def analyze_trace(frames, dt, minimum_duration_s=10.):
    x=np.asarray(frames,dtype=float)
    if x.ndim!=3 or x.shape[2]!=len(COLUMNS) or not np.isfinite(x).all():
        raise ValueError('Finite [step,env,14] pre/post trace required')
    if not np.isfinite(dt) or dt<=0 or minimum_duration_s<=0:
        raise ValueError('Positive finite timebase required')
    rows=[]
    for env in range(x.shape[1]):
        z=x[:,env];start=0;before=None;integrity=True
        for end in range(len(z)):
            f=z[end]
            integrity=integrity and abs(f[9]-f[4]-1)<.01
            next_reset=end+1<len(z) and (z[end+1,4]<=f[4] or bool(f[10] or f[11]))
            next_command=end+1<len(z) and not np.allclose(z[end+1,12:14],f[12:14],atol=1e-6,rtol=0)
            boundary=end==len(z)-1 or next_reset or next_command or bool(f[10] or f[11])
            if not boundary:continue
            part=z[start:end+1];t=np.arange(len(part)+1)*dt
            pos=np.vstack((part[0,:2],part[:,5:7]));v=np.r_[part[0,3],part[:,8]]
            metric=arc_metrics(t,pos,*part[0,12:14],initial_xy=part[0,:2],initial_heading=part[0,2])
            survived=not bool(part[:,10].any())
            enough=bool(t[-1]+1e-8>=minimum_duration_s)
            row=dict(environment=env,start_step=start,end_step=end,command=part[0,12:14].tolist(),
                     terminated=not survived,truncated=bool(f[11]),episode_integrity=bool(integrity),
                     duration_gate_passed=enough,arc=metric,
                     arc_segment_passed=bool(survived and integrity and enough and metric['geometric_gate_passed']),
                     step=None,step_scope='No observed within-episode vx command transition; reset is not a measured step')
            if before is not None and abs(before-part[0,12])>1e-9 and len(t)>=3:
                row['step']=step_metrics(t,v,before,part[0,12]);row['step']['passed']=bool(row['step']['passed'] and survived and integrity and enough)
                row['step_scope']='Observed command transition; timing starts at pre-step state, no reset included'
            rows.append(row)
            before=float(f[12]) if next_command and not next_reset and not (f[10] or f[11]) else None
            start=end+1;integrity=True
    return dict(rows=rows,full_acceptance=False,columns=COLUMNS,
                limitations=['Fixed-command grid does not certify dynamic switching or step response',
                             'Arc uses geometric distance; velocity tracking and survival gates remain separate',
                             'No extra command channel or controller feedback'])
