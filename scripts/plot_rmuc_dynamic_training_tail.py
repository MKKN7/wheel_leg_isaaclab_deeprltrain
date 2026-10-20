"""Actual stochastic training transitions, preserving command changes and resets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def export(experiment, destination):
    experiment=Path(experiment);destination=Path(destination)
    status=json.loads((experiment/'status.json').read_text())
    if status.get('state')!='terminal' or status.get('exit_code')!=0:
        raise ValueError('Completed training required; do not read partially written traces')
    paths=sorted(Path(status['log_dir']).glob('training_tail_trace_*.npz'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
    if not paths:raise ValueError('Missing training tail trace')
    chunks=[];dts=[]
    for path in paths:
        with np.load(path) as d:chunks.append(d['frames'].copy());dts.append(float(d['dt']))
    if len(set(dts))!=1:raise ValueError('Inconsistent policy period')
    f=np.concatenate(chunks);dt=dts[0];t=np.arange(len(f))*dt
    contiguous=np.diff(f[:,:,0],axis=0)==1
    changes=contiguous & (np.diff(f[:,:,1],axis=0)!=0)
    counts=changes.sum(0);ids=np.flatnonzero(counts>0)
    if not len(ids):raise ValueError('No within-episode transitions in actual tail')
    ids=sorted(ids,key=lambda e:(-counts[e],e))[:3]
    destination.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(len(ids),2,figsize=(14,3.4*len(ids)),squeeze=False,constrained_layout=True)
    for i,e in enumerate(ids):
        x=f[:,e];gap=np.r_[False,~contiguous[:,e]]
        values=[x[:,1].copy(),x[:,3].copy(),np.rad2deg(x[:,4]-x[:,7]),np.rad2deg(x[:,5]+x[:,8])]
        for v in values:v[gap]=np.nan
        axes[i,0].plot(t,values[0],'k--',label='command vx');axes[i,0].plot(t,values[1],label='measured vx')
        axes[i,1].plot(t,values[2],label='active q0-q3');axes[i,1].plot(t,values[3],label='hip q1+q4')
        axes[i,1].axhspan(-2,2,alpha=.12,color='green',label='steady reference only')
        for ax in axes[i]:
            for k in np.flatnonzero(changes[:,e] & (np.abs(np.diff(x[:,1])) > .02)):ax.axvline(t[k+1],alpha=.2,color='gray')
            ax.set_title(f'env {e}; {counts[e]} changing-command samples');ax.set_xlabel('Recorded tail seconds');ax.grid(alpha=.2);ax.legend(fontsize=8)
        axes[i,0].set_ylabel('m/s');axes[i,1].set_ylabel('degree')
    fig.suptitle('Actual training tail: stochastic policy, transient angles retained, NOT acceptance')
    output=destination/'dynamic_training_speed_and_leg_difference.png';fig.savefig(output,dpi=150);plt.close(fig)
    report=dict(output=str(output),shown_envs=[int(e) for e in ids],selection='Most within-episode changing-command samples (ramps contribute multiple samples), not best-performing environments',changing_command_sample_counts=[int(counts[e]) for e in ids],files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],scope='Stochastic training tail; reset/discontinuity gaps; no behavior pass inferred')
    (experiment/'dynamic_training_charts.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);a=p.parse_args();export(a.experiment,a.destination)
