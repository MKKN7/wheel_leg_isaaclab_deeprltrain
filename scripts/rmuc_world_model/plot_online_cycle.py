"""Paired real simulator traces after model-based training, not imagination curves."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('cycle',type=Path);a=p.parse_args()
out=Path('/home/kellen/桌面/RMUC训练曲线')/('world_model_'+a.cycle.parent.name+'_'+a.cycle.name);out.mkdir(parents=True,exist_ok=True)
conditions=[('.',a.cycle)] if (a.cycle/'evaluation.json').exists() else [(k,a.cycle/k) for k in ['joint10','sensor8']]
for condition,d in conditions:
 spec=json.loads((d/'command.json').read_text());new=json.loads((d/'evaluation.json').read_text());old=json.loads(Path(spec['baseline']).read_text())
 baseline_spec_path=Path(spec['baseline']).parent/'command.json'
 baseline_spec=json.loads(baseline_spec_path.read_text());baseline_argv=baseline_spec['command'];baseline_schedule=json.loads(Path(baseline_argv[baseline_argv.index('--command-sequence')+1]).read_text())
 argv=spec['command'];candidate_schedule=json.loads(Path(argv[argv.index('--command-sequence')+1]).read_text());matched_schedule=baseline_schedule==candidate_schedule
 traces=[np.load(x['leg_tracking_trace']['path']) for x in [old,new]]
 cols={str(x):i for i,x in enumerate(traces[0]['columns'])};fig,axes=plt.subplots(3,2,figsize=(15,10),squeeze=False)
 for row,env in enumerate([0,18,28]):
  for version,z,color in zip(['paired source' if matched_schedule else 'reference DIFFERENT command','candidate'],traces,['tab:gray','tab:blue']):
   f=z['frames'][:,env].copy();dt=float(z['dt']);end=np.flatnonzero(np.diff(f[:,0])<0);f=f[:int(end[0]+1)] if len(end) else f;t=f[:,0]*dt
   axes[row,0].plot(t,f[:,1],color=color,label=version)
   axes[row,1].plot(t,np.rad2deg(f[:,2]-f[:,5]),color=color,label=version+' active')
   axes[row,1].plot(t,np.rad2deg(f[:,3]+f[:,6]),'--',color=color,label=version+' hip')
  argv=spec['command'];schedule=json.loads(Path(argv[argv.index('--command-sequence')+1]).read_text());tt=[x[0] for x in schedule]+[20];vv=[x[1] for x in schedule]+[schedule[-1][1]]
  axes[row,0].step(tt,vv,where='post',color='black',ls='--',label='candidate command')
  if not matched_schedule:axes[row,0].step([x[0] for x in baseline_schedule]+[20],[x[1] for x in baseline_schedule]+[baseline_schedule[-1][1]],where='post',color='tab:gray',ls=':',label='reference command (not paired)')
  axes[row,1].axhspan(-2,2,color='green',alpha=.1,label='steady reference only')
  for col,ax in enumerate(axes[row]):ax.set_title(f'env {env}');ax.set_xlabel('Episode time (s)');ax.set_ylabel('m/s' if col==0 else 'degree');ax.grid(alpha=.2);ax.legend(fontsize=8)
 fig.suptitle(f'{a.cycle.name} / {condition}: actual deterministic physics; first episode only; pass/fail in JSON')
 fig.tight_layout();target=out/(('paired' if condition=='.' else condition)+'_speed_and_leg_difference.png');fig.savefig(target,dpi=130);plt.close(fig);print(target)
