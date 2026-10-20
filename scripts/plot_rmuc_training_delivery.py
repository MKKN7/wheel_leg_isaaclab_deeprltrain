"""Export real training-tail trends and final-policy joint traces; no inference or GPU."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def export_physical_training_tail(log,out,targets=None):
 paths=sorted(Path(log).glob('training_tail_trace_*.npz'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
 if not paths:return {'available':False,'reason':'No actual training trajectory was recorded'}
 chunks=[];dt=None
 for path in paths:
  with np.load(path) as data:
   chunks.append(data['frames']);current_dt=float(data['dt'])
   if dt is not None and dt!=current_dt:raise ValueError('Inconsistent training trace dt')
   dt=current_dt
 frames=np.concatenate(chunks);time=np.arange(len(frames))*dt
 if targets is None:targets=[(.225,0),(.25,0),(.275,0)]
 fig,axes=plt.subplots(len(targets),2,figsize=(13,3*len(targets)),squeeze=False,constrained_layout=True)
 counts=[]
 for i,(v,w) in enumerate(targets):
  mask=(abs(frames[:,:,1]-v)<1e-4)&(abs(frames[:,:,2]-w)<1e-4)
  ids=np.argsort(-mask.sum(0))[:3];counts.append(dict(vx=v,yaw=w,samples=int(mask.sum()),shown_envs=ids.tolist()))
  for env in ids:
   x=frames[:,env];valid=mask[:,env].copy();valid[1:]&=np.diff(x[:,0])>=0
   vx=np.where(valid,x[:,3],np.nan);a=np.where(valid,np.rad2deg(x[:,4]-x[:,7]),np.nan);h=np.where(valid,np.rad2deg(x[:,5]+x[:,8]),np.nan)
   axes[i,0].plot(time,vx,lw=.6,alpha=.6,label=f'env {env}')
   axes[i,1].plot(time,a,lw=.6,alpha=.6,color='tab:blue');axes[i,1].plot(time,h,lw=.6,alpha=.6,color='tab:orange')
  axes[i,0].axhline(v,color='k',ls='--');axes[i,0].axhspan(.9*v,1.1*v,color='green',alpha=.12);axes[i,0].set_ylabel('m/s');axes[i,0].set_title(f'Actual training vx={v}: stochastic actions')
  axes[i,1].axhspan(-2,2,color='green',alpha=.12);axes[i,1].set_ylabel('degree');axes[i,1].set_title('Active difference / mirrored hip difference')
  for ax in axes[i]:ax.grid(alpha=.2);ax.set_xlabel('Recorded training-tail elapsed seconds')
 fig.suptitle('Actual training rollout, pre-action state; 3 example environments per command, NOT acceptance')
 fig.savefig(out/'training_tail_physical_tracking_alignment.png',dpi=150);plt.close(fig)
 return dict(available=True,files=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],conditions=counts,scope='Stochastic training trajectory, includes transients; deterministic acceptance is separate')

def export(experiment,destination):
 e=Path(experiment).resolve();out=Path(destination).resolve();out.mkdir(parents=True,exist_ok=False)
 report=json.loads((e/'evaluation.json').read_text());trace=e/'evaluation.leg_tracking.npz';data=np.load(trace);f=data['frames'];dt=float(data['dt']);ids=data['bucket_ids'];metrics=[]
 selected=[(.25,-.1),(.25,0),(.25,.1),(.5,0),(1.,0),(1.5,0),(1.,1.4),(1.5,1.4)]
 available={(row['vx'],row['yaw']) for row in report['rows']}
 command_path=e/'command.json'
 if command_path.exists():
  command=json.loads(command_path.read_text())['command']
  if '--command_level' in command:
   level=int(command[command.index('--command_level')+1])
   values={v.split('=',1)[0]:v.split('=',1)[1] for v in command if '=' in v}
   vx_key='env.command_stage_linear_x_mps';yaw_key='env.command_stage_yaw_rate_rad_s'
   if vx_key in values and yaw_key in values:
    target=(json.loads(values[vx_key])[level],json.loads(values[yaw_key])[level])
    selected=[target]+[x for x in selected if x!=target]
 selected=[x for x in selected if x in available]
 if not selected:raise ValueError('No requested plot conditions in evaluation')
 nrows=(len(selected)+1)//2
 fig,axes=plt.subplots(nrows,2,figsize=(14,3*nrows),squeeze=False);fig2,axes2=plt.subplots(nrows,2,figsize=(14,3*nrows),squeeze=False)
 for row in report['rows']:
  bucket=report['rows'].index(row);envs=np.flatnonzero(ids==bucket);allx=f[:,envs];mask=allx[:,:,0]>=1200
  # Physical sides: model right = physical left. Hip axes are mirrored.
  active=np.rad2deg(allx[:,:,2]-allx[:,:,5]);hip=np.rad2deg(allx[:,:,3]+allx[:,:,6])
  metrics.append(dict(vx=row['vx'],yaw=row['yaw'],steady_samples=int(mask.sum()),active_signed_mean_deg=float(active[mask].mean()) if mask.any() else None,active_abs_p95_deg=float(np.percentile(abs(active[mask]),95)) if mask.any() else None,hip_signed_mean_deg=float(hip[mask].mean()) if mask.any() else None,hip_abs_p95_deg=float(np.percentile(abs(hip[mask]),95)) if mask.any() else None))
 for i,(v,w) in enumerate(selected):
  j=next(j for j,r in enumerate(report['rows']) if (r['vx'],r['yaw'])==(v,w));env=np.flatnonzero(ids==j)[0];x=f[:,env];end=len(x);start=max(0,end-int(6/dt));x=x[start:end];t=np.arange(start,end)*dt
  cuts=np.flatnonzero(np.diff(x[:,0])<0)+1;speed=x[:,1].copy();active=np.rad2deg(x[:,2]-x[:,5]);hip=np.rad2deg(x[:,3]+x[:,6]);speed[cuts]=np.nan;active[cuts]=np.nan;hip[cuts]=np.nan
  ax=axes.flat[i];ax.plot(t,speed,label='actual vx');ax.axhline(v,color='k',ls='--',label='command');ax.axhspan(v-.1*abs(v),v+.1*abs(v),alpha=.12,color='green');ax.set_title(f'vx={v}, yaw={w}; env={env}');ax.set_ylabel('m/s');ax.grid(alpha=.3)
  ax=axes2.flat[i];ax.plot(t,active,label='active L-R');ax.plot(t,hip,label='hip L-R (mirrored)');ax.axhline(0,color='k',ls='--');ax.set_title(f'vx={v}, yaw={w}; env={env}');ax.set_ylabel('deg');ax.grid(alpha=.3)
  if abs(w)<=1e-4:ax.axhspan(-2,2,alpha=.12,color='green')
 for i in range(len(selected),nrows*2):axes.flat[i].set_visible(False);axes2.flat[i].set_visible(False)
 for ax in [axes.flat[0],axes2.flat[0]]:ax.legend()
 for a in axes[-1]:a.set_xlabel('evaluation elapsed seconds')
 for a in axes2[-1]:a.set_xlabel('evaluation elapsed seconds')
 fig.suptitle('Final-checkpoint evaluation: last 6s, representative replica (not training rollout)');fig.tight_layout();fig.savefig(out/'speed_tracking_final_policy.png',dpi=140);plt.close(fig)
 fig2.suptitle('Body-relative linkage rotation difference, NOT whole-leg swing angle; reset gaps marked');fig2.tight_layout();fig2.savefig(out/'bilateral_link_angle_difference.png',dpi=140);plt.close(fig2)
 status=json.loads((e/'status.json').read_text());log=Path(status['log_dir'])
 from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
 events=EventAccumulator(str(log),size_guidance={'scalars':0});events.Reload();tags=events.Tags()['scalars'];wanted=[x for x in tags if any(k in x for k in ['command_active_turn_positive_linear_error_ema','command_active_turn_negative_linear_error_ema','command_active_turn_positive_yaw_error_ema','command_active_turn_negative_yaw_error_ema'])]
 fig,axes=plt.subplots(2,1,figsize=(12,7));series={}
 for tag in wanted:
  points=events.Scalars(tag);points=points[-max(1,len(points)//3):];series[tag]=[[q.step,q.value] for q in points];ax=axes[0 if 'linear' in tag else 1];ax.plot([q.step for q in points],[q.value for q in points],label=tag.split('/')[-1])
 for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.3);ax.set_xlabel('training iteration')
 axes[0].set_ylabel('speed error m/s');axes[1].set_ylabel('yaw error rad/s');fig.suptitle('Actual training-tail episode EMA; not instantaneous trajectories');fig.tight_layout();fig.savefig(out/'training_tail_tracking.png',dpi=140);plt.close(fig)
 targets=None
 if command_path.exists():targets=json.loads(command_path.read_text()).get('training_plot_conditions')
 physical_training=export_physical_training_tail(log,out,targets)
 manifest=dict(experiment=str(e),checkpoint_sha256=report['checkpoint_sha256'],trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(),selected_conditions=selected,training_series=series,all_condition_joint_metrics=metrics,limitations=['Joint trace is final-checkpoint evaluation, not late training rollout','Differences use q0-q3 and q1+q4 in model order; not whole-leg virtual angle','Plots show one replica; JSON aggregates all steady samples; per-replica gates are in alignment_acceptance.json','No threshold or symmetry success inferred; turning can require asymmetry'])
 manifest['actual_training_physical_trace']=physical_training
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(e/'desktop_charts.json').write_text(json.dumps({'directory':str(out),'checkpoint_sha256':report['checkpoint_sha256']},indent=2));print(out)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();export(a.experiment,a.output)
