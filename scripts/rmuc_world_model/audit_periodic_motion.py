"""Steady first-episode periodic motion evidence; never a standalone acceptance gate."""
import argparse,json
from pathlib import Path
import numpy as np

def periodic_metrics(signal,dt):
 x=np.asarray(signal,dtype=float)
 if len(x)<32 or not np.isfinite(x).all():return {'valid':False}
 t=np.arange(len(x))*dt
 trend=np.polyval(np.polyfit(t,x,1),t);y=x-trend
 power=np.abs(np.fft.rfft(y*np.hanning(len(y))))**2
 freq=np.fft.rfftfreq(len(y),dt);band=(freq>=.5)&(freq<=15)
 if not band.any():return {'valid':False}
 indices=np.flatnonzero(band);peak=indices[np.argmax(power[band])];total=power[band].sum()
 concentration=float(power[max(indices[0],peak-1):min(indices[-1]+1,peak+2)].sum()/total) if total>1e-20 else 0.
 chunks=np.array_split(y,3);amplitudes=[float(np.quantile(a,.95)-np.quantile(a,.05)) for a in chunks]
 return dict(valid=True,dominant_hz=float(freq[peak]),band_power_fraction_near_peak=concentration,amplitude_p95_p05=float(np.quantile(y,.95)-np.quantile(y,.05)),window_amplitudes=amplitudes,last_first_amplitude_ratio=amplitudes[-1]/max(amplitudes[0],1e-12),rms=float(np.sqrt(np.mean(y*y))),trend_per_second=float(np.polyfit(t,x,1)[0]))

def analyze(directory):
 d=json.loads((directory/'evaluation.json').read_text());z=np.load(d['policy_trace']['path']);f=z['frames'];cols={str(k):i for i,k in enumerate(z['columns'])};dt=.008
 spec=json.loads((directory/'command.json').read_text());args=spec['command'];grid=json.loads(Path(args[args.index('--command-grid')+1]).read_text());leg=np.load(d['leg_tracking_trace']['path']);buckets=leg['bucket_ids'];dt=float(leg['dt']);rows=[]
 for env in range(f.shape[1]):
  v=f[:,env];t=v[:,cols['episode_step']]*dt;reset=np.flatnonzero(np.diff(t)<0);end=int(reset[0]+1) if len(reset) else len(t);v=v[:end];t=t[:end]
  # Fixed common interval prevents variable lifetime/fall transients being called steady oscillation.
  complete=bool(len(t) and t[-1]>=12-2*dt);a=v[(t>=6)&(t<12)]
  row=dict(env=env,command=grid[int(buckets[env])],complete_first12s=complete,metrics={})
  if complete:
   signals={k:a[:,cols[k]] for k in ['vx','pitch_deg','action_input_observation_3','wheel_target_right_nm','wheel_target_left_nm']}
   signals['active_difference_deg']=np.rad2deg(a[:,cols['joint_position_0']]-a[:,cols['joint_position_3']]);signals['hip_difference_deg']=np.rad2deg(a[:,cols['joint_position_1']]+a[:,cols['joint_position_4']])
   row['metrics']={k:periodic_metrics(value,dt) for k,value in signals.items()}
  rows.append(row)
 out=dict(scope='First episode fixed6–12s only; detrended FFT0.5–15Hz and three-window amplitude. Missing complete window is invalid, never pass. No amplitude gate invented; requires paired baseline and engineering review. Does not diagnose delay causality.',rows=rows,promoted=False)
 (directory/'periodic_motion_audit.json').write_text(json.dumps(out,indent=2));return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();r=analyze(a.directory);print('audited',len(r['rows']))
