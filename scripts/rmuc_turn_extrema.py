"""Read-only control-step extrema. Thresholds label diagnostics, not acceptance."""
import torch
from rmuc_turn_diagnostics import COLUMNS

class TurnExtrema:
    NAMES=('min_normal_load_fraction','root_heading_vy_abs_mps',
           'wheel_center_longitudinal_slip_mps','quasistatic_support_margin_proxy_m')
    def __init__(self,count,device,dt):
        self.dt=dt
        self.indices=[COLUMNS.index(n) for n in self.NAMES]
        self.phases={}
        for phase in ('all_steps','steady'):
            self.phases[phase]={
                'count':torch.zeros(count,device=device),
                'minimum':torch.full((count,4),float('inf'),device=device),
                'maximum':torch.full((count,4),-float('inf'),device=device),
                'events':torch.zeros(count,4,device=device),
                'streak':torch.zeros(count,4,device=device),
                'max_streak':torch.zeros(count,4,device=device)}
    def update(self,metrics,wheel_normal_z,ready,done):
        values=metrics[:,self.indices]
        finite=torch.isfinite(values).all(-1)&torch.isfinite(wheel_normal_z).all(-1)
        dual=(wheel_normal_z>2.).all(-1)
        events=torch.stack((~dual,values[:,0]<.1,values[:,2]>.15,values[:,3]<0.),-1)
        # A negative support proxy is meaningful only with both wheels supporting.
        events[:,3]&=dual
        for phase,s in self.phases.items():
            selected=finite if phase=='all_steps' else finite&ready
            s['count']+=selected
            s['minimum']=torch.minimum(s['minimum'],torch.where(selected[:,None],values,float('inf')))
            s['maximum']=torch.maximum(s['maximum'],torch.where(selected[:,None],values,-float('inf')))
            hit=events&selected[:,None]
            s['events']+=hit
            s['streak']=torch.where(hit,s['streak']+1,0.)
            s['max_streak']=torch.maximum(s['max_streak'],s['streak'])
            # Count terminal sample then reset; never join separate episodes.
            s['streak'][done]=0.
    def report(self,selection):
        result={}
        labels=('not_both_wheels_above_2N','weak_normal_fraction_below_0p1',
                'local_longitudinal_slip_above_0p15mps','negative_support_proxy_with_two_contacts')
        for phase,s in self.phases.items():
            count=float(s['count'][selection].sum())
            valid=selection&(s['count']>0)
            if not count:
                result[phase]={'samples':0,'extrema':None,'events':None}
                continue
            low=s['minimum'][valid].amin(0).tolist()
            high=s['maximum'][valid].amax(0).tolist()
            fractions=(s['events'][selection].sum(0)/count).tolist()
            durations=(s['max_streak'][selection].amax(0)*self.dt).tolist()
            result[phase]={'samples':int(count),
              'extrema':{n:{'min':a,'max':b} for n,a,b in zip(self.NAMES,low,high)},
              'events':{n:{'sample_fraction':f,'longest_observed_duration_s':d}
                        for n,f,d in zip(labels,fractions,durations)}}
        result['scope']='Control-step observations only; substep events may be missed. Steady excludes terminal samples; all_steps includes them. Support proxy is not ZMP; thresholds are diagnostic, not acceptance.'
        return result
