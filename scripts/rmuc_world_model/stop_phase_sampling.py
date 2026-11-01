"""Training-only stop-event strata; actor inputs and physical commands stay unchanged."""
import torch


def stop_phase_labels(command, dt=.008):
    """Label first two seconds of nonzero->zero steps, without crossing later commands.

    0..2: <=.5 stops; 3..5: >.5 stops. -1: ordinary rehearsal.
    Separate call per episode prevents accidental carry across reset boundaries.
    """
    labels=torch.full_like(command,-1,dtype=torch.long)
    changes=torch.nonzero((command[1:]-command[:-1]).abs()>1e-6).flatten()+1
    for i,start in enumerate(changes.tolist()):
        old=float(command[start-1]);new=float(command[start])
        if abs(new)>1e-6 or abs(old)<.025:continue
        end=int(changes[i+1]) if i+1<len(changes) else len(command)
        age=torch.arange(end-start,device=command.device)*dt
        phase=torch.where(age<.4,0,torch.where(age<1.2,1,2))
        labels[start:end]=torch.where(age<2,phase+3*int(abs(old)>.5),-1)
    return labels


class StopPhaseSampler:
    """Balance actual delay domains; 75% stop strata, 25% ordinary replay per domain."""
    def __init__(self,replay,labels):
        if len(labels)!=len(replay.state):raise ValueError('Misaligned episode labels')
        c=replay.context[replay.starts]
        domains=(c[:,12]>0).long()+2*(c[:,13]>0).long()
        self.groups=[];self.counts=[]
        for key in torch.unique(domains).tolist():
            ordinary=replay.starts[domains==key]
            groups=[ordinary[labels[ordinary]==phase] for phase in range(6)]
            groups=[g for g in groups if len(g)]
            if not groups:raise ValueError('Delay domain has no stop events')
            self.groups.append((ordinary,groups))
            self.counts.append(dict(domain=key,all_windows=len(ordinary),stop_strata=[len(g) for g in groups]))

    def sample(self,batch,generator=None):
        if batch<len(self.groups)*4:raise ValueError('Batch too small for rehearsal and domains')
        pieces=[]
        for i,(ordinary,groups) in enumerate(self.groups):
            n=batch//len(self.groups)+(i<batch%len(self.groups));rehearsal=max(1,n//4)
            pieces.append(ordinary[torch.randint(len(ordinary),(rehearsal,),device=ordinary.device,generator=generator)])
            m=n-rehearsal
            for j,g in enumerate(groups):
                count=m//len(groups)+(j<m%len(groups))
                pieces.append(g[torch.randint(len(g),(count,),device=g.device,generator=generator)])
        out=torch.cat(pieces)
        return out[torch.randperm(batch,device=out.device,generator=generator)]
