"""Explicit delay-domain balance, independent of how many captures each has."""
import torch

class DelayDomainSampler:
    def __init__(self,replay):
        c=replay.context[replay.starts]
        # Actual activation, not filename or requested delay profile.
        domain=(c[:,12]>0).long()+2*(c[:,13]>0).long()
        self.groups=[replay.starts[domain==k] for k in torch.unique(domain).tolist()]
        self.counts=[len(g) for g in self.groups]
        if len(self.groups)<2:raise ValueError('Balanced delay sampling needs multiple actual domains')
    def sample(self,batch,generator=None):
        if batch<len(self.groups):raise ValueError('Batch smaller than domain count')
        pieces=[]
        for i,g in enumerate(self.groups):
            n=batch//len(self.groups)+(i<batch%len(self.groups))
            pieces.append(g[torch.randint(len(g),(n,),device=g.device,generator=generator)])
        result=torch.cat(pieces)
        return result[torch.randperm(batch,device=result.device,generator=generator)]


class ActiveReverseSampler:
    """Reserve actual trainable reverse windows, retaining delay-balanced rehearsal.

    Coverage is based on the restored actor's gates, not a directory name. Only
    domains present in active replay are sampled; missing domains are reported.
    """
    def __init__(self, replay, observations, adaptation, fraction, rehearsal, anchor=False, command_range=None, eligible=None):
        if not 0 < fraction < 1:
            raise ValueError('Reverse focus fraction must be between zero and one')
        self.replay, self.fraction, self.rehearsal = replay, fraction, rehearsal
        starts = replay.starts
        with torch.no_grad():
            obs = observations[starts]
            gate = adaptation.reverse_gate(obs)
            if hasattr(adaptation, 'reverse_update_domain'):
                d = adaptation.reverse_update_domain
                t = ((d[1] - obs[:, 7] / .5) / (d[1] - d[0])).clamp(0, 1)
                scope = t.square() * (3 - 2 * t)
                gate = gate * (1 - scope if anchor else scope)
            elif anchor:
                raise ValueError("Anchor sampling requires existing update scope")
            if not anchor and hasattr(adaptation, 'reverse_extension_domain'):
                d=adaptation.reverse_extension_domain
                t=((d[1]-obs[:,7]/.5)/(d[1]-d[0])).clamp(0,1)
                gate=gate*t.square()*(3-2*t)
            if command_range is not None:
                lo, hi = command_range
                if not lo < hi < 0:raise ValueError("Negative ordered command range required")
                speed=obs[:,7]/.5
                gate=gate*((speed>=lo)&(speed<=hi)).to(gate.dtype)
        active = gate > 1e-6
        if eligible is not None:
            if eligible.shape != (len(observations),) or eligible.dtype != torch.bool:raise ValueError("Invalid replay eligibility mask")
            active = active & eligible[starts]
        if not active.any():
            raise ValueError('No active reverse update windows')
        self.active_by_index = torch.zeros(len(observations), dtype=torch.bool, device=starts.device)
        self.active_by_index[starts] = active
        self.counts = getattr(rehearsal, 'counts', None)
        ctx = replay.context[starts]
        domain = (ctx[:, 12] > 0).long() + 2 * (ctx[:, 13] > 0).long()
        bins = torch.floor(ctx[:, -3] / .1 + 1e-5).long()
        keys = torch.stack([domain, bins], -1)
        self.groups = []
        coverage = []
        for key in keys[active].unique(dim=0):
            ids = starts[active & (keys == key).all(-1)]
            self.groups.append(ids)
            coverage.append(dict(delay_domain=int(key[0]), command_bin=int(key[1]), windows=len(ids)))
        self.report = dict(fraction=fraction, active_windows=int(active.sum()),
                           total_windows=len(starts), groups=coverage,
                           available_delay_domains=domain.unique().tolist(),
                           active_delay_domains=domain[active].unique().tolist())

    def sample(self, batch, generator=None):
        n = round(batch * self.fraction)
        if n < len(self.groups) or batch - n < 2:
            raise ValueError('Batch too small for focus groups and rehearsal')
        pieces = []
        for i, ids in enumerate(self.groups):
            count = n // len(self.groups) + (i < n % len(self.groups))
            pieces.append(ids[torch.randint(len(ids), (count,), device=ids.device, generator=generator)])
        if self.rehearsal is not None:
            pieces.append(self.rehearsal.sample(batch - n, generator))
        else:
            starts = self.replay.starts
            pieces.append(starts[torch.randint(len(starts), (batch-n,), device=starts.device, generator=generator)])
        result = torch.cat(pieces)
        self.last_active_fraction = float(self.active_by_index[result].float().mean())
        return result[torch.randperm(batch, device=result.device, generator=generator)]
