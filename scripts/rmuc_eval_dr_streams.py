"""Independent per-environment, per-reset DR draws for paired diagnostics."""
import torch

class WheelDRStreams:
    def __init__(self, count, reset_budget, seed, strength, damping_fraction, gain_fraction, device):
        if count<1 or reset_budget<1 or not 0<=strength<=1 or not 0<=damping_fraction<1 or not 0<=gain_fraction<1:
            raise ValueError('Invalid independent DR dimensions or fractions')
        generator=torch.Generator(device='cpu').manual_seed(int(seed))
        unit=torch.rand((count,reset_budget,2),generator=generator)*2.-1.
        self.scales=(1.+unit*torch.tensor([damping_fraction,gain_fraction])*strength).to(device)
        self.indices=torch.zeros(count,dtype=torch.long,device=device)

    def sample(self, env_ids):
        if bool((self.indices[env_ids]>=self.scales.shape[1]).any()):
            raise RuntimeError('Independent DR reset budget exhausted')
        values=self.scales[env_ids,self.indices[env_ids]]
        self.indices[env_ids]+=1
        return values
