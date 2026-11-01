"""Vectorized window gathering. Episode boundaries remain validated by the loader."""
import torch

class WindowReplay:
    def __init__(self, sequences, history, horizon, device):
        self.history, self.horizon = history, horizon
        self.device = torch.device(device)
        if not sequences:
            raise ValueError('Empty replay')
        dynamic = sequences[0]['context'].ndim == 2
        if any((q['context'].ndim == 2) != dynamic for q in sequences):
            raise ValueError('Mixed context schemas')
        states, actions, contexts, starts, offsets = [], [], [], [], [0]
        offset = 0
        for q in sequences:
            valid = q['starts'][(q['starts'] >= history) & (q['starts'] + horizon <= len(q['state']))]
            if not len(valid):
                continue
            states.append(q['state']); actions.append(q['action'])
            contexts.append(q['context'] if dynamic else q['context'][None].expand(len(q['state']), -1))
            starts.append(valid + offset); offsets.append(offsets[-1] + len(valid))
            offset += len(q['state'])
        if not starts:
            raise ValueError('No valid replay windows')
        self.state = torch.cat(states).to(self.device)
        self.action = torch.cat(actions).to(self.device)
        self.context = torch.cat(contexts).to(self.device)
        self.starts = torch.cat(starts).to(self.device)
        self.offsets = torch.tensor(offsets, device=self.device)
        self.past = torch.arange(-history, 0, device=self.device)
        self.future = torch.arange(horizon, device=self.device)
        self.dynamic = dynamic

    def sample(self, batch, generator=None):
        # Uniform trajectories, then uniform valid starts: same weighting as legacy sample().
        q = torch.randint(len(self.offsets)-1, (batch,), device=self.device, generator=generator)
        u = torch.rand(batch, device=self.device, generator=generator)
        index = self.offsets[q] + (u*(self.offsets[q+1]-self.offsets[q])).long()
        t = self.starts[index]
        past, future = t[:, None]+self.past, t[:, None]+self.future
        context = self.context[future] if self.dynamic else self.context[t]
        return self.state[past], self.action[past], self.action[future], self.state[future], context
