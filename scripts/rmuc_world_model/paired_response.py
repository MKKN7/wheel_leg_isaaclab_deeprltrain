"""Paired recorded-action response loss. Hold out envs before building tensors."""
import json
from pathlib import Path
import torch
from rich_replay import load_capture

class PairedResponse:
    def __init__(self, baseline, probe, device, horizon=32, policy_start_seconds=None):
        br=json.loads((baseline/'evaluation.json').read_text());pr=json.loads((probe/'evaluation.json').read_text())
        keys=['seed','initial_state_audit','initial_mass_inertia_readback','initial_material_readback']
        if policy_start_seconds is None:keys.append('checkpoint_sha256')
        for key in keys:
            if br[key]!=pr[key]:raise ValueError('Pair mismatch: '+key)
        start=round(float(pr['diagnostic_action_bias_start_seconds'] if policy_start_seconds is None else policy_start_seconds)/.008)
        if start < 4:raise ValueError('Insufficient pre-intervention history')
        b,meta=load_capture(baseline,4,horizon,True,True);p,pm=load_capture(probe,4,horizon,True,True)
        assert meta['state_columns']==pm['state_columns'] and meta['extra_state_names']==pm['extra_state_names']
        bd={q['env']:q for q in b};pd={q['env']:q for q in p};ids=sorted(e for e in bd if e in pd and e%4!=3)
        if not ids:raise ValueError('No training pairs')
        for e in ids:
            assert len(bd[e]['state'])>=start+horizon and len(pd[e]['state'])>=start+horizon
            assert torch.equal(bd[e]['state'][:start],pd[e]['state'][:start])
            assert torch.equal(bd[e]['action'][:start],pd[e]['action'][:start])
            assert torch.equal(bd[e]['context'][start:start+horizon],pd[e]['context'][start:start+horizon])
        fields={'state':lambda q:q['state'][start-4:start], 'history':lambda q:q['action'][start-4:start], 'action':lambda q:q['action'][start:start+horizon], 'target':lambda q:q['state'][start:start+horizon], 'context':lambda q:q['context'][start:start+horizon]}
        self.base={k:torch.stack([fn(bd[e]) for e in ids]).to(device) for k,fn in fields.items()}
        self.probe={k:torch.stack([fn(pd[e]) for e in ids]).to(device) for k,fn in fields.items()}
        self.names=meta['extra_state_names'];self.ids=ids;self.horizon=horizon
        self.columns=meta['state_columns']+self.names
    def metric(self,x):
        from train_imagination import physical
        ix=lambda key:[16+i for i,n in enumerate(self.names) if n.startswith(key+':')]
        shape=x.shape[:-1];root,q,dq,vx,yaw,pitch,roll=physical(x.reshape(-1,x.shape[-1]),ix)
        return torch.stack([vx,yaw,pitch,q[:,2]-q[:,0],q[:,3]+q[:,1]],-1).reshape(*shape,5)
    def loss(self, model, absolute_weight=0.):
        b,p=self.base,self.probe
        pred_b=model.rollout(b['state'],b['history'],b['action'],b['context'])
        pred_p=model.rollout(b['state'],b['history'],p['action'],b['context'])
        # Multi-horizon matched deltas, not a sign-only target or a fake Jacobian.
        indices=[7,15,31];scale=pred_b.new_tensor([.005,.02,.005,.01,.01])
        pb,pp=self.metric(pred_b[:,indices]),self.metric(pred_p[:,indices])
        tb,tp=self.metric(b['target'][:,indices]),self.metric(p['target'][:,indices])
        response=(((pp-pb)-(tp-tb))/scale).square().mean()
        absolute=.5*(((pb-tb)/scale).square().mean()+((pp-tp)/scale).square().mean())
        self.last_response=response.detach();self.last_absolute=absolute.detach()
        return response+absolute_weight*absolute
