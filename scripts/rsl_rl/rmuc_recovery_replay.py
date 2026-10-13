"""Fixed successful-state action replay, added before PPO gradient clipping."""
import hashlib
import json
import math
from pathlib import Path
import torch


class AuxiliaryGradients:
    """Add precomputed auxiliary gradients during the following main backward."""
    def __init__(self, parameters):
        self.parameters=list(parameters)
        self.gradients={}
        self.handles=[p.register_hook(lambda grad,p=p: grad+self.gradients[p]
                     if p in self.gradients else grad) for p in self.parameters]

    def prepare(self, loss):
        self.clear()
        values=torch.autograd.grad(loss,self.parameters,allow_unused=True)
        self.gradients={p:g.detach() for p,g in zip(self.parameters,values) if g is not None}

    def clear(self):
        self.gradients={}


def load_reference_groups(manifest_path,device):
    path=Path(manifest_path);manifest=json.loads(path.read_text())
    if sorted(x['angle'] for x in manifest['datasets'] if x.get('condition','nominal')=='nominal') != [0,5,15]:
        raise ValueError('Replay requires all three verified reference angles')
    groups=[]
    for entry in manifest['datasets']:
        dataset=path.parent/entry.get('file',f"reference_{entry['angle']}.pt")
        if hashlib.sha256(dataset.read_bytes()).hexdigest()!=entry['sha256']:
            raise ValueError('Replay dataset hash mismatch')
        data=torch.load(dataset,map_location=device,weights_only=False)
        if (data['checkpoint_sha256']!=manifest['checkpoint_sha256']
                or not bool(data['terminal_passed'].all())
                or data['recovery_ground']['inclination_degrees']!=0.):
            raise ValueError('Replay source must be same-checkpoint successful flat references')
        obs,action=data['observation'],data['action']
        if (obs.shape!=(entry['samples'],25) or action.shape!=(entry['samples'],6)
                or not torch.isfinite(obs).all() or not torch.isfinite(action).all()):
            raise ValueError('Invalid replay observation/action data')
        expected_cap=entry.get('communication_delay_cap_steps',-1)
        if data.get('evaluation_communication_delay_cap_steps',-1)!=expected_cap:
            raise ValueError('Replay communication curriculum mismatch')
        if data.get('diagnostic_communication_delay_steps') is not None:
            raise ValueError('Forced-delay interventions are not reference curriculum data')
        readback=data.get('initial_dr_readback',{})
        expected_strength=entry.get('dr_strength',0.)
        if any(abs(v-expected_strength)>1e-6 for v in readback.get('strength',[0.])):
            raise ValueError('Replay DR strength mismatch')
        delay_by_env=torch.tensor(readback.get('communication_delay_steps',
            [0]*len(data['terminal_passed'])),device=obs.device,dtype=torch.long)
        actual_delay=delay_by_env[data['env_id']]
        for case in range(4):
            for early in (True,False):
                mask=(data['case']==case)&((data['control_step']<128)==early)
                if not mask.any():raise ValueError('Missing replay case/phase stratum')
                # Give rare delayed trajectories their own sampling stratum.
                for delay in actual_delay[mask].unique():
                    selected=mask&(actual_delay==delay)
                    groups.append((obs[selected],action[selected]))
    return groups,manifest


def configure_recovery_replay(runner,manifest_path,coefficient,seed=268):
    if not math.isfinite(coefficient) or coefficient<=0:
        raise ValueError('Replay coefficient must be finite and positive')
    actor=runner.alg.get_policy()
    if actor.is_recurrent or getattr(runner.alg,'symmetry',None):
        raise ValueError('Fixed replay supports feed-forward PPO without symmetry')
    device=next(actor.parameters()).device
    groups,manifest=load_reference_groups(manifest_path,device)
    sampler=torch.Generator(device=device).manual_seed(seed)
    auxiliary=AuxiliaryGradients(p for p in actor.parameters() if p.requires_grad)
    original_generator=runner.alg.storage.mini_batch_generator
    original_update=runner.alg.update
    state=dict(losses=[],updates=0)
    start=int(runner.current_learning_iteration)
    directory=Path(runner.logger.log_dir);directory.mkdir(parents=True,exist_ok=True)
    (directory/'recovery_replay_contract.json').write_text(json.dumps(dict(
        manifest=str(Path(manifest_path).resolve()),
        manifest_sha256=hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
        reference_checkpoint_sha256=manifest['checkpoint_sha256'],coefficient=coefficient,
        strata=len(groups),samples_per_stratum=16,sampler_seed=seed,
        scope='Auxiliary raw action MSE on fixed reference states; not a hard behavior guarantee'),indent=2)+'\n')

    def generator(*args,**kwargs):
        for batch in original_generator(*args,**kwargs):
            observations=[];targets=[]
            for obs,action in groups:
                ids=torch.randint(len(obs),(16,),device=device,generator=sampler)
                observations.append(obs[ids]);targets.append(action[ids])
            prediction=actor({'policy':torch.cat(observations)},stochastic_output=False)
            loss=coefficient*(prediction-torch.cat(targets)).square().mean()
            auxiliary.prepare(loss)
            state['losses'].append(float(loss.detach()))
            try:yield batch
            finally:auxiliary.clear()

    def update(*args,**kwargs):
        state['losses']=[]
        try:result=original_update(*args,**kwargs)
        finally:auxiliary.clear()
        if not state['losses']:raise RuntimeError('Replay never entered PPO minibatches')
        result['recovery_replay_loss']=sum(state['losses'])/len(state['losses'])
        with (directory/'recovery_replay.jsonl').open('a') as f:
            f.write(json.dumps(dict(iteration=start+state['updates'],
                minibatches=len(state['losses']),samples=len(groups)*16*len(state['losses']),
                loss=result['recovery_replay_loss']))+'\n')
        state['updates']+=1
        return result

    runner.alg.storage.mini_batch_generator=generator
    runner.alg.update=update
