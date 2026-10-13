"""CPU-only explicit frozen-source physical residual migration; never trains."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent/'rsl_rl'))
from rmuc_physics_adaptation import PhysicsAdaptation
from verify_rmuc_export import restore_actor


def prepare(source, output):
    if output.exists():
        raise FileExistsError(output)
    payload=torch.load(source,map_location='cpu',weights_only=False)
    state=payload['actor_state_dict']; prefix='rmuc_physics_adaptation.'
    old={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    if not old or any(k.startswith('frozen_source.') for k in old):
        raise ValueError('Require a source with exactly one non-migrated physical head')
    torch.manual_seed(443)
    frozen=PhysicsAdaptation(); frozen.load_state_dict(old,strict=True)
    migrated=PhysicsAdaptation(frozen)
    updated={k:v for k,v in state.items() if not k.startswith(prefix)}
    updated.update({prefix+k:v for k,v in migrated.state_dict().items()})
    for k,v in state.items():
        mapped=prefix+'frozen_source.'+k[len(prefix):] if k.startswith(prefix) else k
        if not torch.equal(v,updated[mapped]):
            raise AssertionError(f'Frozen source changed: {k}')
    result=copy.deepcopy(payload);result['actor_state_dict']=updated
    result.pop('optimizer_state_dict',None)
    source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    result['physics_migration_provenance']={'source':str(source.resolve()),'source_sha256':source_sha,
        'optimizer_reset_required':True,'old_actor_frozen':True,'total_budget_expanded':False}
    output.parent.mkdir(parents=True,exist_ok=True)
    # Output is a new checkpoint; preserve source iteration and critic state.
    with output.open('xb') as stream:torch.save(result,stream)
    actor_old=restore_actor(source);actor_new=restore_actor(output)
    torch.manual_seed(443);obs=torch.randn(2048,25)*.3
    speeds=torch.tensor([0.,-1.,.8,1.,1.5,1.5001,1.6999,1.7,3.2])
    yaws=torch.tensor([-.8,.8,-1.,1.,-1.2,1.2,-1.4,1.4,4.,4.0001,4.1999,4.2])
    commands=torch.cartesian_prod(speeds,yaws);boundaries=torch.randn(len(commands),25)*.1
    boundaries[:,7]=commands[:,0]*.5;boundaries[:,8]=commands[:,1]-boundaries[:,2]/.25
    obs=torch.cat([obs,boundaries])
    with torch.inference_mode():
        before=actor_old({'policy':obs});after=actor_new({'policy':obs})
    if not torch.equal(before,after):
        raise AssertionError(f'Initial complete actor mismatch: {(before-after).abs().max()}')
    report={**result['physics_migration_provenance'],'derived_checkpoint':str(output.resolve()),
        'derived_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'source_tensor_count':len(state),'frozen_tensor_mismatches':[],
        'same_input_samples':len(obs),'complete_actor_max_abs_error':float((before-after).abs().max()),
        'physical_total_budget':migrated.budget.tolist(),'iteration':payload.get('iter'),
        'training_started':False,'promoted':False}
    report_path=output.with_suffix('.migration.json')
    with report_path.open('x') as stream:json.dump(report,stream,indent=2)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps(prepare(a.source,a.output),indent=2))
