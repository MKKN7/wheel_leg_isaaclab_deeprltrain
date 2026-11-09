"""Fail-closed opt-in continuation of a verified identical actor parameter selection."""
from pathlib import Path
import hashlib,json,os
import torch

def validate_optimizer_resume(checkpoint,modules,manifest):
    if not checkpoint or not manifest:raise ValueError('Explicit physics optimizer resume requires audited manifest')
    p=Path(checkpoint);d=json.loads(Path(manifest).read_text())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=d['checkpoint_sha256']:raise ValueError('Resume checkpoint hash mismatch')
    payload=torch.load(p,map_location='cpu',weights_only=False);groups=payload['optimizer_state_dict']['param_groups']
    if [x.get('rmuc_group') for x in groups]!=['physics_adaptation','critic'] or groups[0].get('rmuc_physics_modules')!=modules:raise ValueError('Optimizer module provenance mismatch')
    if bool(groups[0].get('rmuc_oracle_full_actor',False))!=(os.environ.get('RMUC_ORACLE_FULL_ACTOR')=='1'):raise ValueError('Optimizer actor scope mismatch')
    protected=Path(os.environ['RMUC_COORDINATION_PROTECTED_KEYS'])
    if hashlib.sha256(protected.read_bytes()).hexdigest()!=d['protected_keys_sha256']:raise ValueError('Protected keys mismatch')
    import importlib.util
    spec=importlib.util.spec_from_file_location('_rmuc_resume_restore',Path(__file__).resolve().parents[1]/'verify_rmuc_export.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    actor=module.restore_actor(p);keys=json.loads(protected.read_text())['keys'];names=dict(actor.named_parameters())
    expected=[k for k in names if k not in keys and not k.startswith('distribution.')]
    if sorted(expected)!=sorted(d['trainable_keys']):raise ValueError('Trainable parameter set mismatch')
    if len(groups[0]['params'])!=len(names):raise ValueError('Optimizer parameter order/count provenance mismatch')
    return True
