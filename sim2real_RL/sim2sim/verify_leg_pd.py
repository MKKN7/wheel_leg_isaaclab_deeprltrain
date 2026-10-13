"""Execute actual Isaac IdealPD AST and compare the deployment reference."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from leg_pd import leg_pd_torque


def main():
    base=Path('/home/kellen/桌面/IsaacLab-3.0/source/isaaclab/isaaclab/actuators')
    source=base/'actuator_pd.py'; clipping=base/'actuator_base.py'
    tree=ast.parse(source.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='IdealPDActuator')
    compute=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='compute')
    clip=next(n for n in ast.walk(ast.parse(clipping.read_text()))
              if isinstance(n,ast.FunctionDef) and n.name=='_clip_effort')
    for f in [compute,clip]:
        f.decorator_list=[];f.returns=None
        for a in f.args.args:a.annotation=None
    namespace={'torch':torch}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[compute,clip],type_ignores=[])),str(source),'exec'),namespace)
    rng=np.random.default_rng(239)
    target,q,qd,vd,ff=[rng.normal(size=(1024,4)) for _ in range(5)]
    # Include the actual zero target-velocity/feedforward operating mode.
    vd[:512]=0.;ff[:512]=0.
    actuator=NS(stiffness=60.,damping=2.,effort_limit=40.)
    actuator._clip_effort=lambda effort:namespace['_clip_effort'](actuator,effort)
    action=NS(joint_positions=torch.from_numpy(target),joint_velocities=torch.from_numpy(vd),joint_efforts=torch.from_numpy(ff))
    expected=namespace['compute'](actuator,action,torch.from_numpy(q),torch.from_numpy(qd)).joint_efforts.numpy()
    actual=leg_pd_torque(target,q,qd,vd,ff)
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
    report=dict(passed=True,samples=1024,max_abs_error_nm=float(abs(actual-expected).max()),
        positive_saturated=int((expected==40).sum()),negative_saturated=int((expected==-40).sum()),
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        clip_source_sha256=hashlib.sha256(clipping.read_bytes()).hexdigest(),
        hardware_controller_verified=False,
        scope='PD algebra and output clipping only; firmware timing, current conversion, filters and limits still require hardware validation')
    Path(__file__).with_name('leg_pd_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
