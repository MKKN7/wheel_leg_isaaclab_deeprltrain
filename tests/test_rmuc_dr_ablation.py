"""Diagnostic DR component experiments cannot become policy certificates."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest

TREE=ast.parse((Path(__file__).resolve().parents[1]/'scripts/evaluate_rmuc_grid.py').read_text())


@pytest.mark.parametrize('mode,gas,expected',[('none',None,True),('physics_only',None,False),('observation_only',None,False),('none',10.775,False),('none',0.,False)])
def test_actual_report_expression_rejects_diagnostic_certification(mode,gas,expected):
    node=next(n for n in ast.walk(TREE) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='diagnostic_dr_ablation' for k in n.keys))
    expression=next(v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value=='passed')
    args=SimpleNamespace(radial_spring_force=None,gas_spring_torque=gas,dr_ablation=mode,audit_critic=False,action_bias_sweep=None,pitch_rate_feedback=None)
    result=eval(compile(ast.Expression(expression),'<actual-report>','eval'),{'args':args,'rows':[{'passed':True}]})
    assert result is expected


def test_radial_diagnostic_cannot_certify_policy():
    node=next(n for n in ast.walk(TREE) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='diagnostic_dr_ablation' for k in n.keys))
    expression=next(v for k,v in zip(node.keys,node.values) if isinstance(k,ast.Constant) and k.value=='passed')
    args=SimpleNamespace(radial_spring_force=40.93,gas_spring_torque=None,dr_ablation='none',audit_critic=False,action_bias_sweep=None,pitch_rate_feedback=None)
    assert eval(compile(ast.Expression(expression),'<actual-report>','eval'),{'args':args,'rows':[{'passed':True}]}) is False
