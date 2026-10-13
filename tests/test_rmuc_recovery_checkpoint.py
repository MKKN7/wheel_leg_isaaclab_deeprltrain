import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

spec = importlib.util.spec_from_file_location('recovery_checkpoint',
    Path(__file__).resolve().parents[1]/'scripts/rsl_rl/rmuc_recovery_checkpoint.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def environment(duration=1000):
    return SimpleNamespace(cfg=SimpleNamespace(recovery_slew_anneal_steps=duration,
        recovery_slew_start_rad_s=4., recovery_slew_end_rad_s=3.5), _global_step_counter=0)


class Runner:
    def save(self, path, infos=None):
        torch.save(dict(infos=infos, actor_state_dict={'w': torch.tensor([7.])},
                        iter=999), path)


def test_resume_preserves_sample_progress_and_unrelated_metadata(tmp_path):
    env=environment();runner=Runner()
    m.configure_recovery_slew_checkpoint(runner,env)
    env._global_step_counter=384
    p=tmp_path/'model.pt';infos={'stop_reason':'budget'}
    runner.save(p,infos=infos)
    assert infos=={'stop_reason':'budget'}
    restarted=environment();second=Runner()
    m.configure_recovery_slew_checkpoint(second,restarted,p,True)
    assert restarted._global_step_counter==0
    assert restarted._recovery_slew_sample_offset==384
    restarted._global_step_counter=128
    second.save(tmp_path/'next.pt',infos={'other':1})
    result=torch.load(tmp_path/'next.pt',weights_only=False)
    assert result['infos'][m.KEY]['elapsed_samples']==512
    assert result['infos']['other']==1 and result['actor_state_dict']['w'].item()==7


@pytest.mark.parametrize('prior_duration',[0,1000])
def test_legacy_only_fixed_rate_source_can_start_new_anneal(tmp_path,prior_duration):
    p=tmp_path/'model.pt';Runner().save(p)
    (tmp_path/'params').mkdir()
    (tmp_path/'params/env.yaml').write_text(f'recovery_slew_anneal_steps: {prior_duration}\n')
    if prior_duration:
        with pytest.raises(ValueError,match='lacks saved'):
            m.configure_recovery_slew_checkpoint(Runner(),environment(),p,True)
    else:
        env=environment();m.configure_recovery_slew_checkpoint(Runner(),env,p,True)
        assert env._recovery_slew_sample_offset==0


def test_mismatched_schedule_is_rejected(tmp_path):
    runner=Runner();m.configure_recovery_slew_checkpoint(runner,environment())
    p=tmp_path/'model.pt';runner.save(p)
    with pytest.raises(ValueError,match='differs'):
        m.configure_recovery_slew_checkpoint(Runner(),environment(2000),p,True)


def test_fixed_rate_does_not_wrap_or_load():
    runner=Runner();original=runner.save
    assert not m.configure_recovery_slew_checkpoint(runner,environment(0),'missing.pt',True)
    assert runner.save==original
