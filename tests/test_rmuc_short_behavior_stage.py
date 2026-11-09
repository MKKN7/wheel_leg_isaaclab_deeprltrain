from types import SimpleNamespace
import hashlib,json
import pytest
import torch
from rmuc_short_behavior_stage import validate_stage_budget,validate_resume_review,configure_short_behavior_stage


def test_old_eight_update_recipe_is_rejected_but_two_actor_stage_is_allowed():
    with pytest.raises(ValueError,match='at most2'):validate_stage_budget(11,3)
    assert validate_stage_budget(5,3)==2
    assert validate_stage_budget(3,3)==0


def test_resume_requires_exact_hash_numeric_retention_and_visual_review():
    payload={'rmuc_short_behavior_stage':{'requires_behavior_review_before_resume':True}}
    valid={'candidate_sha256':'current','eligible_numeric_retention':True,'lost':[],
           'lost_survival':[],'lost_pitch':[],'visual_review':{'approved_for_continuation':True}}
    validate_resume_review({},'current',None)
    validate_resume_review(payload,'current',valid)
    for bad in [None,{**valid,'candidate_sha256':'other'},
                {**valid,'lost':[3]},{**valid,'lost_survival':[1]},
                {**valid,'lost_overshoot':[2]},{**valid,'eligible_numeric_retention':False},
                {**valid,'visual_review':{'approved_for_continuation':False}}]:
        with pytest.raises(ValueError):validate_resume_review(payload,'current',bad)


def test_save_marks_pending_behavior_review_without_changing_network(tmp_path):
    cp=tmp_path/'source.pt';torch.save({'actor_state_dict':{'weight':torch.ones(2)}},cp)
    original={'actor_state_dict':{'weight':torch.ones(2)}}
    runner=SimpleNamespace(logger=SimpleNamespace(log_dir=str(tmp_path)),alg=SimpleNamespace(save=lambda:original.copy()))
    configure_short_behavior_stage(runner,cp,5,3)
    saved=runner.alg.save()
    assert torch.equal(saved['actor_state_dict']['weight'],original['actor_state_dict']['weight'])
    assert saved['rmuc_short_behavior_stage']['planned_actor_updates']==2
    assert saved['rmuc_short_behavior_stage']['requires_behavior_review_before_resume']
    assert saved['rmuc_short_behavior_stage']['parent_source_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
    assert json.loads((tmp_path/'short_behavior_stage_contract.json').read_text())==saved['rmuc_short_behavior_stage']
