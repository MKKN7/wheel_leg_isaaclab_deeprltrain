"""Short actor repair stages for the opted-in reverse entry course.

Each stage must return to paired behavior evaluation before another actor
stage can resume its checkpoint. This does not certify deployment behavior.
"""
import hashlib
import json
from pathlib import Path

import torch


def validate_stage_budget(total_updates, critic_warmup_updates):
    actor_updates = max(0, int(total_updates) - int(critic_warmup_updates))
    if actor_updates > 2:
        raise ValueError('Reverse entry repair permits at most2 actor updates before paired behavior review')
    return actor_updates


def validate_resume_review(payload, checkpoint_sha, review):
    if not payload.get('rmuc_short_behavior_stage', {}).get('requires_behavior_review_before_resume'):
        return
    if review is None or review.get('candidate_sha256') != checkpoint_sha:
        raise ValueError('Short-stage checkpoint requires its exact paired behavior review before actor continuation')
    if (not review.get('eligible_numeric_retention')
            or any(review.get(k) for k in ('lost', 'lost_survival', 'lost_pitch', 'lost_overshoot'))
            or not review.get('visual_review', {}).get('approved_for_continuation')):
        raise ValueError('Short-stage behavior regressed or visual motion review is incomplete')


def configure_short_behavior_stage(runner, checkpoint, total_updates, critic_warmup_updates, review_path=None):
    updates = validate_stage_budget(total_updates, critic_warmup_updates)
    if not updates:
        return
    source = Path(checkpoint)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = torch.load(source, map_location='cpu', weights_only=False)
    review = json.loads(Path(review_path).read_text()) if review_path else None
    validate_resume_review(payload, digest, review)
    metadata = {
        'requires_behavior_review_before_resume': True,
        'planned_actor_updates': updates,
        'critic_warmup_updates': int(critic_warmup_updates),
        'parent_source_sha256': digest,
        'stage_scope': 'Reverse shared-entry repair; at most2 actor updates then paired evaluation.',
        'not_deployment_certification': True,
    }
    Path(runner.logger.log_dir, 'short_behavior_stage_contract.json').write_text(json.dumps(metadata, indent=2)+'\n')
    original_save = runner.alg.save

    def save():
        result = original_save()
        result['rmuc_short_behavior_stage'] = metadata.copy()
        return result

    runner.alg.save = save
