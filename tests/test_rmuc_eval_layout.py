import importlib.util
from pathlib import Path
import random

import pytest
import torch

spec = importlib.util.spec_from_file_location('eval_layout', Path(__file__).resolve().parents[1] /
                                            'scripts/rmuc_eval_layout.py')
layout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(layout)


def test_default_preserves_old_order_and_seed_isolated():
    assert layout.bucket_assignment(3, 2) == [0, 0, 1, 1, 2, 2]
    random.seed(81)
    before = random.getstate()
    ids = layout.bucket_assignment(9, 14, 123)
    assert random.getstate() == before
    assert ids == layout.bucket_assignment(9, 14, 123)
    assert ids != layout.bucket_assignment(9, 14, 124)
    assert all(ids.count(bucket) == 14 for bucket in range(9))


def test_shuffled_episode_counts_and_thresholds_stay_with_command():
    ids = torch.tensor(layout.bucket_assignment(3, 4, 17))
    commands = torch.tensor([1., 2., 3.])[ids]
    limits = torch.tensor([.2, .15, .1])[ids]
    completed = torch.arange(12)
    counts = torch.zeros(3, dtype=torch.long).scatter_add_(0, ids, completed)
    for bucket in range(3):
        assert counts[bucket] == sum(int(completed[i]) for i in range(12) if int(ids[i]) == bucket)
        assert torch.all(commands[ids == bucket] == bucket + 1)
        assert torch.all(limits[ids == bucket] == [.2, .15, .1][bucket])


def test_invalid_counts_rejected():
    with pytest.raises(ValueError):
        layout.bucket_assignment(0, 4)
