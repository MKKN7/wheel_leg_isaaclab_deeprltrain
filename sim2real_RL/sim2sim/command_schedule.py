"""Deterministic operator command changes on policy ticks; no task feedback."""
from bisect import bisect_right
import json
from pathlib import Path

import numpy as np


class CommandSchedule:
    """Hold each explicit vx/yaw command until the next 8ms policy tick entry."""

    def __init__(self, entries):
        if not isinstance(entries, list) or not entries:
            raise ValueError('Command schedule must be a nonempty list')
        self.entries = []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {'policy_step', 'vx', 'yaw'}:
                raise ValueError('Each command requires exactly policy_step, vx, yaw')
            step = entry['policy_step']
            if type(step) is not int or step < 0:
                raise ValueError('policy_step must be a nonnegative integer')
            values = [entry['vx'], entry['yaw']]
            if any(type(v) not in (int, float) for v in values) or not np.isfinite(values).all():
                raise ValueError('vx and yaw must be finite numbers')
            if self.entries and step <= self.entries[-1]['policy_step']:
                raise ValueError('Command steps must be strictly increasing')
            self.entries.append(dict(policy_step=step, vx=float(values[0]), yaw=float(values[1])))
        if self.entries[0]['policy_step'] != 0:
            raise ValueError('First command must start at policy_step 0')
        self.steps = [entry['policy_step'] for entry in self.entries]

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def at(self, policy_step):
        if type(policy_step) is not int or policy_step < 0:
            raise ValueError('Replay policy step must be a nonnegative integer')
        index = bisect_right(self.steps, policy_step) - 1
        return index, self.entries[index]
