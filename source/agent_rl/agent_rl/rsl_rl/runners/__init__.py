"""Optional custom runners.

The stock Isaac Lab/RSL-RL runner is sufficient for the RMUC task.  The
historical DreamWaQ/HIM/NP3O runners target an older ``rsl_rl.modules`` API,
so importing them must not prevent standard PPO tasks from starting on newer
RSL-RL releases.
"""

import logging

_LOGGER = logging.getLogger(__name__)
__all__: list[str] = []

for _module_name, _class_name in (
    ("on_policy_runner_dreamwaq", "OnPolicyDreamWaqRunner"),
    ("on_policy_runner_him", "OnPolicyHIMRunner"),
    ("on_constraint_policy_runner", "OnConstraintPolicyRunner"),
):
    try:
        _module = __import__(f"{__name__}.{_module_name}", fromlist=[_class_name])
        globals()[_class_name] = getattr(_module, _class_name)
        __all__.append(_class_name)
    except (ImportError, AttributeError) as exc:
        _LOGGER.warning("Skipping optional %s runner: %s", _class_name, exc)
