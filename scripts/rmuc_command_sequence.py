"""CPU entrypoint to the shared train/evaluation sequence contract."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / 'source/agent_tasks/agent_tasks/direct/rmuc_wheel_leg/command_sequence.py'
_spec = importlib.util.spec_from_file_location('rmuc_shared_command_sequence', _path)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
validate_sequence = _module.validate_sequence
sequence_commands = _module.sequence_commands
dynamic_tracking_cost = _module.dynamic_tracking_cost
tracking_cohort_mask = _module.tracking_cohort_mask
dynamic_stop_posture_cost = _module.dynamic_stop_posture_cost
