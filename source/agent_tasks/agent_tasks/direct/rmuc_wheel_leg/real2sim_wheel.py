"""Opt-in wheel identification sensitivity model; never certifies hardware.

Physical Left/Right corresponds to policy [right_wheel_spin, left_wheel_spin].
The feedback_sensor hypothesis delays low-level wheel feedback ONLY; it does
not delay the policy's 25D observation. Transport delay consumes already
clipped electromagnetic torque. Mechanical B/C remain outside this module's
motor path, applied through the environment's existing passive body wrenches.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import torch


PAIR_FIELDS = (
    "mechanical_viscous_nm_s_rad", "coulomb_friction_nm",
    "damping_compensation_nm_s_rad", "coulomb_compensation_nm", "delay_s",
    "equivalent_inertia_kg_m2_metadata",
)
REQUIRED = {
    "schema_version", "scope", "physical_order", "wheel_joint_order",
    "compensation_enabled", "friction_epsilon_rad_s", "delay_mode",
    "identification_status", *PAIR_FIELDS,
}


def validate_profile(profile):
    """Reject ambiguous side ordering, unsupported units and silent metadata."""
    if not isinstance(profile, dict) or not REQUIRED <= set(profile):
        raise ValueError("Real2sim wheel profile is missing required fields")
    if set(profile) - REQUIRED - {"notes", "armature_override_kg_m2"}:
        raise ValueError("Unknown real2sim wheel profile fields")
    if type(profile["schema_version"]) is not int or profile["schema_version"] != 1:
        raise ValueError("Real2sim wheel profile requires schema_version=1")
    if profile["scope"] != "real2sim_diagnostic":
        raise ValueError("Real2sim wheel profile remains diagnostic")
    if profile["physical_order"] != ["Left", "Right"]:
        raise ValueError("Physical wheel order must be Left, Right")
    if profile["wheel_joint_order"] != ["right_wheel_spin", "left_wheel_spin"]:
        raise ValueError("Policy wheel names must follow physical Left, Right")
    if profile["identification_status"] != "reported_fit_not_independently_validated":
        raise ValueError("Identification remains reported and independently unvalidated")
    if type(profile["compensation_enabled"]) is not bool:
        raise ValueError("compensation_enabled must be boolean")
    for field in PAIR_FIELDS:
        pair = profile[field]
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in pair)):
            raise ValueError(f"{field} must contain two finite nonnegative SI values")
    if any(v <= 0 for v in profile["equivalent_inertia_kg_m2_metadata"]):
        raise ValueError("Reported equivalent inertia metadata must be positive")
    if "armature_override_kg_m2" in profile:
        pair = profile["armature_override_kg_m2"]
        if (not isinstance(pair, list) or len(pair) != 2
                or any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in pair)):
            raise ValueError("armature_override_kg_m2 must contain two finite nonnegative SI values")
    epsilon = profile["friction_epsilon_rad_s"]
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("Friction smoothing speed must be finite positive rad/s")
    if profile["delay_mode"] not in ("none", "motor_transport", "feedback_sensor"):
        raise ValueError("Unsupported delay hypothesis")
    if any(v > .012 for v in profile["delay_s"]):
        raise ValueError("Diagnostic delay must be in [0, 12] ms; total pipeline delay requires separate verification")
    if not profile["compensation_enabled"] and any(
        v != 0 for field in ("damping_compensation_nm_s_rad", "coulomb_compensation_nm")
        for v in profile[field]
    ):
        raise ValueError("Disabled compensation requires zero active coefficients")
    if "notes" in profile and (not isinstance(profile["notes"], list)
                                or not all(isinstance(v, str) for v in profile["notes"])):
        raise ValueError("notes must be a list of strings")
    return deepcopy(profile)


def load_profile(path):
    return validate_profile(json.loads(Path(path).read_text()))


def configure_total_delay(cfg, profile):
    """Assign the total wheel latency once; leave sensor noise untouched."""
    profile = validate_profile(profile)
    if profile['delay_mode'] not in ('none', 'motor_transport'):
        raise ValueError('Observed-to-execution delay cannot use feedback-only mode')
    cfg.observation_delay_steps = 0
    cfg.domain_randomization_max_observation_delay_steps = 0
    cfg.domain_randomization_max_observation_delay_probability = 0.
    cfg.domain_randomization_sensor_model = dict(cfg.domain_randomization_sensor_model,
        observation_delay_probability=0., communication_delay_probability=0.,
        action_delay_enabled=False)


class _FractionalDelay:
    """Causal linear interpolation of two retained physics-step samples."""

    def __init__(self, template, dt, delay_s):
        ratios = [float(v) / dt for v in delay_s]
        # Avoid classifying exact millisecond multiples as fractional due to FP.
        ratios = [float(round(v)) if abs(v - round(v)) < 1e-9 else v for v in ratios]
        floors = [math.floor(v) for v in ratios]
        fractions = [v - floor for v, floor in zip(ratios, floors)]
        self.floor = torch.tensor(floors, device=template.device, dtype=torch.long)
        self.fraction = torch.tensor(fractions, device=template.device, dtype=template.dtype)
        self.floors = floors
        self.fractions = fractions
        self.history = template.new_zeros((template.shape[0], max(floors) + 2, 2))

    def step(self, value):
        self.history = torch.roll(self.history, shifts=1, dims=1)
        self.history[:, 0] = value
        sides = torch.arange(2, device=value.device)
        recent = self.history[:, self.floor, sides]
        older = self.history[:, self.floor + 1, sides]
        return recent * (1 - self.fraction) + older * self.fraction

    def reset(self, env_ids):
        if env_ids is None:
            self.history.zero_()
        else:
            self.history[env_ids] = 0


class Real2SimWheel:
    """Bounded-delay controller/plant sensitivity helper, disabled by default.

Caller supplies physical-step wheel speeds and clipped electromagnetic motor
efforts in identical policy joint coordinates. Call each method once per
physics step. This class does not modify geometry, inertia, other observations,
motor limits or mechanical drag, and does not estimate parameters online.
"""

    def __init__(self, template, dt, profile):
        self.profile = validate_profile(profile)
        if type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be finite and positive")
        if template.ndim != 2 or template.shape[1] != 2 or not template.is_floating_point():
            raise ValueError("Wheel template must have floating shape [num_envs, 2]")
        self.dt = float(dt)
        self.shape = template.shape
        self.device, self.dtype = template.device, template.dtype
        for attribute, field in (
            ("mechanical_viscous", "mechanical_viscous_nm_s_rad"),
            ("coulomb", "coulomb_friction_nm"),
            ("damping_compensation", "damping_compensation_nm_s_rad"),
            ("coulomb_compensation", "coulomb_compensation_nm"),
        ):
            setattr(self, attribute, template.new_tensor(self.profile[field]).reshape(1, 2))
        self.epsilon = float(self.profile["friction_epsilon_rad_s"])
        self.mode = self.profile["delay_mode"]
        active_delay = self.profile["delay_s"] if self.mode != "none" else [0., 0.]
        self._delay = _FractionalDelay(template, self.dt, active_delay)
        self.last_feedback_omega = torch.zeros_like(template)
        self.last_feedback_torque = torch.zeros_like(template)
        self.last_motor_input = torch.zeros_like(template)
        self.last_motor_output = torch.zeros_like(template)
        self.feedback_calls = 0
        self.motor_calls = 0
        self._motor_peak_hits = template.new_zeros(2)
        self._motor_peak_abs = template.new_zeros(2)
        self._delayed_motor_peak_hits = template.new_zeros(2)
        self._delayed_motor_peak_abs = template.new_zeros(2)
        self._motor_input_abs_sum = template.new_zeros(2)
        self._motor_output_abs_sum = template.new_zeros(2)
        self._motor_samples = 0

    def _check(self, value):
        if value.shape != self.shape or value.device != self.device or value.dtype != self.dtype:
            raise ValueError("Wheel signal must match template shape, device and dtype")

    def feedback(self, omega):
        """Active D plus Coulomb feedforward, to be included BEFORE motor clip."""
        self._check(omega)
        observed = self._delay.step(omega) if self.mode == "feedback_sensor" else omega
        result = (-self.damping_compensation * observed
                  + self.coulomb_compensation * torch.tanh(observed / self.epsilon))
        self.last_feedback_omega.copy_(observed)
        self.last_feedback_torque.copy_(result)
        self.feedback_calls += 1
        return result

    def motor_effort(self, effort):
        """Delay already-clipped motor output; never clips passive mechanics."""
        self._check(effort)
        result = self._delay.step(effort) if self.mode == "motor_transport" else effort
        self.last_motor_input.copy_(effort)
        self.last_motor_output.copy_(result)
        # Count contact with the peak, not asserted clipping: preclip demand is
        # intentionally unavailable here. Keep reductions on device until readback.
        self._motor_peak_hits.add_((effort.abs() >= 5.5 - 1e-6).sum(dim=0))
        self._motor_peak_abs.copy_(torch.maximum(self._motor_peak_abs, effort.abs().amax(dim=0)))
        self._delayed_motor_peak_hits.add_((result.abs() >= 5.5 - 1e-6).sum(dim=0))
        self._delayed_motor_peak_abs.copy_(torch.maximum(self._delayed_motor_peak_abs, result.abs().amax(dim=0)))
        self._motor_input_abs_sum.add_(effort.abs().sum(dim=0))
        self._motor_output_abs_sum.add_(result.abs().sum(dim=0))
        self._motor_samples += effort.shape[0]
        self.motor_calls += 1
        return result

    def reset(self, env_ids):
        """Zero histories only for resetting environments; retain run statistics."""
        self._delay.reset(env_ids)
        for field in (self.last_feedback_omega, self.last_feedback_torque,
                      self.last_motor_input, self.last_motor_output):
            if env_ids is None:
                field.zero_()
            else:
                field[env_ids] = 0

    def readback(self):
        return {
            "profile": deepcopy(self.profile),
            "physics_dt_s": self.dt,
            "effective_delay_s": self.profile["delay_s"] if self.mode != "none" else [0., 0.],
            "configured_delay_cap_s": .012,
            "delay_cap_scope": "selected_hypothesis_only_total_observation_to_execution_pipeline_requires_separate_verification",
            "delay_floor_steps": list(self._delay.floors),
            "delay_interpolation_fraction": list(self._delay.fractions),
            "delay_discretization": ("causal_two_tap_fractional_approximation"
                                     if any(self._delay.fractions) else "integer_physics_steps"),
            "delay_hypothesis_scope": {
                "none": "no_additional_identified_delay",
                "feedback_sensor": "low_level_feedback_only_policy_observation_unchanged",
                "motor_transport": "clipped_electromagnetic_command_only",
            }[self.mode],
            "inertia_applied": False,
            "inertia_application_scope": "helper_metadata_only_engine_override_requires_separate_runtime_readback",
            "armature_override_requested_kg_m2": deepcopy(self.profile.get("armature_override_kg_m2")),
            "armature_override_interpretation": (
                "nominal_joint_equivalent_approximation_not_exact_fixed_parent_identification"
                if "armature_override_kg_m2" in self.profile else "not_requested"
            ),
            "friction_smoothing_identified": False,
            "feedback_calls": self.feedback_calls,
            "motor_calls": self.motor_calls,
            "motor_samples_per_side": self._motor_samples,
            "motor_peak_hit_fraction_by_side": (
                (self._motor_peak_hits / self._motor_samples).cpu().tolist()
                if self._motor_samples else [None, None]
            ),
            "motor_max_abs_input_nm_by_side": self._motor_peak_abs.cpu().tolist(),
            "motor_max_abs_output_nm_by_side": self._delayed_motor_peak_abs.cpu().tolist(),
            "motor_output_peak_hit_fraction_by_side": (
                (self._delayed_motor_peak_hits / self._motor_samples).cpu().tolist()
                if self._motor_samples else [None, None]
            ),
            "motor_mean_abs_input_nm_by_side": (
                (self._motor_input_abs_sum / self._motor_samples).cpu().tolist()
                if self._motor_samples else [None, None]
            ),
            "motor_mean_abs_output_nm_by_side": (
                (self._motor_output_abs_sum / self._motor_samples).cpu().tolist()
                if self._motor_samples else [None, None]
            ),
            "motor_peak_statistic_is_actual_clipping_count": False,
            "parameter_drift_identified": False,
            "behavior_certified": False,
        }
