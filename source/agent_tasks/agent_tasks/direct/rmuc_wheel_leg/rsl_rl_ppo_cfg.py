"""RSL-RL 5.x-compatible PPO configuration for the RMUC adapter."""
from __future__ import annotations

from pathlib import Path

import yaml
from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


_CONFIG_PATH = Path(__file__).resolve().parents[5] / "configs" / "rmuc_wheel_leg.yaml"
_CONFIG = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8")) if _CONFIG_PATH.exists() else {}
_ENV = dict(_CONFIG.get("environment", {}))
_PPO = dict(_CONFIG.get("ppo", {}))


@configclass
class RmucRecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Independent fresh-skill exploration, not locomotion fine-tuning defaults."""
    class_name = 'OnPolicyRunner'
    seed = 268
    num_steps_per_env = 128
    max_iterations = 1000
    save_interval = 100
    experiment_name = 'rmuc_recovery'
    obs_groups = {'actor': ['policy'], 'critic': ['policy']}
    clip_actions = 1.0
    check_for_nan = True
    policy = RslRlPpoActorCriticCfg(init_noise_std=.15,
        actor_hidden_dims=[128,64,32],critic_hidden_dims=[256,128,64],activation='elu')
    algorithm = RslRlPpoAlgorithmCfg(value_loss_coef=.5,use_clipped_value_loss=False,
        clip_param=.2,entropy_coef=.005,num_learning_epochs=4,num_mini_batches=4,
        learning_rate=3e-4,schedule='fixed',gamma=.99,lam=.95,desired_kl=.01,max_grad_norm=1.)


@configclass
class RmucWheelLegPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Use the stock ActorCritic API shipped with Isaac Lab/RSL-RL 5.x."""

    class_name = "OnPolicyRunner"
    seed = int(_PPO.get("seed", _ENV.get("seed", 42)))
    num_steps_per_env = int(_PPO.get("num_steps_per_env", 32))
    max_iterations = int(_PPO.get("max_iterations", 20_000))
    save_interval = int(_PPO.get("save_interval", 125))
    experiment_name = "rmuc_wheel_leg_course"
    # Keep the history in the environment for diagnostics/export, while using
    # the stock feed-forward policy until a version-matched sequence model is
    # explicitly installed.
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    clip_actions = 1.0
    check_for_nan = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=float(_PPO.get("init_noise_std", 0.12)),
        actor_hidden_dims=list(_PPO.get("actor_hidden_dims", [128, 64, 32])),
        critic_hidden_dims=list(_PPO.get("critic_hidden_dims", [256, 128, 64])),
        activation=str(_PPO.get("activation", "elu")),
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=float(_PPO.get("value_loss_coef", 0.5)),
        # Rewards are raw per-control-step values; returns reach hundreds or
        # thousands. Reusing the actor's 0.1 ratio clip for absolute values
        # suppresses corrective critic gradients after reward changes.
        use_clipped_value_loss=False,
        clip_param=float(_PPO.get("clip_param", 0.2)),
        entropy_coef=float(_PPO.get("entropy_coef", 0.001)),
        num_learning_epochs=int(_PPO.get("num_learning_epochs", 4)),
        num_mini_batches=int(_PPO.get("num_mini_batches", 4)),
        learning_rate=float(_PPO.get("learning_rate", 5.0e-4)),
        schedule=str(_PPO.get("schedule", "fixed")),
        gamma=float(_PPO.get("gamma", 0.99)),
        lam=float(_PPO.get("lam", 0.95)),
        desired_kl=float(_PPO.get("desired_kl", 0.01)),
        max_grad_norm=float(_PPO.get("max_grad_norm", 1.0)),
    )
