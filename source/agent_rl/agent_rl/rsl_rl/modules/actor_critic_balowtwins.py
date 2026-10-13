# =============================================================================
# This file contains code derived from the following third-party projects.
#
#   [rsl_rl / legged_gym]
#     License  : BSD-3-Clause
#     Copyright: Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
#
#   [ddt_rl_isaacgym]
#     URL      : https://github.com/DDTRobot/ddt_rl_isaacgym
#
# =============================================================================

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic

from agent_rl.rsl_rl.modules.policy import *  # noqa: F401
from agent_rl.rsl_rl.modules.nn import *

class ActorCriticBarlowTwins(nn.Module):
    is_recurrent = False
    def __init__(self,  num_prop,
                        num_scan,
                        num_state_est,
                        num_priv_latent, 
                        num_hist,
                        num_actions,
                        scan_encoder_dims=[256, 256, 256],
                        actor_hidden_dims=[256, 256, 256],
                        critic_hidden_dims=[256, 256, 256],
                        hist_encoder=False,
                        activation='elu',
                        init_noise_std=1.0,
                        fixed_std=False,
                        action_mean_clip: float = 20.0,
                        **kwargs):
        super(ActorCriticBarlowTwins, self).__init__()

        self.kwargs = kwargs
        priv_encoder_dims= kwargs['priv_encoder_dims']
        cost_dims = kwargs['num_costs']
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.num_state_est = num_state_est
        self.num_priv_latent = num_priv_latent
        self.action_mean_clip = action_mean_clip
        self.if_scan_encode = scan_encoder_dims is not None and num_scan > 0
        self.hist_encoder = hist_encoder

        # n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        self.num_obs = num_prop + num_scan + num_priv_latent + num_hist * num_prop
        self.obs_normalize = EmpiricalNormalization(self.num_obs)

        self.teacher_act = kwargs['teacher_act']
        if self.teacher_act:
            print("ppo with teacher actor")
        else:
            print("ppo with teacher actor")

        self.imi_flag = kwargs['imi_flag']
        if self.imi_flag:
            print("run imitation")
        else:
            print("no imitation")

        if len(priv_encoder_dims) > 0:
            # self.priv_encoder = MLP(num_priv_latent,None,priv_encoder_dims,activation,last_act=True)
            self.priv_encoder = MLP(num_priv_latent,None,priv_encoder_dims,activation)
            priv_encoder_output_dim = priv_encoder_dims[-1]
        else:
            self.priv_encoder = nn.Identity()
            priv_encoder_output_dim = num_priv_latent

        if self.if_scan_encode:
            # scan_encoder_layers = mlp_factory(activation,num_scan,None,scan_encoder_dims,last_act=True)
            # self.scan_encoder = nn.Sequential(*scan_encoder_layers)
            # self.scan_encoder_output_dim = scan_encoder_dims[-1]
            # self.scan_encoder = MLP(num_scan,scan_encoder_dims[-1],scan_encoder_dims[:-1],activation,last_act=False)
            self.scan_encoder = MLP(num_scan,scan_encoder_dims[-1],scan_encoder_dims[:-1],activation)
            self.scan_encoder_output_dim = scan_encoder_dims[-1]
        else:
            print("not using scan encoder")
            self.scan_encoder = nn.Identity()
            self.scan_encoder_output_dim = num_scan

        self.history_encoder = StateHistoryEncoder(num_hist, num_prop, 16, activation)

        # #MlpBarlowTwinsActor
        self.actor_teacher_backbone = MlpBarlowTwinsActor(num_prop=num_prop,
                                      num_hist=5,
                                      num_state_est=num_state_est,
                                      num_actions=num_actions,
                                      actor_dims=[512,256,128],
                                      mlp_encoder_dims=[128,64],
                                      activation=activation,
                                      latent_dim=16,
                                      obs_encoder_dims=[128,64])
        
        print(self.actor_teacher_backbone)

        # Value function
        # self.critic = MLP(num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,1,critic_hidden_dims,activation,last_act=False)
        if self.hist_encoder:
            self.critic = MLP(num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim+16,1,critic_hidden_dims,activation)
        else:
            self.critic = MLP(num_prop+self.scan_encoder_output_dim+priv_encoder_output_dim,1,critic_hidden_dims,activation)
        print(self.critic)

        # cost function：与 mlp_factory(..., last_act=False) 一致，末尾再接 Softplus
        if self.hist_encoder:
            self.cost = nn.Sequential(
                MLP(
                    num_prop + self.scan_encoder_output_dim + priv_encoder_output_dim + 16,
                    cost_dims,
                    critic_hidden_dims,
                    activation,
                    output_activation="identity",
                ),
                nn.Softplus(),
            )
        else:
            self.cost = nn.Sequential(
                MLP(
                    num_prop + self.scan_encoder_output_dim + priv_encoder_output_dim,
                    cost_dims,
                    critic_hidden_dims,
                    activation,
                    output_activation="identity",
                ),
                nn.Softplus(),
            )

        # Action noise
        self.fixed_std = fixed_std
        std = init_noise_std * torch.ones(num_actions)
        self.std = torch.tensor(std) if fixed_std else nn.Parameter(std)
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args(False)

        
    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
        
    def set_teacher_act(self,flag):
        self.teacher_act = flag
        if self.teacher_act:
            print("acting by teacher")
        else:
            print("acting by student")

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    def _sanitize_tensor(self, tensor, clip: float | None = None):
        if clip is None:
            return torch.nan_to_num(tensor, nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clip, neginf=-clip)
        return torch.clamp(tensor, -clip, clip)
    
    def get_std(self):
        return self.std
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, obs):
        mean = self._sanitize_tensor(self.act_teacher(obs), self.action_mean_clip)
        std = torch.nan_to_num(self.get_std().expand_as(mean), nan=1.0, posinf=10.0, neginf=1.0e-6).clamp_min(1.0e-6)
        self.distribution = Normal(mean, std)

    def act(self, obs,**kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_teacher(self,obs, **kwargs):
        obs = self._sanitize_tensor(obs)
        # obs_prop = obs[:, :self.num_prop]
        # obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)[:,:,:]
        mean = self.actor_teacher_backbone(obs_prop,obs_hist)
        return mean
        
    def evaluate(self, obs, **kwargs):
        obs = self.obs_normalize(self._sanitize_tensor(obs))

        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        if self.hist_encoder:
            history_latent = self.infer_hist_latent(obs)
            backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        else:
            backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)

        value = self.critic(backbone_input)
        return value
    
    def evaluate_cost(self,obs, **kwargs):
        obs = self.obs_normalize(self._sanitize_tensor(obs))

        obs_prop = obs[:, :self.num_prop]
        
        scan_latent = self.infer_scandots_latent(obs)
        latent = self.infer_priv_latent(obs)
        if self.hist_encoder:
            history_latent = self.infer_hist_latent(obs)
            backbone_input = torch.cat([obs_prop,latent,scan_latent,history_latent], dim=1)
        else:
            backbone_input = torch.cat([obs_prop,latent,scan_latent], dim=1)
        value = self.cost(backbone_input)
        return value
    
    def infer_priv_latent(self, obs):
        priv = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_priv_latent]
        return self.priv_encoder(priv)
    
    def infer_scandots_latent(self, obs):
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        return self.scan_encoder(scan)
    
    def infer_hist_latent(self, obs):
        hist = obs[:, -self.num_hist*self.num_prop:]
        return self.history_encoder(hist.view(-1, self.num_hist, self.num_prop))
    
    def imitation_learning_loss(self, obs,imi_weight=1):
        obs = self._sanitize_tensor(obs)
        # obs_prop = obs[:, :self.num_prop]
        # obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist*self.num_prop:].view(-1, self.num_hist, self.num_prop)
        scan = obs[:, self.num_prop:self.num_prop + self.num_scan]
        # contact = obs[:, self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_state_est]
        # vel = obs_hist[:,-1,:3]

        # priv = torch.cat([contact,vel],dim=-1)
        priv = obs[:,self.num_prop + self.num_scan: self.num_prop + self.num_scan + self.num_state_est]

        loss = self.actor_teacher_backbone.BarlowTwinsLoss(obs_prop,obs_hist,priv,5e-3)
        # loss = self.actor_teacher_backbone.SimSiamLoss(obs_prop,obs_hist[:,:,3:],priv,scan)
        # loss = self.actor_teacher_backbone.VaeLoss(obs_prop,obs_hist[:,:,3:],priv,scan)
        # loss = self.actor_teacher_backbone.VaeLoss(obs_prop,obs_hist[:,:,3:],priv)
        #loss = self.actor_teacher_backbone.maeLoss(obs_prop,obs_hist,priv)
        # loss = recon_loss + kl_loss + mseloss
        return loss
    
    def imitation_mode(self):
        pass
    
    # def save_torch_jit_policy(self, checkpoint_path, device):
    #     """Export TorchScript and ONNX next to the training checkpoint (``model.pt`` / ``model.onnx``)."""
    #     self.actor_teacher_backbone.eval()
    #     export_dir = _resolve_policy_export_dir(checkpoint_path)
    #     out_pt = os.path.join(export_dir, "model.pt")
    #     out_onnx = os.path.join(export_dir, "model.onnx")

    #     obs_demo_input = torch.randn(1, self.num_prop - 3).half().to(device)
    #     hist_demo_input = torch.randn(1, self.num_hist, self.num_prop - 3).half().to(device)
    #     model_jit = torch.jit.trace(
    #         self.actor_teacher_backbone, (obs_demo_input, hist_demo_input)
    #     )
    #     model_jit.save(out_pt)
    #     obs_demo_input = torch.randn(1, self.num_prop - 3).to(device)
    #     hist_demo_input = torch.randn(1, self.num_hist, self.num_prop - 3).to(device)
    #     torch.onnx.export(
    #         self.actor_teacher_backbone,
    #         (obs_demo_input, hist_demo_input),
    #         out_onnx,
    #         input_names=["nn_input0", "nn_input1"],
    #         output_names=["nn_output"],
    #         verbose=False,
    #         opset_version=13,
    #         export_params=True,
    #     )
