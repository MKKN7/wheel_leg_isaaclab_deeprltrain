# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn.functional as F

from agent_rl.rsl_rl.modules.policy import Actor, Critic, BaseModules
from agent_rl.rsl_rl.modules.nn import Identity, MLP, CustomCNN


class LeggedActor(Actor):
    orders = ("policy", "lin_vel", "ang_vel", "height", "feet_contact", "goal", "scandots", "priv_implicit")

    def __init__(
            self,
            num_observations,
            num_actions,
            goal_encoder_hidden_dims=None,
            scandots_encoder_hidden_dims=(256, 128, 32),
            priv_estimator_hidden_dims=(128, 64),
            priv_encoder_hidden_dims=(128, 64),
            actor_hidden_dims=(512, 256),
            activation="relu",
            **kwargs
    ):
        super().__init__(num_observations, num_actions)
        self.activation = activation

        self.num_policy_obs = num_observations["policy"]
        self.num_goal_obs = num_observations.get("goal", 0)
        self.num_lin_vel = num_observations.get("lin_vel", 0)
        self.num_ang_vel = num_observations.get("ang_vel", 0)
        self.num_height = num_observations.get("height", 0)
        self.num_feet_contact = num_observations.get("feet_contact", 0)
        self.num_priv_implicit = num_observations.get("priv_implicit", 0)
        self.num_scandots = num_observations.get("scandots", 0)
        
        self.num_actor_obs = self.num_policy_obs
        if self.num_lin_vel > 0:
            self.lin_vel_estimator = MLP(self.num_policy_obs, self.num_lin_vel, priv_estimator_hidden_dims, activation)
            self.num_actor_obs += self.num_lin_vel
            self._add_estimator("lin_vel_estimator", self.lin_vel_estimator)
        if self.num_ang_vel > 0:
            self.ang_vel_estimator = MLP(self.num_policy_obs, self.num_ang_vel, priv_estimator_hidden_dims, activation)
            self.num_actor_obs += self.num_ang_vel
            self._add_estimator("ang_vel_estimator", self.ang_vel_estimator)
        if self.num_height > 0:
            self.height_estimator = MLP(self.num_policy_obs, self.num_height, priv_estimator_hidden_dims, activation)
            self.num_actor_obs += self.num_height
            self._add_estimator('height_estimator', self.height_estimator)
        if self.num_feet_contact > 0:
            self.feet_contact_estimator = MLP(self.num_policy_obs, self.num_feet_contact, priv_estimator_hidden_dims,
                                              activation, "sigmoid")
            self.num_actor_obs += self.num_feet_contact
            self._add_estimator("feet_contact_estimator", self.feet_contact_estimator)
        

        if self.num_goal_obs > 0:
            if goal_encoder_hidden_dims is not None:
                self.goal_encoder = MLP(self.num_goal_obs, hidden_dims=goal_encoder_hidden_dims, activation=activation)
                goal_encoder_output_dim = goal_encoder_hidden_dims[-1]
                self.num_actor_obs += goal_encoder_output_dim
            else:
                self.goal_encoder = Identity()
                self.num_actor_obs += self.num_goal_obs

        if self.num_priv_implicit > 0:
            self.priv_encoder = MLP(self.num_priv_implicit, hidden_dims=priv_encoder_hidden_dims, activation=activation)
            priv_encoder_output_dim = priv_encoder_hidden_dims[-1]
            self.num_actor_obs += priv_encoder_output_dim

        if self.num_scandots > 0:
            self.scandots_encoder = MLP(self.num_scandots, scandots_encoder_hidden_dims[-1],
                                        scandots_encoder_hidden_dims[:-1],
                                        activation, "tanh")
            scandots_encoder_output_dim = scandots_encoder_hidden_dims[-1]
            self.num_actor_obs += scandots_encoder_output_dim
            # self.scandots_encoder = Identity()
            # self.num_actor_obs += self.num_scandots
        
        self.actor_backbone = MLP(self.num_actor_obs, self.num_actions, actor_hidden_dims, activation)

    def forward(self, obs_dict):
        inputs, outputs = dict(), dict()
        inputs["policy"] = obs_dict["policy"]
        # estimator
        with torch.inference_mode():
            if self.num_lin_vel > 0:
                lin_vel = self._infer_lin_vel_estimation(obs_dict)
                inputs["lin_vel"] = lin_vel
                outputs["lin_vel"] = lin_vel
                if torch.any(torch.isnan(lin_vel)):
                    print('WARNING: NaN lin_vel')
            if self.num_ang_vel > 0:
                ang_vel = self._infer_ang_vel_estimation(obs_dict)
                inputs["ang_vel"] = ang_vel
                outputs["ang_vel"] = ang_vel
                if torch.any(torch.isnan(ang_vel)):
                    print('WARNING: NaN ang_vel')
            if self.num_height > 0:
                height = self._infer_height_estimation(obs_dict)
                inputs["height"] = height
                outputs["height"] = height
                if torch.any(torch.isnan(height)):
                    print('WARNING: NaN height')
            if self.num_feet_contact > 0:
                feet_contact = self._infer_feet_contact_estimation(obs_dict)
                inputs["feet_contact"] = feet_contact
                outputs["feet_contact"] = feet_contact
        # encoder
        if self.num_goal_obs > 0:
            goal_latent = self._infer_goal_latent(obs_dict)
            inputs["goal"] = goal_latent
            if torch.any(torch.isnan(goal_latent)):
                    print('WARNING: NaN goal_latent')
            
        if self.num_scandots > 0:
            scandots_latent = self._infer_scandots_latent(obs_dict)
            inputs["scandots"] = scandots_latent
            if torch.any(torch.isnan(scandots_latent)):
                    print('WARNING: NaN scandots_latent')
            is_nan = torch.stack([torch.isnan(p).any() for p in  self.scandots_encoder.parameters()]).any()
            if is_nan:
                print('WARNING: NaN scandots_model weights')
        if self.num_priv_implicit > 0:
            priv_latent = self._infer_priv_latent(obs_dict)
            inputs["priv_implicit"] = priv_latent
            if torch.any(torch.isnan(priv_latent)):
                    print('WARNING: NaN priv_latent')

        outputs["actions"] = self._infer_actor_backbone(inputs)
        if torch.any(torch.isnan(outputs["actions"])):
                    print('WARNING: NaN actions')
        return outputs

    def _infer_lin_vel_estimation(self, obs_dict):
        if torch.any(torch.isnan(obs_dict["policy"])):
            print('WARNING: NaN policy input')
        return self.lin_vel_estimator(obs_dict["policy"])
    
    def _infer_ang_vel_estimation(self, obs_dict):
        if torch.any(torch.isnan(obs_dict["policy"])):
            print('WARNING: NaN policy input')
        return self.ang_vel_estimator(obs_dict["policy"])
    
    def _infer_height_estimation(self, obs_dict):
        if torch.any(torch.isnan(obs_dict["policy"])):
            print('WARNING: NaN policy input')
        return self.height_estimator(obs_dict["policy"])

    def _infer_feet_contact_estimation(self, obs_dict):
        return self.feet_contact_estimator(obs_dict["policy"])

    def _infer_goal_latent(self, obs_dict):
        return self.goal_encoder(obs_dict["goal"])

    def _infer_priv_latent(self, obs_dict):
        return self.priv_encoder(obs_dict["priv_implicit"])

    def _infer_scandots_latent(self, obs_dict):
        if torch.any(torch.isnan(obs_dict["scandots"])):
            print('WARNING: NaN scandots input')
        return self.scandots_encoder(obs_dict["scandots"])

    def _infer_actor_backbone(self, inputs):
        return self.actor_backbone(torch.cat([inputs[k] for k in self.orders if k in inputs], dim=-1))

    def dagger(self, obs_dict):
        loss_dict = dict()
        if self.num_lin_vel > 0:
            lin_vel_estimated = self._infer_lin_vel_estimation(obs_dict)
            loss_dict["lin_vel_estimator"] = F.mse_loss(lin_vel_estimated, obs_dict["lin_vel"])
        if self.num_ang_vel > 0:
            ang_vel_estimated = self._infer_ang_vel_estimation(obs_dict)
            loss_dict["ang_vel_estimator"] = F.mse_loss(ang_vel_estimated, obs_dict["ang_vel"])
        if self.num_height > 0:
            height_estimated = self._infer_height_estimation(obs_dict)
            loss_dict["height_estimator"] = F.mse_loss(height_estimated, obs_dict["height"])
        if self.num_feet_contact > 0:
            feet_contact_estimated = self.policy.actor._infer_feet_contact_estimation(obs_dict)
            loss_dict["feet_contact_estimator"] = F.binary_cross_entropy(
                feet_contact_estimated, obs_dict["feet_contact"])
        return loss_dict
    
    def do_symm(self, state_info, start_index):
        
        clipped_state_info = state_info[:,start_index:start_index + 12]
        left_state_info = clipped_state_info[:, 0::2]
        right_state_info = clipped_state_info[:, 1::2]
        stacked_state_info = torch.stack((right_state_info, left_state_info), dim=2)
        swapped_state_info = stacked_state_info.view(state_info.size(0), -1)
        #hip roll, hip yaw, ankle roll components need to be flipped 
        swapped_state_info[:, :4] *= -1
        swapped_state_info[:, 10:] *= -1
        state_info[:,  start_index:start_index + 12] = swapped_state_info
        
         
    
    def get_symm_obs(self, obs_dict):
        symm_obs = dict()
        symm_obs['policy'] = obs_dict['policy']
        self.do_symm(symm_obs['policy'],6)
        self.do_symm(symm_obs['policy'],18)
        self.do_symm(symm_obs['policy'],30)
        #roll and yaw components in base orientation and base augular vel need to be flipped 
        symm_obs['policy'][:,[0, 2, 3, 5]] *= -1
        #lateral commands and angular vel commands need to be flipped
        if self.num_goal_obs > 0:
            symm_obs['goal'] = obs_dict['goal']
            symm_obs['goal'][:,1:3] *= -1
        if self.num_scandots >0:
            symm_obs['scandots'] = obs_dict['scandots']
            batch_size = symm_obs['scandots'].size(0)
            heights_map = symm_obs['scandots'].reshape(batch_size, 17, 11)
            fliped_heights_segment = torch.flip(heights_map, dims=[2])
            symm_obs['scandots'] = fliped_heights_segment.reshape(batch_size, -1)
        return symm_obs
    
    def get_symm_action(self, action_batch):
        symm_action_ = action_batch.clone()
        self.do_symm(symm_action_,0)
        return symm_action_
    
class LeggedActorScanDotCNN(Actor):
    orders = ("policy", "lin_vel", "feet_contact", "goal", "scandots", "priv_implicit")
    
    def __init__(
        self,
        num_observations,
        num_actions,
        goal_encoder_hidden_dims=None,
        scandots_encoder_input_dims=(17,11),
        scandots_encoder_conv_layers=[
            {"out_channels":2, "kernel_size":(9,3), "stride": 1, "padding":0, "pool":None, "batch_norm": True},
            {"out_channels":4, "kernel_size":3, "stride": 1, "padding":0, "pool":None, "batch_norm": True}
        ],
        scandots_encoder_hidden_dims=(256, 128, 32),
        priv_estimator_hidden_dims=(128, 64),
        priv_encoder_hidden_dims=(128, 64),
        actor_hidden_dims=(512, 256),
        activation="relu",
        scandots_activation="lrelu",
        **kwargs
    ):
        super().__init__(num_observations, num_actions)
        self.activation = activation
        self.scandots_encoder_input_dims = scandots_encoder_input_dims

        self.num_policy_obs = num_observations["policy"]
        self.num_goal_obs = num_observations.get("goal", 0)
        self.num_lin_vel = num_observations.get("lin_vel", 0)
        self.num_feet_contact = num_observations.get("feet_contact", 0)
        self.num_priv_implicit = num_observations.get("priv_implicit", 0)
        self.num_scandots = num_observations.get("scandots", 0)

        self.num_actor_obs = self.num_policy_obs
        if self.num_lin_vel > 0:
            self.lin_vel_estimator = MLP(self.num_policy_obs, self.num_lin_vel, priv_estimator_hidden_dims, activation)
            self.num_actor_obs += self.num_lin_vel
            self._add_estimator("lin_vel_estimator", self.lin_vel_estimator)
        if self.num_feet_contact > 0:
            self.feet_contact_estimator = MLP(self.num_policy_obs, self.num_feet_contact, priv_estimator_hidden_dims,
                                              activation, "sigmoid")
            self.num_actor_obs += self.num_feet_contact
            self._add_estimator("feet_contact_estimator", self.feet_contact_estimator)

        if self.num_goal_obs > 0:
            if goal_encoder_hidden_dims is not None:
                self.goal_encoder = MLP(self.num_goal_obs, hidden_dims=goal_encoder_hidden_dims, activation=activation)
                goal_encoder_output_dim = goal_encoder_hidden_dims[-1]
                self.num_actor_obs += goal_encoder_output_dim
            else:
                self.goal_encoder = Identity()
                self.num_actor_obs += self.num_goal_obs

        if self.num_priv_implicit > 0:
            self.priv_encoder = MLP(self.num_priv_implicit, hidden_dims=priv_encoder_hidden_dims, activation=activation)
            priv_encoder_output_dim = priv_encoder_hidden_dims[-1]
            self.num_actor_obs += priv_encoder_output_dim

        if self.num_scandots > 0:
            # self.scandots_encoder = MLP(self.num_scandots, scandots_encoder_hidden_dims[-1],
            #                             scandots_encoder_hidden_dims[:-1],
            #                             activation, "tanh")
            self.scandots_encoder = CustomCNN(in_channels=1,
                                              input_dim=scandots_encoder_input_dims,
                                              output_dim=scandots_encoder_hidden_dims[-1],
                                              conv_layers=scandots_encoder_conv_layers,
                                              hidden_layers=scandots_encoder_hidden_dims[:-1],
                                              activation=scandots_activation,
                                              output_activation="tanh")
            scandots_encoder_output_dim = scandots_encoder_hidden_dims[-1]
            self.num_actor_obs += scandots_encoder_output_dim

        self.actor_backbone = MLP(self.num_actor_obs, self.num_actions, actor_hidden_dims, activation)
        
    def forward(self, obs_dict):
        inputs, outputs = dict(), dict()
        inputs["policy"] = obs_dict["policy"]
        # estimator
        with torch.inference_mode():
            if self.num_lin_vel > 0:
                lin_vel = self._infer_lin_vel_estimation(obs_dict)
                inputs["lin_vel"] = lin_vel
                outputs["lin_vel"] = lin_vel
                if torch.any(torch.isnan(lin_vel)):
                    print('WARNING: NaN lin_vel')
            
            if self.num_feet_contact > 0:
                feet_contact = self._infer_feet_contact_estimation(obs_dict)
                inputs["feet_contact"] = feet_contact
                outputs["feet_contact"] = feet_contact
        # encoder
        if self.num_goal_obs > 0:
            goal_latent = self._infer_goal_latent(obs_dict)
            inputs["goal"] = goal_latent
            if torch.any(torch.isnan(goal_latent)):
                    print('WARNING: NaN goal_latent')
            
        if self.num_scandots > 0:
            scandots_latent = self._infer_scandots_latent(obs_dict)
            inputs["scandots"] = scandots_latent
            if torch.any(torch.isnan(scandots_latent)):
                    print('WARNING: NaN scandots_latent')
            is_nan = torch.stack([torch.isnan(p).any() for p in  self.scandots_encoder.parameters()]).any()
            if is_nan:
                print('WARNING: NaN scandots_model weights')
        if self.num_priv_implicit > 0:
            priv_latent = self._infer_priv_latent(obs_dict)
            inputs["priv_implicit"] = priv_latent
            if torch.any(torch.isnan(priv_latent)):
                    print('WARNING: NaN priv_latent')

        outputs["actions"] = self._infer_actor_backbone(inputs)
        if torch.any(torch.isnan(outputs["actions"])):
                    print('WARNING: NaN actions')
        return outputs

    def _infer_lin_vel_estimation(self, obs_dict):
        if torch.any(torch.isnan(obs_dict["policy"])):
            print('WARNING: NaN policy input')
        return self.lin_vel_estimator(obs_dict["policy"])

    def _infer_feet_contact_estimation(self, obs_dict):
        return self.feet_contact_estimator(obs_dict["policy"])

    def _infer_goal_latent(self, obs_dict):
        return self.goal_encoder(obs_dict["goal"])

    def _infer_priv_latent(self, obs_dict):
        return self.priv_encoder(obs_dict["priv_implicit"])

    def _infer_scandots_latent(self, obs_dict):
        if torch.any(torch.isnan(obs_dict["scandots"])):
            print('WARNING: NaN scandots input')
        batch_size = obs_dict["scandots"].size(0)
        return self.scandots_encoder(obs_dict["scandots"].reshape([batch_size,1,
                                                                  self.scandots_encoder_input_dims[0],
                                                                  self.scandots_encoder_input_dims[1]]))

    def _infer_actor_backbone(self, inputs):
        return self.actor_backbone(torch.cat([inputs[k] for k in self.orders if k in inputs], dim=-1))

    def dagger(self, obs_dict):
        loss_dict = dict()
        if self.num_lin_vel > 0:
            lin_vel_estimated = self._infer_lin_vel_estimation(obs_dict)
            loss_dict["lin_vel_estimator"] = F.mse_loss(lin_vel_estimated, obs_dict["lin_vel"])
        if self.num_feet_contact > 0:
            feet_contact_estimated = self.policy.actor._infer_feet_contact_estimation(obs_dict)
            loss_dict["feet_contact_estimator"] = F.binary_cross_entropy(
                feet_contact_estimated, obs_dict["feet_contact"])
        return loss_dict
    
    def do_symm(self, state_info, start_index):
        
        clipped_state_info = state_info[:,start_index:start_index + 12]
        left_state_info = clipped_state_info[:, 0::2]
        right_state_info = clipped_state_info[:, 1::2]
        stacked_state_info = torch.stack((right_state_info, left_state_info), dim=2)
        swapped_state_info = stacked_state_info.view(state_info.size(0), -1)
        #hip roll, hip yaw, ankle roll components need to be flipped 
        swapped_state_info[:, :4] *= -1
        swapped_state_info[:, 10:] *= -1
        state_info[:,  start_index:start_index + 12] = swapped_state_info    
    
    def get_symm_obs(self, obs_dict):
        symm_obs = dict()
        symm_obs['policy'] = obs_dict['policy']
        self.do_symm(symm_obs['policy'],6)
        self.do_symm(symm_obs['policy'],18)
        self.do_symm(symm_obs['policy'],30)
        #roll and yaw components in base orientation and base augular vel need to be flipped 
        symm_obs['policy'][:,[0, 2, 3, 5]] *= -1
        #lateral commands and angular vel commands need to be flipped
        if self.num_goal_obs > 0:
            symm_obs['goal'] = obs_dict['goal']
            symm_obs['goal'][:,1:3] *= -1
        if self.num_scandots >0:
            symm_obs['scandots'] = obs_dict['scandots']
            batch_size = symm_obs['scandots'].size(0)
            heights_map = symm_obs['scandots'].reshape(batch_size, self.scandots_encoder_input_dims[0],
                                                                  self.scandots_encoder_input_dims[1])
            fliped_heights_segment = torch.flip(heights_map, dims=[2])
            symm_obs['scandots'] = fliped_heights_segment.reshape(batch_size, -1)
        return symm_obs
    
    def get_symm_action(self, action_batch):
        symm_action_ = action_batch.clone()
        self.do_symm(symm_action_,0)
        return symm_action_
    
class LeggedCritic(Critic):
    def __init__(self, num_observations, **kwargs):
        super().__init__(num_observations, **kwargs)

        self.num_policy_obs = num_observations["policy"]
        self.num_goal_obs = num_observations.get("goal", 0)
        self.num_lin_vel = num_observations.get("lin_vel", 0)
        self.num_ang_vel = num_observations.get("ang_vel", 0)
        self.num_height = num_observations.get("height", 0)
        self.num_feet_contact = num_observations.get("feet_contact", 0)
        self.num_priv_implicit = num_observations.get("priv_implicit", 0)
        self.num_scandots = num_observations.get("scandots", 0)

        self.num_critic_obs = self.num_policy_obs + self.num_goal_obs + self.num_lin_vel + self.num_ang_vel + self.num_height + self.num_feet_contact + self.num_priv_implicit + self.num_scandots
        self.critic_backbone = MLP(self.num_critic_obs,
                                   self.num_values,
                                   kwargs["critic_hidden_dims"],
                                   kwargs["activation"])

    def forward(self, obs_dict):
        inputs = [obs_dict["policy"]]
        output = dict()
        if self.num_goal_obs > 0:
            inputs.append(obs_dict["goal"])
        if self.num_lin_vel > 0:
            inputs.append(obs_dict["lin_vel"])
        if self.num_ang_vel > 0:
            inputs.append(obs_dict["ang_vel"])
        if self.num_height > 0:
            inputs.append(obs_dict["height"])
        if self.num_feet_contact > 0:
            inputs.append(obs_dict["feet_contact"])
        if self.num_priv_implicit > 0:
            inputs.append(obs_dict["priv_implicit"])
        if self.num_scandots > 0:
            inputs.append(obs_dict["scandots"])
        output["value"] = self.critic_backbone(torch.cat(inputs, dim=-1))
        return output

class LeggedCriticScanDotCNN(Critic):
    orders = ("policy", "lin_vel", "feet_contact", "goal", "scandots", "priv_implicit")
    
    def __init__(self, 
                num_observations,
                scandots_encoder_input_dims=(17,11),
                scandots_encoder_conv_layers=[
                    {"out_channels":2, "kernel_size":(9,3), "stride": 1, "padding":0, "pool":None, "batch_norm": True},
                    {"out_channels":4, "kernel_size":3, "stride": 1, "padding":0, "pool":None, "batch_norm": True}
                ],
                scandots_encoder_hidden_dims=(256, 128, 32), 
                scandots_activation="lrelu",
                **kwargs):
        super().__init__(num_observations, **kwargs)
        
        self.num_policy_obs = num_observations["policy"]
        self.num_goal_obs = num_observations.get("goal", 0)
        self.num_lin_vel = num_observations.get("lin_vel", 0)
        self.num_feet_contact = num_observations.get("feet_contact", 0)
        self.num_priv_implicit = num_observations.get("priv_implicit", 0)
        self.num_scandots = num_observations.get("scandots", 0)
        self.scandots_encoder_input_dims = scandots_encoder_input_dims
        
        if self.num_scandots > 0:
            self.scandots_encoder = CustomCNN(in_channels=1,
                                              input_dim=scandots_encoder_input_dims,
                                              output_dim=scandots_encoder_hidden_dims[-1],
                                              conv_layers=scandots_encoder_conv_layers,
                                              hidden_layers=scandots_encoder_hidden_dims[:-1],
                                              activation=scandots_activation,
                                              output_activation="tanh")
        
        self.num_critic_obs = self.num_policy_obs + self.num_goal_obs + self.num_lin_vel + self.num_feet_contact + self.num_priv_implicit
        scandots_encoder_output_dim = scandots_encoder_hidden_dims[-1]
        self.num_critic_obs += scandots_encoder_output_dim
        self.critic_backbone = MLP(self.num_critic_obs,
                                   self.num_values,
                                   kwargs["critic_hidden_dims"],
                                   kwargs["activation"])
        
    def _infer_scandots_latent(self, obs_dict):
        if torch.any(torch.isnan(obs_dict["scandots"])):
            print('WARNING: NaN scandots input')
        batch_size = obs_dict["scandots"].size(0)
        return self.scandots_encoder(obs_dict["scandots"].reshape([batch_size,1,
                                                                  self.scandots_encoder_input_dims[0],
                                                                  self.scandots_encoder_input_dims[1]]))
    
    def forward(self, obs_dict):
        inputs = [obs_dict["policy"]]
        output = dict()
        if self.num_goal_obs > 0:
            inputs.append(obs_dict["goal"])
        if self.num_lin_vel > 0:
            inputs.append(obs_dict["lin_vel"])
        if self.num_feet_contact > 0:
            inputs.append(obs_dict["feet_contact"])
        if self.num_priv_implicit > 0:
            inputs.append(obs_dict["priv_implicit"])
        if self.num_scandots > 0:
            scandots_latent = self._infer_scandots_latent(obs_dict)
            inputs.append(scandots_latent)
            if torch.any(torch.isnan(scandots_latent)):
                    print('WARNING: NaN scandots_latent')
            is_nan = torch.stack([torch.isnan(p).any() for p in  self.scandots_encoder.parameters()]).any()
            if is_nan:
                print('WARNING: NaN scandots_model weights')
        output["value"] = self.critic_backbone(torch.cat(inputs, dim=-1))
        return output


        
