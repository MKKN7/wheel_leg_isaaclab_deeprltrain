# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, num_observations, num_actions, **kwargs):
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.estimators = {}

        self.actor_backbone = None
        self.actor_backbone_deploy = None  # actor_backbone_deploy is a special component which should not be added into estimators

    def _add_estimator(self, name, estimator):
        self.estimators[name] = estimator

    def compute_rl_loss(self) -> tuple[torch.Tensor, dict]:
        return 0, {}

    def compute_action_loss(self, obs_dict) -> tuple[torch.Tensor, dict]:
        return 0, {}

    def dagger(self, obs_dict) -> dict:
        return {}


class Critic(nn.Module):
    def __init__(self, num_observations, num_values, **kwargs):
        super().__init__()

        self.num_observations = num_observations
        self.num_values = num_values
        self.critic_backbone = None
        
class BaseModules(nn.Module):
    forward_func = dict()
    loss_func = dict()
    estimators = dict()
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def regist_modules(self, 
                        actor_backbone,
                        backbone_obs_name:list,
                        backbone_out_name:str,
                        encoders:dict,
                        encoders_public:list,
                        encoders_private:list,
                        estimators_name:list,):
        """
        ### Summary:
            regist rl policy modules
        ### Details:
            basic structure: 
                obs ->
                multiple encoder(high level) ->
                backbong(low level) ->
                actions
        ### Args:
            actor_backbone (_type_): 
                low level backbone type
            backbone_obs_name (list): 
                backbone observation name(config tag)
            backbone_out_name (list): 
                low level backbone output tensor name
            encoders (dict): 
                a dictionary contains all high level encoders, including public encoders, private encoders, estimators
            encoders_public (list): 
                a list contains encoders' names that join public backward
            encoders_private (list): 
                a list contains encoders' names that backward itself
            estimators_name (list): 
                a list contains estimators' names
        """
        self.actor_backbone = actor_backbone
        self.actor_backbone_deploy = None
        self.backbone_obs_name = backbone_obs_name
        self.backbone_out_name = backbone_out_name
        
        self.encoders = encoders
        self.encoders_public = encoders_public
        self.encoders_private = encoders_private
        self.estimators_name = estimators_name
        self.backbone_name = 'backbone'
        
        for name in self.estimators_name:
            self.estimators[name] = self.encoders[name]
        
    def regist_estimator_loss(self, name:str, loss_func):
        """
        ### Summary:
            regist estimator loss functions
            
        ### Examples:
            def _infer_estimator_loss(self, ref, cur):
                ...
                return ...

        ### Args:
            name (str): 
                the key of the estimator
            loss_func (_type_): 
                function itself
        """
        self.loss_func[name] = loss_func
            
    def regist_encoder_forward(self, name:str, forward_func):
        """
        ### Summary:
            regist encoder forward functions
            
        ### Examples:
            def _infer_encoder_forward(self, inputs):
                ...
                return ...

        ### Args:
            name (str): 
                the key of the encoder
            loss_func (_type_): 
                function itself
        """
        self.forward_func[name] = forward_func
        
    def regist_backbone_forward(self, forward_func):
        """
        ### Summary:
            regist backbone forward functions
            
        ### Examples:
            def _infer_backbone_forward(self, inputs):
                ...
                return ...

        ### Args:
            loss_func (_type_): 
                function itself
        """
        self.forward_func[self.backbone_name] = forward_func
        
    def forward(self, obs_dict):
        inputs, outputs = dict(), dict()
        for name in self.backbone_obs_name:
            inputs[name] = obs_dict[name]
        # private update encoder
        with torch.inference_mode():
            for name in self.estimators_name:
                if name in self.forward_func.keys():
                    # get_est = self.forward_func[name](self.encoders[name],obs_dict['policy']) # custom forward
                    get_est = self.forward_func[name](obs_dict['policy']) # custom forward
                else:
                    get_est = self.encoders[name](obs_dict['policy']) # default forward
                inputs[name] = get_est
                outputs[name] = get_est
                if torch.any(torch.isnan(get_est)):
                    print('WARNING: Nan in estimator',name)
                    
            for name in self.encoders_private:
                if name in self.forward_func.keys():
                    # get_private = self.forward_func[name](self.encoders[name],obs_dict[name])
                    get_private = self.forward_func[name](obs_dict[name])
                else:
                    get_private = self.encoders[name](obs_dict[name])
                inputs[name] = get_private
                if torch.any(torch.isnan(get_private)):
                    print('WARNING: Nan in private encoder ',name)
            
        # public update encodder
        for name in self.encoders_public:
            if name in self.forward_func.keys():
                # get_public = self.forward_func[name](self.encoders[name],obs_dict[name])
                get_public = self.forward_func[name](obs_dict[name])
            else:
                get_public = self.encoders[name](obs_dict[name])
            inputs[name] = get_public
            if torch.any(torch.isnan(get_public)):
                print('WARNING: Nan in public encoder ',name)
        orders:list = self.backbone_obs_name+self.estimators_name+self.encoders_public
        if self.backbone_name in self.forward_func.keys():
            outputs[self.backbone_out_name] = self.forward_func[self.backbone_name](self.actor_backbone,torch.cat([inputs[k] for k in orders if k in inputs], dim=-1))
        else:
            outputs[self.backbone_out_name] = self.actor_backbone(torch.cat([inputs[k] for k in orders if k in inputs], dim=-1))
        
        if torch.any(torch.isnan(outputs[self.backbone_out_name])):
            print('WARNING: Nan in backbone')
        return outputs
    
    def dagger(self, obs_dict) -> dict:
        """
        ### Summary:
            calculate estimators' loss

        ### Args:
            obs_dict (_type_):
                observation dictionary

        ### Returns:
            loss_dict(dict)
        """
        loss_dict = dict()
        for name in self.estimators_name:
            if name in self.forward_func.keys():
                # get_est = self.forward_func[name](self.encoders[name],obs_dict['policy'])
                get_est = self.forward_func[name](obs_dict['policy'])
            else:
                get_est = self.encoders[name](obs_dict['policy'])
            if name in self.loss_func.keys():
                loss_dict[name] = self.loss_func[name](obs_dict[name],get_est)
        return loss_dict
    
    def private_loss(self, obs_dict) -> dict:
        """
        ### Summary:
            calculate private encoders' loss

        ### Args:
            obs_dict (_type_):
                observation dictionary

        ### Returns:
            loss_dict(dict)
        """
        loss_dict = dict()
        for name in self.encoders_private:
            if name in self.forward_func.keys():
                # get_private = self.forward_func[name](self.encoders[name],obs_dict[name])
                get_private = self.forward_func[name](obs_dict[name])
            else:
                get_private = self.encoders[name](obs_dict[name])
            if name in self.loss_func.keys():
                loss_dict[name] = self.loss_func[name](obs_dict[name],get_private)
        return loss_dict
    
    def compute_rl_loss(self) -> tuple[torch.Tensor, dict]:
        return 0, {}

    def compute_action_loss(self, obs_dict) -> tuple[torch.Tensor, dict]:
        return 0, {}
            
            
        
