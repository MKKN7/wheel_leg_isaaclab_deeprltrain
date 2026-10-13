import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.utils import resolve_nn_activation

from agent_rl.rsl_rl.modules.policy import LeggedActor
from agent_rl.rsl_rl.modules.nn import VQVAE


class StateHistoryEncoder(nn.Module):
    def __init__(self, num_his_frame, state_dim, encoding_dim, activation):
        super(StateHistoryEncoder, self).__init__()

        channel_size = 10
        activation = resolve_nn_activation(activation)

        self.encoder = nn.Sequential(nn.Linear(state_dim, 3 * channel_size), activation)

        if num_his_frame == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=8, stride=4),
                activation,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                activation,
                nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=5, stride=1),
                activation, nn.Flatten())
        elif num_his_frame == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=4, stride=2),
                activation,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=2, stride=1),
                activation,
                nn.Flatten())
        elif num_his_frame == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels=3 * channel_size, out_channels=2 * channel_size, kernel_size=6, stride=2),
                activation,
                nn.Conv1d(in_channels=2 * channel_size, out_channels=channel_size, kernel_size=4, stride=2),
                activation,
                nn.Flatten())
        else:
            raise (ValueError("num_his_frame must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, encoding_dim), activation
        )

    def forward(self, obs):
        projection = self.encoder(obs)
        output = self.conv_layers(projection.permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class DeployLeggedActor(LeggedActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.num_state_his = self.num_observations["state_his"]
        self.num_his_frame = self.num_state_his // self.num_policy_obs
        if self.num_priv_implicit > 0:
            self.priv_estimator = StateHistoryEncoder(
                self.num_his_frame, self.num_policy_obs, kwargs["priv_encoder_hidden_dims"][-1], self.activation)
            self._add_estimator("priv_estimator", self.priv_estimator)

        # disable grad
        # if self.num_lin_vel > 0:
        #     self.lin_vel_estimator.requires_grad_(False)
        # if self.num_feet_contact > 0:
        #     self.feet_contact_estimator.requires_grad_(False)
        if self.num_priv_implicit > 0:
            self.priv_encoder.requires_grad_(False)

    def _infer_priv_latent(self, obs_dict):
        with torch.inference_mode():
            return self.priv_estimator(obs_dict["state_his"].view(-1, self.num_his_frame, self.num_policy_obs))

    def compute_action_loss(self, obs_dict) -> tuple[torch.Tensor, dict]:
        inputs = dict()
        inputs["policy"] = obs_dict["policy"]
        with torch.inference_mode():
            if self.num_lin_vel > 0:
                inputs["lin_vel"] = self._infer_lin_vel_estimation(obs_dict)
            if self.num_feet_contact > 0:
                inputs["feet_contact"] = self._infer_feet_contact_estimation(obs_dict)
            if self.num_goal_obs > 0:
                goal_latent = self._infer_goal_latent(obs_dict)
                inputs["goal"] = goal_latent
            if self.num_scandots > 0:
                scandots_latent = self._infer_scandots_latent(obs_dict)
                inputs["scandots"] = scandots_latent
            if self.num_priv_implicit > 0:
                priv_latent = self._infer_priv_latent(obs_dict)
                inputs["priv_implicit"] = priv_latent
            actions = super()._infer_actor_backbone(inputs)
        pred_actions = self._infer_actor_backbone({k: v.clone().detach() for k, v in inputs.items()})
        loss = F.mse_loss(pred_actions, actions.clone().detach())
        return loss, {"actor_backbone_deploy": loss}

    def dagger(self, obs_dict):
        loss_dict = super().dagger(obs_dict)

        if self.num_priv_implicit > 0:
            with torch.inference_mode():
                priv_latent = super()._infer_priv_latent(obs_dict)
            hist_latent = self.priv_estimator(obs_dict["state_his"].view(-1, self.num_his_frame, self.num_policy_obs))
            loss_dict["priv_estimator"] = F.mse_loss(hist_latent, priv_latent.clone().detach())

        return loss_dict


class VQVAEDeployLeggedActor(DeployLeggedActor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.vqvae_loss_coef = kwargs.get("vqvae_loss_coef", 0.5)
        self.vq_commitment_loss_coef = kwargs.get("vq_commitment_loss_coef", 0.25)

        state_dim = self.num_policy_obs + self.num_lin_vel + self.num_feet_contact
        goal_dim = self.num_actor_obs - state_dim
        self.actor_backbone_deploy = VQVAE(
            state_dim, goal_dim, self.num_actions,
            kwargs["num_embeddings"],
            kwargs["embedding_dim"],
            kwargs["vae_hidden_dims"],
            kwargs["activation"],
        )

        # disable grad
        self.actor_backbone.requires_grad_(False)

    def _infer_actor_backbone(self, inputs):
        state = torch.cat([inputs[k] for k in self.orders[:3] if k in inputs], dim=-1)
        goal = torch.cat([inputs[k] for k in self.orders[3:] if k in inputs], dim=-1)
        return self.actor_backbone_deploy(state, goal)

    def _compute_vq_loss(self):
        vq_loss_dict = self.actor_backbone_deploy.compute_vq_loss()
        vqvae_loss = self.vqvae_loss_coef * (
                vq_loss_dict["q_latent"] + self.vq_commitment_loss_coef * vq_loss_dict["e_latent"])
        return vqvae_loss, vq_loss_dict

    def compute_rl_loss(self):
        return self._compute_vq_loss()

    def compute_action_loss(self, obs_dict):
        loss, loss_dict = super().compute_action_loss(obs_dict)
        vqvae_loss, vq_loss_dict = self._compute_vq_loss()
        loss += vqvae_loss
        loss_dict.update(vq_loss_dict)
        return loss, loss_dict

    # def dagger(self, obs_dict):
    #     loss_dict = super().dagger(obs_dict)
    #
    #     # compute action loss
    #     loss_dict["actor"] = self.compute_action_loss(obs_dict)
    #     # compute vq loss
    #     vqvae_loss, vq_loss_dict = self._compute_vq_loss()
    #     loss_dict["actor"] += vqvae_loss
    #     for k in vq_loss_dict:
    #         loss_dict[f"vqvae_{k}"] = vq_loss_dict[k]
    #
    #     return loss_dict

from agent_rl.rsl_rl.modules.nn import EmpiricalNormalization, MLPBatchNorm, MLP

def off_diagonal(x):
    n,m = x.shape
    assert n==m
    return x.flatten()[:-1].view(n-1,n+1)[:,1:].flatten()

class MlpBarlowTwinsActor(nn.Module):
    def __init__(self,
                 num_prop,
                 num_hist,
                 num_state_est,
                 obs_encoder_dims,
                 mlp_encoder_dims,
                 actor_dims,
                 latent_dim,
                 num_actions,
                 activation) -> None:
        super(MlpBarlowTwinsActor,self).__init__()
        self.num_prop = num_prop
        self.num_hist = num_hist
        self.num_state_est = num_state_est

        self.obs_normalizer = EmpiricalNormalization(shape=num_prop)
        
        # self.cnn_encoder = CnnHistoryEncoder(num_prop,10,latent_dim)
        self.mlp_encoder = MLPBatchNorm(num_prop*num_hist, None, mlp_encoder_dims, activation, last_act=False)
        
        self.latent_layer = nn.Sequential(nn.Linear(mlp_encoder_dims[-1],32),
                                          nn.BatchNorm1d(32),
                                          nn.ELU(),
                                          nn.Linear(32,latent_dim))
        self.vel_layer = nn.Linear(mlp_encoder_dims[-1],num_state_est)

        # self.actor = MLP(num_prop + num_state_est + latent_dim, num_actions, actor_dims, activation, last_act=False)
        self.actor = MLP(num_prop + num_state_est + latent_dim, num_actions, actor_dims, activation)

        # self.actor = MixedMlp(input_size=num_prop,
        #                       latent_size=latent_dim+num_state_est,
        #                       hidden_size=128,
        #                       num_actions=num_actions,
        #                       num_experts=4)
        
        # self.vel_layer = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
        #                          input_dims=64,
        #                          out_dims=num_state_est,
        #                          hidden_dims=[32]))
        
        # self.obs_encoder = nn.Sequential(*mlp_batchnorm_factory(activation=activation,
        #                          input_dims=num_prop,
        #                          out_dims=latent_dim,
        #                          hidden_dims=[64]))
        
        self.projector = MLPBatchNorm(latent_dim, 64, [64], activation, last_act=False, bias=False)
        
        # self.history_encoder = StateHistoryEncoder(activation, num_prop, num_hist*2, 3,final_act=False)
        
        self.bn = nn.BatchNorm1d(64,affine=False)

    def normalize(self,obs,obs_hist):
        obs = self.obs_normalizer(obs)
        obs_hist = self.obs_normalizer(obs_hist.reshape(-1,self.num_prop)).reshape(-1,10,self.num_prop)
        return obs,obs_hist

    def forward(self,obs,obs_hist):
        obs,obs_hist = self.normalize(obs,obs_hist)
        # with torch.no_grad():
        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b,_,_ = obs_hist_full.size()
        # obs_hist_full = obs_hist_full[:,5:,:].view(b,-1)
        with torch.no_grad():
            latent = self.mlp_encoder(obs_hist_full[:,5:,:].reshape(b,-1))
            z = self.latent_layer(latent)
            vel = self.vel_layer(latent)
            # vel = self.history_encoder(obs_hist_full).detach()
            # #z = F.normalize(latents[:,3:],dim=-1,p=2).detach()
            # z = latents[:,3:].detach()
            # vel = latents[:,:3].detach()
        actor_input = torch.cat([vel.detach(),z.detach(),obs.detach()],dim=-1)
        mean  = self.actor(actor_input)
        # mean = self.actor(torch.cat([vel.detach(),z.detach()],dim=-1),obs.detach())
        return mean
    
    # def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
    #     obs = obs.detach()
    #     obs_hist = obs_hist.detach()
        
    #     b = obs.size()[0]

    #     obs_hist = obs_hist[:,5:,:].reshape(b,-1)

    #     latent = self.mlp_encoder(obs_hist)
    #     z1 = self.latent_layer(latent)
    #     vel = self.vel_layer(latent.detach())

    #     z2 = self.obs_encoder(obs)

    #     z1 = self.projector(z1) 
    #     z2 = self.projector(z2)

    #     c = self.bn(z1).T @ self.bn(z1)
    #     c.div_(b)

    #     on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    #     off_diag = off_diagonal(c).pow_(2).sum()

    #     priv_loss = F.mse_loss(vel,priv)

    #     loss = on_diag + weight*off_diag + priv_loss
        
    #     return loss

    def BarlowTwinsLoss(self,obs,obs_hist,priv,weight):
        obs,obs_hist = self.normalize(obs,obs_hist)

        obs = obs.detach()
        obs_hist = obs_hist.detach()

        obs_hist_full = torch.cat([
                obs_hist[:,1:,:],
                obs.unsqueeze(1)
            ], dim=1)
        b = obs.size()[0]

        # obs_hist = obs_hist[:,5:,:].reshape(b,-1)

        z1 = self.mlp_encoder(obs_hist_full[:,5:,:].reshape(b,-1))
        z2 = self.mlp_encoder(obs_hist[:,5:,:].reshape(b,-1))

        z1_l = self.latent_layer(z1)
        z1_v = self.vel_layer(z1)

        z2_l = self.latent_layer(z2)
        # z2_v = z2[:,:3]

        # z1_l = F.normalize(z1_l,dim=-1,p=2)
        # z2_l = F.normalize(z2_l,dim=-1,p=2)

        z1_l = self.projector(z1_l) 
        z2_l = self.projector(z2_l)

        c = self.bn(z1_l).T @ self.bn(z2_l)
        c.div_(b)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        
        priv_loss = F.mse_loss(z1_v,priv)

        loss = on_diag + weight*off_diag + priv_loss
        
        return loss