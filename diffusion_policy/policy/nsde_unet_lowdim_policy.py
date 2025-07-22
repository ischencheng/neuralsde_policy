from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import torchsde
import copy

from diffusion_policy.model.diffusion.sde import NeuralSDE
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.network import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class NSDEUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            # task params
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            # SDE parameters
            func_type: str,
            # UNet model
            model: ConditionalUnet1D,
            # parameters passed to step
            num_inference_steps=1,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # sde model
            f_ckpt_path: Optional[str] = None,
            d_ckpt_path: Optional[str] = None,
            denoising_magnitude=1.0,
            diffusion_magnitude=1.0,
            noise_std=0.01,
            logit_range=(-10,-1),
            delta_t=1.0,
            discrete_dims=None,
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        # saved parameters
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs
        self.func_type = func_type
        self.delta_t = delta_t
        self.noise_std = noise_std
        self.discrete_dims = discrete_dims
        
        self.f_model = None
        self.d_model = None
        self.g_model = None
        model.horizon=n_obs_steps
        if hasattr(model, 'global_cond_dim'):
            model.global_cond_dim = n_obs_steps*obs_dim

        if self.func_type == 'f':
            self.f_model = model
        elif self.func_type == 'd':
            self.d_model = model
            self.f_model = copy.deepcopy(model)
            if f_ckpt_path is None:
                raise ValueError("f_ckpt_path must be provided when training the diffusion term (d).")
            print(f"Loading pretrained f_model from {f_ckpt_path}")
            payload = torch.load(f_ckpt_path, map_location='cpu')
            self.f_model.load_state_dict(payload['state_dicts']['model'])
            for param in self.f_model.parameters():
                param.requires_grad = False
        elif self.func_type == 'g':
            self.g_model = model
            self.d_model = copy.deepcopy(model)
            self.f_model = copy.deepcopy(model)
            
            if f_ckpt_path is None:
                raise ValueError("f_ckpt_path must be provided when training the diffusion term (g).")
            
            print(f"Loading pretrained f_model from {f_ckpt_path}")
            payload = torch.load(f_ckpt_path, map_location='cpu')
            self.f_model.load_state_dict(payload['state_dicts']['model'])

            if d_ckpt_path is not None:
                print(f"Loading pretrained d_model from {d_ckpt_path}")
                payload = torch.load(d_ckpt_path, map_location='cpu')
                self.d_model.load_state_dict(payload['state_dicts']['model'])
            
            # Freeze f_model
            for param in self.f_model.parameters():
                param.requires_grad = False
            for param in self.d_model.parameters():
                param.requires_grad = False
        else:
            raise ValueError(f"Invalid func_type: {self.func_type}")

        input_dim = action_dim
        if not (obs_as_local_cond or obs_as_global_cond):
            input_dim += obs_dim

        # create sde model
        self.sde_model = NeuralSDE(
            flow=self.f_model,
            diffusion=self.g_model,
            logit_range=logit_range,
            denoiser=self.d_model,
            denoising_magnitude=denoising_magnitude,
            diffusion_magnitude=diffusion_magnitude,
            state_shape=(n_obs_steps, input_dim),
            noise_std=noise_std,
            delta_t=self.delta_t,
            discrete_dims=self.discrete_dims,
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.num_inference_steps = num_inference_steps

    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        sde_model = self.sde_model
        sde_model.local_cond = local_cond
        sde_model.global_cond = global_cond

        trajectory = condition_data
        for i in range(self.horizon-self.n_obs_steps):
            sde_model.timestep=i*self.delta_t
            y0=trajectory[:,-self.n_obs_steps:].flatten(start_dim=1)
            states = torchsde.sdeint(
                sde=sde_model,
                y0=y0,
                ts=torch.linspace(i*self.delta_t,(i+1)*self.delta_t,self.num_inference_steps+1).to(y0.device),
                method="euler",
                dt=1e0*self.delta_t,
                adaptive=False,
                rtol=1e-10,
                atol=1e-2,
                dt_min=1e-10,
            )
            states=rearrange(states, "t b (o d) -> b t o d", o=sde_model.state_shape[0]).contiguous()
            if self.discrete_dims is not None:
                states[...,self.discrete_dims]=sde_model.discrete_dims_cache
            trajectory = torch.cat([trajectory,states[:,-1,-1:].clamp(-1,1)],dim=1)

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' in obs_dict # not implemented yet
        past_action = obs_dict['past_action']
        past_action = self.normalizer['action'].normalize(past_action)
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        cond_data = None
        cond_mask = None

        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, To, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To] = past_action[:,:To]
            cond_mask[:,:To] = True
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, To, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To] = past_action[:,:To]
            cond_mask[:,:To] = True
        else:
            # condition through impainting
            shape = (B, To, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To] = torch.cat([past_action[:,:To],nobs[:,:To]],dim=-1)
            cond_mask[:,:To] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To
        if self.oa_step_convention:
            start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, lr: float, betas: Tuple[float, float], eps: float ,
        ) -> torch.optim.Optimizer:
        
        model_to_train = None
        if self.func_type == 'f':
            assert self.f_model is not None
            model_to_train = self.f_model
        elif self.func_type == 'd':
            assert self.d_model is not None
            model_to_train = self.d_model
        elif self.func_type == 'g':
            assert self.g_model is not None
            model_to_train = self.g_model
        else:
            raise ValueError(f"Invalid func_type: {self.func_type}")

        optimizer = torch.optim.AdamW(
            params=model_to_train.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        
        batch_size = action.shape[0]
        To = self.n_obs_steps

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        cond_data = None
        ds_dt = None
        timesteps = torch.rand(
            (batch_size,), device=action.device
        )
        timesteps = timesteps * (self.horizon - To)
        #get the integer part of timesteps
        indices = timesteps.int()
        #get the fractional part of timesteps
        tau=timesteps-indices
        tau = tau.view(-1, 1, 1)
        #create indices for each observation step
        indices = indices.unsqueeze(1) + torch.arange(To+1, device=indices.device)
        indices = indices.long()

        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
            action_dim = action.shape[-1]
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, action_dim)
            states = torch.gather(action, 1, expanded_indices)
            cond_data = tau*states[:,:To] + (1.0-tau)*states[:,1:To+1]
            ds_dt = (states[:,1:To+1]-states[:,:To])/self.delta_t
            ds_dt[...,action_dim-1] = states[:,1:To+1,action_dim-1]

        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            action_dim = action.shape[-1]
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, action_dim)
            states = torch.gather(action, 1, expanded_indices)
            cond_data = tau*states[:,:To] + (1.0-tau)*states[:,1:To+1]
            ds_dt = (states[:,1:To+1]-states[:,:To])/self.delta_t
            ds_dt[...,action_dim-1] = states[:,1:To+1,action_dim-1]
        else:
            trajectory = torch.cat([action, obs], dim=-1).detach()
            traj_dim = trajectory.shape[-1]
            expanded_indices = indices.unsqueeze(-1).expand(-1, -1, traj_dim)
            states = torch.gather(trajectory, 1, expanded_indices)
            cond_data = tau*states[:,:To] + (1-tau)*states[:,1:To+1]
            ds_dt = (states[:,1:To+1]-states[:,:To])/self.delta_t
            ds_dt[...,action_dim-1] = states[:,1:To+1,action_dim-1]
        
        # generate impainting mask
        condition_mask = torch.zeros_like(ds_dt, dtype=torch.bool)
        if self.func_type == 'f' or self.func_type == 'g':
            condition_mask[:,:To-1] = True

        # Sample noise that we'll add to the images
        noise = torch.randn(cond_data.shape, device=cond_data.device)*self.noise_std
        cond_data = cond_data + noise

        # compute loss mask
        loss_mask = ~condition_mask
        loss = 0.0
        # Predict the noise residual
        if self.func_type == 'f':
            assert self.f_model is not None
            pred = self.f_model(cond_data, timesteps, local_cond=local_cond, global_cond=global_cond)
            target = ds_dt
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
        elif self.func_type == 'd':
            assert self.d_model is not None
            pred = self.d_model(cond_data, timesteps, local_cond=local_cond, global_cond=global_cond)
            target = -noise
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
        elif self.func_type == 'g':
            assert self.g_model is not None
            pred = self.g_model(cond_data, timesteps, local_cond=local_cond, global_cond=global_cond)
            with torch.no_grad():
                assert self.f_model is not None
                f_pred = self.f_model(cond_data,timesteps,local_cond=local_cond, global_cond=global_cond)
                target = (ds_dt-f_pred)**2*self.delta_t
            logit_min, logit_max = self.sde_model.logit_min, self.sde_model.logit_max
            scaled_pred = (torch.tanh(pred)+1.0)*0.5*(logit_max-logit_min)+logit_min
            loss = F.mse_loss(torch.exp(scaled_pred), torch.sqrt(target), reduction='none')+\
                F.mse_loss(scaled_pred, 0.5*torch.log(target+1.0e-6), reduction='none')
            loss = loss * loss_mask.type(loss.dtype)

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
