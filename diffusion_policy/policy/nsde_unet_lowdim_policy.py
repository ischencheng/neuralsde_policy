from typing import Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import torchsde
import copy
from pydrake.all import PiecewisePolynomial
import numpy as np

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
            state_shape=(1,input_dim),
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
            condition_data,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        sde_model = self.sde_model
        sde_model.local_cond = local_cond
        sde_model.global_cond = global_cond

        trajectory = condition_data
        for i in range(self.n_obs_steps-1,self.horizon):
            sde_model.timestep=i*self.delta_t
            y0=trajectory[:,-1:].flatten(start_dim=1)
            states = torchsde.sdeint(
                sde=sde_model,
                y0=y0,
                ts=torch.linspace(i*self.delta_t,(i+1)*self.delta_t,self.num_inference_steps+1).to(y0.device),
                method="milstein",
                dt=1e0*self.delta_t,
                adaptive=True,
                rtol=1e-4,
                atol=1e-4,
                dt_min=1e-10,
            )
            states=rearrange(states, "t b d -> b t d").contiguous()
            trajectory = torch.cat([trajectory,states[:,-1:].clamp(-1,1)],dim=1)

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
            cond_data[:,:To] = past_action[:,:To]
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, To, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_data[:,:To] = past_action[:,:To]
        else:
            # condition through impainting
            shape = (B, To, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_data[:,:To] = torch.cat([past_action[:,:To],nobs[:,:To]],dim=-1)

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
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

        nobs = batch['nobs']
        naction = batch['naction']
        ns = batch['ns']
        dns_dt = batch['dns_dt']
        t = batch['t']
        noise = batch['noise']
        cns = batch['cns']
        dcns_dt = batch['dcns_dt']
        cnoise = batch['cnoise']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        s = None
        ds_dt = None

        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = nobs
            local_cond[:,self.n_obs_steps:,:] = 0
            s = ns
            ds_dt = dns_dt
        elif self.obs_as_global_cond:
            global_cond = nobs[:,:self.n_obs_steps,:].reshape(
                nobs.shape[0], -1)        
            s = ns
            ds_dt = dns_dt
        else:
            s = cns
            ds_dt = dcns_dt
            noise = cnoise

        loss = 0.0
        # Predict the noise residual
        if self.func_type == 'f':
            assert self.f_model is not None
            pred = self.f_model(s, t/self.delta_t, local_cond=local_cond, global_cond=global_cond)
            target = ds_dt
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.func_type == 'd':
            assert self.d_model is not None
            pred = self.d_model(s, t/self.delta_t, local_cond=local_cond, global_cond=global_cond)
            target = -noise
            loss = F.mse_loss(pred, target, reduction='none')
        elif self.func_type == 'g':
            assert self.g_model is not None
            pred = self.g_model(s, t/self.delta_t, local_cond=local_cond, global_cond=global_cond)
            with torch.no_grad():
                assert self.f_model is not None
                f_pred = self.f_model(s,t/self.delta_t,local_cond=local_cond, global_cond=global_cond)
                target = (ds_dt-f_pred)**2*self.delta_t
            logit_min, logit_max = self.sde_model.logit_min, self.sde_model.logit_max
            scaled_pred = (torch.tanh(pred)+1.0)*0.5*(logit_max-logit_min)+logit_min
            loss = F.mse_loss(torch.exp(scaled_pred), torch.sqrt(target), reduction='none')+\
                F.mse_loss(scaled_pred, 0.5*torch.log(target+1.0e-6), reduction='none')

        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def get_collate_fn(self, normalizer: LinearNormalizer, horizon: int, n_obs_steps: int, noise_std: float, delta_t: float):
        """
        Returns a collate function that applies normalization and the transformation.
        """
        def collate_fn(batch):
            # batch is a list of dicts
            processed_batch = []
            for sample in batch:
                # normalize sample
                n_sample = normalizer.normalize(sample)
                naction = n_sample['action']
                nobs = n_sample['obs']

                # PiecewisePolynomial and np.concatenate require numpy arrays
                if torch.is_tensor(naction):
                    naction = naction.detach().cpu().numpy()
                if torch.is_tensor(nobs):
                    nobs = nobs.detach().cpu().numpy()

                # Create a trajectory from the action sequence.
                traj_times = np.linspace(0, horizon-1, horizon)*delta_t
                traj_positions = naction
                traj = PiecewisePolynomial.FirstOrderHold(
                    traj_times, traj_positions.T
                )

                # Sample time and get state/velocity time should be in [n_obs_steps-1, horizon-1]

                time = np.float32((np.random.rand()*(horizon-n_obs_steps) + n_obs_steps-1)*delta_t)
                ns = traj.value(time).T
                noise = np.random.randn(*ns.shape)*noise_std
                ns = ns+noise
                dns_dt = traj.EvalDerivative(time).T
                
                concatenated_traj_positions = np.concatenate([naction, nobs], axis=-1)
                concatenated_traj = PiecewisePolynomial.FirstOrderHold(
                    traj_times, concatenated_traj_positions.T
                )
                cns = concatenated_traj.value(time).T
                cnoise = np.random.randn(*cns.shape)*noise_std
                cns = cns+cnoise
                dcns_dt = concatenated_traj.EvalDerivative(time).T

                processed_batch.append({
                    'obs': sample['obs'],
                    'action': sample['action'], # keep original action
                    'nobs': nobs,
                    'naction': naction,
                    'ns': ns.astype(np.float32),
                    'dns_dt': dns_dt.astype(np.float32),
                    'cns': cns.astype(np.float32),
                    'dcns_dt': dcns_dt.astype(np.float32),
                    't': time,
                    'noise': noise.astype(np.float32),
                    'cnoise': cnoise.astype(np.float32),
                })
            
            # use default_collate to batch the processed samples
            return torch.utils.data.default_collate(processed_batch)
        
        return collate_fn
