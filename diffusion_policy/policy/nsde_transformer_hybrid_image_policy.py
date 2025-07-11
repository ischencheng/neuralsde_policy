from typing import Dict, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torchsde
from diffusion_policy.model.diffusion.sde import NeuralSDE
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules
from diffusion_policy.common.obs2action import pack_pose_as_action

class NSDETransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=False,
            obs_as_cond=False,
            # sde model
            func_type :str='f',
            f_ckpt_path: Optional[str] = None,
            d_ckpt_path: Optional[str] = None,
            denoising_magnitude=1.0,
            diffusion_magnitide=1.0,
            noise_std=0.02,
            logit_range=(-10,-1),
            delta_t=1.0,
            # parameters passed to step
            **kwargs):
        super().__init__()


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.long

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model_kwargs = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "horizon": horizon,
            "n_obs_steps": n_obs_steps,
            "cond_dim": cond_dim,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_emb": n_emb,
            "p_drop_emb": p_drop_emb,
            "p_drop_attn": p_drop_attn,
            "causal_attn": causal_attn,
            "time_as_cond": time_as_cond,
            "obs_as_cond": obs_as_cond,
            "n_cond_layers": n_cond_layers,
        }

        self.func_type = func_type
        self.f_model = None
        self.d_model = None
        self.g_model = None

        if self.func_type == 'f':
            self.f_model = TransformerForDiffusion(**model_kwargs)
        elif self.func_type == 'd':
            self.d_model = TransformerForDiffusion(**model_kwargs)
        elif self.func_type == 'g':
            self.g_model = TransformerForDiffusion(**model_kwargs)
            self.d_model = TransformerForDiffusion(**model_kwargs)
            self.f_model = TransformerForDiffusion(**model_kwargs)
            
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

        self.obs_encoder = obs_encoder
        # self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=True
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.delta_t = delta_t
        self.kwargs = kwargs
        # create sde model
        self.sde_model = NeuralSDE(
            flow=self.f_model,
            diffusion=self.g_model,
            logit_range=logit_range,
            denoiser=self.d_model,
            denoising_magnitude=torch.tensor(denoising_magnitude,dtype=self.dtype,device=self.device),
            diffusion_magnitide=torch.tensor(diffusion_magnitide,dtype=self.dtype,device=self.device),
            state_shape=(1,horizon-1,input_dim),
            noise_std=torch.tensor(noise_std,dtype=self.dtype,device=self.device),
        )

        # if num_inference_steps is None:
        #     num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        sde_model = self.sde_model
        sde_model.cond=cond
        trajectory = condition_data
        for i in range(self.horizon-self.n_obs_steps):
            y0=trajectory[:,-2:]
            states = torchsde.sdeint(
                sde=sde_model,
                y0=y0,
                ts=torch.linspace(i,(i+1)*self.delta_t,2).to(y0.device),
                method="milstein",  # 'srk', 'euler', 'milstein', etc.
                dt=1e0,
                adaptive=False,
                rtol=1e-10,
                atol=1e-2,
                dt_min=1e-10,
            )
            trajectory = torch.cat([trajectory,states[:,-1]],dim=1)
        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' in obs_dict # not implemented yet
        past_action = obs_dict['past_action']
        #remove past_action from obs_dict
        obs_dict.pop('past_action')
        #ensure the key order is past_action, agent_pos, image
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, To, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To] = past_action[:,:To]
            cond_mask[:,:To] = True
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, To, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To] = torch.cat([past_action[:,:To],nobs_features[:,:To]],dim=-1)
            cond_mask[:,:To] = True


        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            To=To,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)


        start = To-1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        
        optim_groups = list()
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })

        if self.func_type == 'f':
            assert self.f_model is not None
            optim_groups.extend(self.f_model.get_optim_groups(weight_decay=transformer_weight_decay))
        elif self.func_type == 'd':
            assert self.d_model is not None
            optim_groups.extend(self.d_model.get_optim_groups(weight_decay=transformer_weight_decay))
        elif self.func_type == 'g':
            assert self.g_model is not None
            optim_groups.extend(self.g_model.get_optim_groups(weight_decay=transformer_weight_decay))

        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        cond_data = None
        ds_dt = None
        timesteps = torch.rand(
            0, 1, 
            (batch_size,), device=nactions.device
        ).long()
        timesteps = timesteps * (self.horizon - To)
        #get the integer part of timesteps
        indices = timesteps.int()
        #create indices for each observation step
        indices = indices.unsqueeze(1) + torch.arange(To+1, device=indices.device)
        #get the fractional part of timesteps
        tau=timesteps-indices

        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            #get actions for these indices
            states = torch.gather(nactions, 1, indices)
            cond_data = tau*states[:,:To] + (1-tau)*states[:,1:To+1]
            ds_dt = (states[:,1:To+1]-states[:,:To])/self.delta_t
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()
            states = torch.gather(trajectory, 1, indices)
            cond_data = tau*states[:,:To] + (1-tau)*states[:,1:To+1]
            ds_dt = (states[:,1:To+1]-states[:,:To])/self.delta_t

        # generate impainting mask
        condition_mask = torch.zeros_like(ds_dt, dtype=torch.bool)
        condition_mask[:,:To] = True

        # Sample noise that we'll add to the images
        noise = torch.randn(cond_data.shape, device=cond_data.device)
        cond_data = cond_data + noise*self.noise_std

        # compute loss mask
        loss_mask = ~condition_mask
        loss = 0.0
        # Predict the noise residual
        if self.func_type == 'f':
            pred = self.f_model(cond_data, timesteps, cond)
            target = ds_dt
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
        elif self.func_type == 'd':
            pred = self.d_model(cond_data, timesteps, cond)
            target = -noise
            loss = F.mse_loss(pred, target, reduction='none')
            loss = loss * loss_mask.type(loss.dtype)
        elif self.func_type == 'g':
            pred = self.g_model(cond_data, timesteps, cond)
            with torch.no_grad():
                f_pred = self.f_model(cond_data,timesteps,cond)
                target = (ds_dt-f_pred)**2*self.delta_t
            logit_min=self.sde_model.logit_range[0]
            logit_max=self.sde_model.logit_range[1]
            scaled_pred = (torch.tanh(pred)+1.0)*0.5*(logit_max-logit_min)+logit_min
            loss = F.mse_loss(torch.exp(scaled_pred), torch.sqrt(target), reduction='none')+\
                F.mse_loss(scaled_pred, 0.5*torch.log(target+1.0e-6), reduction='none')
            loss = loss * loss_mask.type(loss.dtype)


        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
