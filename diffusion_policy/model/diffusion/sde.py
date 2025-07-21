import torch
import torchsde
from einops import rearrange
class NeuralSDE(torchsde.SDEIto):  # Assuming Ito SDEs are supported by default

    def __init__(
        self,
        flow=None,
        diffusion=None,
        logit_range=None,
        denoiser=None,
        denoising_magnitude=1.0,
        diffusion_magnitude=1.0,
        state_shape=None,
        noise_std=1.0,
        local_cond=None,
        global_cond=None,
        delta_t=0.1,
    ):
        super().__init__(noise_type="diagonal")
        self.flow = flow  # The drift model
        self.diffusion = diffusion  # The learned diffusion model
        self.logit_min=logit_range[0] if logit_range is not None else None
        self.logit_max=logit_range[1] if logit_range is not None else None
        self.denoiser = denoiser
        self.denoising_magnitude = denoising_magnitude  # Scaling factor for denoising
        self.diffusion_magnitude = diffusion_magnitude  # Scaling factor for diffusion
        self.noise_std = noise_std
        self.state_shape = state_shape #b,nc,h,w
        self.local_cond = local_cond
        self.global_cond = global_cond
        self.count=0
        self.delta_t = delta_t
        self.prev_t = 0.0
        self.prev_dt = -1.0

    # Drift term: f + c * denoising
    @torch.no_grad()
    def f(self, t: torch.Tensor, X: torch.Tensor):
        # if t<1e-10:
        #     self.prev_t=0.0
        #     self.prev_t = 0.0
        #     self.prev_dt = -1.0
        # if t>self.prev_t+1.0e-10:
        #     self.count+=1
        #     self.prev_dt = t - self.prev_t
        #     self.prev_t = t
        X=X.reshape(-1,*self.state_shape)
        flow=self.flow(X,t/self.delta_t,self.local_cond,self.global_cond)
        if self.denoiser:
            denoise=self.denoiser(X,t/self.delta_t,self.local_cond,self.global_cond)
            denoise_correction = denoise *self.denoising_magnitude
            flow=flow+denoise_correction
        return flow.flatten(start_dim=1)

    # Diffusion term: g(X)
    @torch.no_grad()
    def g(self, t: torch.Tensor, X: torch.Tensor):
        if self.diffusion is None:
            return torch.zeros_like(X)
        X=X.reshape(-1,*self.state_shape)
        diffusion = (torch.tanh(self.diffusion(X, t/self.delta_t, self.local_cond, self.global_cond)) + 1.0) * 0.5 \
            * (self.logit_max - self.logit_min) + self.logit_min
        diffusion = torch.exp(diffusion)
        diffusion=diffusion*self.diffusion_magnitude
        return diffusion.flatten(start_dim=1)

    
