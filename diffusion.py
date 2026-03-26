"""DDPM Gaussian Diffusion process.

Handles the noise schedule, forward process (adding noise), training loss,
and reverse process (iterative denoising / sampling).
All operations are in latent space.
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm


class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        # Linear beta schedule
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # Store everything we need for forward and reverse process
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)

        # For forward process q(x_t | x_0):
        #   x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod).to(device)

        # For reverse process p(x_{t-1} | x_t):
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(device)
        self.posterior_mean_coef1 = (torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - alphas_cumprod)).to(device)
        self.posterior_mean_coef2 = (torch.sqrt(alphas) * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)).to(device)

    def _extract(self, tensor, t, shape):
        """Extract values from tensor at timestep indices t, broadcast to shape."""
        out = tensor.gather(0, t)
        return out.reshape(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """Forward process: add noise to x0 at timestep t.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_ab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_ab * x0 + sqrt_one_minus_ab * noise

    def p_losses(self, model, x0, t):
        """Compute training loss: MSE between predicted and actual noise."""
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise=noise)
        noise_pred = model(x_t, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, model, x_t, t_index):
        """One step of reverse process: x_t -> x_{t-1}."""
        B = x_t.shape[0]
        t = torch.full((B,), t_index, device=self.device, dtype=torch.long)

        # Model predicts the noise
        noise_pred = model(x_t, t)

        # Compute predicted x_0 from x_t and predicted noise
        sqrt_ab = self._extract(self.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_ab = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        x0_pred = (x_t - sqrt_one_minus_ab * noise_pred) / sqrt_ab

        # Compute posterior mean
        coef1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        coef2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = coef1 * x0_pred + coef2 * x_t

        if t_index > 0:
            variance = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean

    @torch.no_grad()
    def sample(self, model, shape):
        """Full reverse process: pure noise -> clean sample.

        shape: e.g. (batch_size, 4, 16, 8, 8) for latent video
        """
        model.eval()
        x = torch.randn(shape, device=self.device)

        for t in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps,
                       desc="Sampling"):
            x = self.p_sample(model, x, t)

        return x

    @torch.no_grad()
    def sample_with_intermediates(self, model, shape, save_every=100):
        """Sample and return intermediate steps for visualization."""
        model.eval()
        x = torch.randn(shape, device=self.device)
        intermediates = [x.clone()]

        for t in tqdm(reversed(range(self.num_timesteps)), total=self.num_timesteps,
                       desc="Sampling"):
            x = self.p_sample(model, x, t)
            if t % save_every == 0:
                intermediates.append(x.clone())

        return x, intermediates
