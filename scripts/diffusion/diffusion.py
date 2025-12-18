import torch
import torch.nn as nn
from tqdm import tqdm

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = device

        # Noise Schedule (Linear)
        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        """
        x_t = sqrt(alpha_hat) * x_0 + sqrt(1 - alpha_hat) * epsilon
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None]
        epsilon = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=self.device)

    def sample(self, model, condition, n_samples):
        """
        Reverse Process (Inference)
        condition: (Batch, 2, Seq_Len) -> Noisy Input Signal
        """
        model.eval()
        with torch.no_grad():
            # Start from pure noise
            x = torch.randn_like(condition).to(self.device)
            
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, desc="Sampling"):
                t = (torch.ones(n_samples) * i).long().to(self.device)
                
                # Model Input: Concatenate current latent x and condition
                # x shape: (B, 2, L), condition shape: (B, 2, L) -> (B, 4, L)
                model_input = torch.cat([x, condition], dim=1)
                
                predicted_noise = model(model_input, t)
                
                alpha = self.alpha[t][:, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None]
                beta = self.beta[t][:, None, None]
                
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                
                # x_{t-1} calculation
                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
        model.train()
        return x