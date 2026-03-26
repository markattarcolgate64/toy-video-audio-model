import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoVAE(nn.Module):
    """Simple convolutional VAE applied per-frame.

    Encodes 32x32 RGB frames to 8x8 latents with 4 channels (12x compression).
    Decoder reconstructs back to 32x32 RGB.
    Weights are shared across all frames — no temporal modeling in the VAE.
    """

    def __init__(self, in_channels=3, latent_channels=4):
        super().__init__()
        self.latent_channels = latent_channels

        # Encoder: (3, 32, 32) -> (8, 8, 8) -> split to mu, logvar each (4, 8, 8)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),   # 32->16
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 16->8
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels * 2, 3, stride=1, padding=1),  # 8->8, 8 channels
        )

        # Decoder: (4, 8, 8) -> (3, 32, 32)
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 16->32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels, 3, stride=1, padding=1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def encode_frame(self, x):
        """Encode a single frame. x: (B, 3, 32, 32) -> mu, logvar each (B, 4, 8, 8)."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Sample z = mu + sigma * epsilon."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode_frame(self, z):
        """Decode a single frame. z: (B, 4, 8, 8) -> (B, 3, 32, 32)."""
        return self.decoder(z)

    def encode(self, video):
        """Encode a video. video: (B, 3, T, H, W) -> latent (B, 4, T, 8, 8).

        Returns latent, mu, logvar.
        """
        B, C, T, H, W = video.shape
        # Reshape to process all frames at once: (B*T, C, H, W)
        frames = video.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        mu, logvar = self.encode_frame(frames)
        z = self.reparameterize(mu, logvar)
        # Reshape back: (B, T, latent_ch, h, w) -> (B, latent_ch, T, h, w)
        _, Cz, Hz, Wz = z.shape
        z = z.reshape(B, T, Cz, Hz, Wz).permute(0, 2, 1, 3, 4)
        mu = mu.reshape(B, T, Cz, Hz, Wz).permute(0, 2, 1, 3, 4)
        logvar = logvar.reshape(B, T, Cz, Hz, Wz).permute(0, 2, 1, 3, 4)
        return z, mu, logvar

    def decode(self, latent):
        """Decode a latent video. latent: (B, 4, T, 8, 8) -> (B, 3, T, 32, 32)."""
        B, Cz, T, Hz, Wz = latent.shape
        frames = latent.permute(0, 2, 1, 3, 4).reshape(B * T, Cz, Hz, Wz)
        decoded = self.decode_frame(frames)
        _, C, H, W = decoded.shape
        decoded = decoded.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        return decoded

    def forward(self, video):
        """Full forward pass. video: (B, 3, T, H, W) -> recon, mu, logvar."""
        z, mu, logvar = self.encode(video)
        recon = self.decode(z)
        return recon, mu, logvar

    @staticmethod
    def loss_function(recon, target, mu, logvar, kl_weight=0.001):
        """VAE loss = reconstruction MSE + KL divergence."""
        recon_loss = F.mse_loss(recon, target)
        # KL divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_weight * kl_loss, recon_loss, kl_loss
