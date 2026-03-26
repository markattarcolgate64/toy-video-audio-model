"""Phase 2: Train the DiT in latent space (VAE frozen)."""

import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import SyntheticBouncingBallDataset
from vae import VideoVAE
from model import DiT
from diffusion import GaussianDiffusion
from utils import save_video


def main():
    parser = argparse.ArgumentParser(description="Train DiT on latent bouncing ball videos")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_videos", type=int, default=256)
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae.pt")
    parser.add_argument("--output_dir", type=str, default="outputs/dit")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample_every", type=int, default=25)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Load VAE
    print("Loading VAE...")
    vae = VideoVAE().to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # Generate dataset and encode to latents
    print("Generating bouncing ball dataset and encoding to latents...")
    dataset = SyntheticBouncingBallDataset(num_videos=args.num_videos)
    videos = dataset.data.to(device)  # (N, 3, T, H, W)

    # Encode all videos to latent space (in batches to save memory)
    latents_list = []
    encode_batch_size = 32
    with torch.no_grad():
        for i in range(0, len(videos), encode_batch_size):
            batch = videos[i:i + encode_batch_size]
            z, mu, _ = vae.encode(batch)
            # Use mu (deterministic) for training data, not sampled z
            latents_list.append(mu.cpu())
    latents = torch.cat(latents_list, dim=0).to(device)  # (N, 4, T, 8, 8)
    print(f"Latent shape: {latents.shape}")

    # Create dataloader
    latent_dataset = TensorDataset(latents)
    dataloader = DataLoader(latent_dataset, batch_size=args.batch_size, shuffle=True)

    # Create DiT
    model = DiT(
        latent_channels=4,
        num_frames=16,
        latent_size=8,
        patch_size=2,
        hidden_dim=384,
        num_heads=6,
        num_layers=6,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"DiT parameters: {param_count:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Diffusion process
    diffusion = GaussianDiffusion(num_timesteps=1000, device=device)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for (batch,) in dataloader:
            batch = batch.to(device)
            B = batch.shape[0]

            # Random timesteps
            t = torch.randint(0, diffusion.num_timesteps, (B,), device=device)

            # Compute loss
            loss = diffusion.p_losses(model, batch, t)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.5f}")

        # Generate samples
        if (epoch + 1) % args.sample_every == 0:
            print("  Generating sample...")
            model.eval()
            with torch.no_grad():
                # Sample latents from diffusion
                sample_latents = diffusion.sample(model, shape=(4, 4, 16, 8, 8))
                # Decode with VAE
                sample_videos = vae.decode(sample_latents)

            for i in range(min(4, sample_videos.shape[0])):
                path = f"{args.output_dir}/sample_epoch_{epoch+1:03d}_{i}.gif"
                save_video(sample_videos[i].cpu(), path)
            print(f"  Saved samples to {args.output_dir}/")

    # Save checkpoint
    torch.save(model.state_dict(), "checkpoints/dit.pt")
    print("Saved DiT checkpoint to checkpoints/dit.pt")


if __name__ == "__main__":
    main()
