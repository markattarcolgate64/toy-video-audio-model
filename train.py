"""Train DiT directly on pixel-space bouncing ball videos."""

import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import SyntheticBouncingBallDataset
from model import DiT
from diffusion import GaussianDiffusion
from utils import save_video


def main():
    parser = argparse.ArgumentParser(description="Train pixel-space DiT on bouncing ball videos")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_videos", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="outputs/dit")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample_every", type=int, default=50)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Generate dataset
    print("Generating bouncing ball dataset...")
    dataset = SyntheticBouncingBallDataset(num_videos=args.num_videos)
    videos = dataset.data  # (N, 3, T, H, W) in [-1, 1]
    print(f"Dataset shape: {videos.shape}")
    print(f"Data stats: mean={videos.mean():.4f}, std={videos.std():.4f}, "
          f"min={videos.min():.4f}, max={videos.max():.4f}")

    # Create dataloader
    video_dataset = TensorDataset(videos)
    dataloader = DataLoader(video_dataset, batch_size=args.batch_size, shuffle=True)

    # Create DiT (pixel-space: 3 channels, 32x32, patch_size=4)
    model = DiT(
        in_channels=3,
        num_frames=16,
        image_size=32,
        patch_size=4,
        hidden_dim=384,
        num_heads=6,
        num_layers=6,
    ).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"DiT parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

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

            # Compute loss (predict noise)
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
                sample_videos = diffusion.sample(model, shape=(4, 3, 16, 32, 32))
            for i in range(min(4, sample_videos.shape[0])):
                path = f"{args.output_dir}/sample_epoch_{epoch+1:03d}_{i}.gif"
                save_video(sample_videos[i].cpu(), path)
            print(f"  Saved samples to {args.output_dir}/")

    # Save checkpoint
    torch.save(model.state_dict(), "checkpoints/dit.pt")
    print("Saved DiT checkpoint to checkpoints/dit.pt")


if __name__ == "__main__":
    main()
