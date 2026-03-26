"""Phase 1: Train the VAE on bouncing ball frames."""

import os
import argparse
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import SyntheticBouncingBallDataset
from vae import VideoVAE
from utils import save_image_grid


def main():
    parser = argparse.ArgumentParser(description="Train VAE on bouncing ball frames")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--num_videos", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="outputs/vae")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Generate bouncing ball dataset
    print("Generating bouncing ball dataset...")
    dataset = SyntheticBouncingBallDataset(num_videos=args.num_videos)
    # Flatten to individual frames: (N*T, 3, 32, 32)
    videos = dataset.data  # (N, 3, T, H, W)
    B, C, T, H, W = videos.shape
    frames = videos.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    print(f"Total frames: {frames.shape[0]}")

    frame_dataset = TensorDataset(frames)
    dataloader = DataLoader(frame_dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    model = VideoVAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {param_count:,}")

    # Keep a fixed batch for reconstruction visualization
    vis_frames = frames[:16].to(device)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0
        n_batches = 0

        for (batch,) in dataloader:
            batch = batch.to(device)
            # Forward: treat each frame as an independent image
            # Unsqueeze to add fake time dim: (B, 3, 1, H, W)
            batch_5d = batch.unsqueeze(2)
            recon, mu, logvar = model(batch_5d)
            loss, recon_loss, kl_loss = VideoVAE.loss_function(
                recon, batch_5d, mu, logvar, kl_weight=args.kl_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        avg_recon = total_recon / n_batches
        avg_kl = total_kl / n_batches
        print(f"Epoch {epoch+1:3d}/{args.epochs} | Loss: {avg_loss:.5f} | "
              f"Recon: {avg_recon:.5f} | KL: {avg_kl:.4f}")

        # Save reconstruction comparison
        if (epoch + 1) % 20 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                vis_5d = vis_frames.unsqueeze(2)
                recon, _, _ = model(vis_5d)
                recon_frames = recon.squeeze(2)
            # Interleave original and reconstruction
            pairs = []
            for i in range(min(8, len(vis_frames))):
                pairs.append(vis_frames[i])
                pairs.append(recon_frames[i])
            save_image_grid(pairs, f"{args.output_dir}/recon_epoch_{epoch+1:03d}.png", nrow=4)
            print(f"  Saved reconstruction comparison to {args.output_dir}/recon_epoch_{epoch+1:03d}.png")

    # Save checkpoint
    torch.save(model.state_dict(), "checkpoints/vae.pt")
    print("Saved VAE checkpoint to checkpoints/vae.pt")


if __name__ == "__main__":
    main()
