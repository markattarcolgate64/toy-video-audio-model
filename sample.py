"""Generate videos using trained VAE + DiT."""

import os
import argparse
import torch

from vae import VideoVAE
from model import DiT
from diffusion import GaussianDiffusion
from utils import save_video


def main():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument("--vae_checkpoint", type=str, default="checkpoints/vae.pt")
    parser.add_argument("--dit_checkpoint", type=str, default="checkpoints/dit.pt")
    parser.add_argument("--latent_stats", type=str, default="checkpoints/latent_stats.pt")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="generated")
    parser.add_argument("--format", type=str, default="gif", choices=["gif", "mp4"])
    parser.add_argument("--show_process", action="store_true",
                        help="Save denoising progression visualization")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    print("Loading VAE...")
    vae = VideoVAE().to(device)
    vae.load_state_dict(torch.load(args.vae_checkpoint, map_location=device, weights_only=True))
    vae.eval()

    print("Loading DiT...")
    dit = DiT(
        latent_channels=4,
        num_frames=16,
        latent_size=8,
        patch_size=2,
        hidden_dim=384,
        num_heads=6,
        num_layers=6,
    ).to(device)
    dit.load_state_dict(torch.load(args.dit_checkpoint, map_location=device, weights_only=True))
    dit.eval()

    diffusion = GaussianDiffusion(num_timesteps=1000, device=device)
    latent_shape = (1, 4, 16, 8, 8)

    # Load latent normalization stats
    print("Loading latent normalization stats...")
    stats = torch.load(args.latent_stats, map_location=device, weights_only=True)
    latent_mean = stats["mean"].to(device)
    latent_std = stats["std"].to(device)
    print(f"Latent stats: mean={latent_mean:.6f}, std={latent_std:.6f}")

    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        for i in range(args.num_samples):
            if args.show_process:
                # Save denoising progression
                z, intermediates = diffusion.sample_with_intermediates(
                    dit, shape=latent_shape, save_every=100
                )
                # Decode and save each intermediate (denormalize first)
                all_frames = []
                for z_inter in intermediates:
                    z_denorm = z_inter * latent_std + latent_mean
                    video = vae.decode(z_denorm)
                    all_frames.append(video[0].cpu())
                # Concatenate all intermediates into one long video
                combined = torch.cat(all_frames, dim=1)  # concat along time
                ext = f".{args.format}"
                save_video(combined, f"{args.output_dir}/process_{i}{ext}", fps=4)
            else:
                z = diffusion.sample(dit, shape=latent_shape)

            # Denormalize back to original latent scale
            z = z * latent_std + latent_mean
            # Decode final result
            video = vae.decode(z)
            ext = f".{args.format}"
            path = f"{args.output_dir}/sample_{i}{ext}"
            save_video(video[0].cpu(), path)
            print(f"  Saved {path}")

    print("Done!")


if __name__ == "__main__":
    main()
