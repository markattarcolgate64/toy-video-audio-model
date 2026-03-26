"""Generate videos using trained pixel-space DiT."""

import os
import argparse
import torch

from model import DiT
from diffusion import GaussianDiffusion
from utils import save_video


def main():
    parser = argparse.ArgumentParser(description="Generate videos with trained model")
    parser.add_argument("--dit_checkpoint", type=str, default="checkpoints/dit.pt")
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

    # Load DiT
    print("Loading DiT...")
    dit = DiT(
        in_channels=3,
        num_frames=16,
        image_size=32,
        patch_size=4,
        hidden_dim=384,
        num_heads=6,
        num_layers=6,
    ).to(device)
    dit.load_state_dict(torch.load(args.dit_checkpoint, map_location=device, weights_only=True))
    dit.eval()

    diffusion = GaussianDiffusion(num_timesteps=1000, device=device)
    video_shape = (1, 3, 16, 32, 32)

    print(f"Generating {args.num_samples} samples...")
    with torch.no_grad():
        for i in range(args.num_samples):
            if args.show_process:
                z, intermediates = diffusion.sample_with_intermediates(
                    dit, shape=video_shape, save_every=100
                )
                all_frames = []
                for z_inter in intermediates:
                    all_frames.append(z_inter[0].cpu())
                combined = torch.cat(all_frames, dim=1)  # concat along time
                ext = f".{args.format}"
                save_video(combined, f"{args.output_dir}/process_{i}{ext}", fps=4)
            else:
                z = diffusion.sample(dit, shape=video_shape)

            ext = f".{args.format}"
            path = f"{args.output_dir}/sample_{i}{ext}"
            save_video(z[0].cpu(), path)
            print(f"  Saved {path}")

    print("Done!")


if __name__ == "__main__":
    main()
