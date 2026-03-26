import os
import torch
import numpy as np
import imageio


def tensor_to_frames(tensor):
    """Convert (3, T, H, W) float tensor in [-1, 1] to (T, H, W, 3) uint8 numpy array."""
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1) / 2 * 255  # [-1,1] -> [0,255]
    tensor = tensor.to(torch.uint8)
    # (3, T, H, W) -> (T, H, W, 3)
    frames = tensor.permute(1, 2, 3, 0).numpy()
    return frames


def save_video(tensor, path, fps=8):
    """Save a (3, T, H, W) float tensor as gif or mp4.

    Dispatches based on file extension.
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    frames = tensor_to_frames(tensor)
    frame_list = [frames[i] for i in range(frames.shape[0])]

    if path.endswith(".gif"):
        imageio.mimsave(path, frame_list, fps=fps, loop=0)
    elif path.endswith(".mp4"):
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
        for frame in frame_list:
            writer.append_data(frame)
        writer.close()
    else:
        raise ValueError(f"Unsupported format: {path}. Use .gif or .mp4")


def save_image_grid(tensors, path, nrow=4):
    """Save a list of (3, H, W) tensors as a grid image for VAE reconstruction comparison."""
    from torchvision.utils import make_grid
    from PIL import Image

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    grid = make_grid(tensors, nrow=nrow, normalize=True, value_range=(-1, 1))
    grid = grid.permute(1, 2, 0).mul(255).clamp(0, 255).to(torch.uint8).numpy()
    Image.fromarray(grid).save(path)
