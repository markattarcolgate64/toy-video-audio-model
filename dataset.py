import torch
from torch.utils.data import Dataset


class SyntheticBouncingBallDataset(Dataset):
    """Procedurally generated bouncing ball videos.

    Each video has `num_balls` colored circles bouncing inside a frame
    with constant velocity and perfect elastic wall reflections.
    All videos are pre-generated and stored in memory.
    """

    # Bright colors on black background (RGB, in [0, 1])
    COLORS = [
        (1.0, 0.0, 0.0),   # red
        (0.0, 1.0, 0.0),   # green
        (0.0, 0.4, 1.0),   # blue
        (1.0, 1.0, 0.0),   # yellow
        (0.0, 1.0, 1.0),   # cyan
        (1.0, 0.0, 1.0),   # magenta
    ]

    def __init__(self, num_videos=256, num_frames=16, image_size=32,
                 num_balls=2, ball_radius=3):
        super().__init__()
        self.num_videos = num_videos
        self.num_frames = num_frames
        self.image_size = image_size
        self.num_balls = num_balls
        self.ball_radius = ball_radius

        # Pre-generate all videos: (N, 3, T, H, W) in [-1, 1]
        self.data = self._generate_all()

    def _generate_all(self):
        all_videos = []
        for _ in range(self.num_videos):
            video = self._generate_one_video()
            all_videos.append(video)
        return torch.stack(all_videos)

    def _generate_one_video(self):
        s = self.image_size
        r = self.ball_radius

        # Initialize ball states
        positions = []
        velocities = []
        colors = []
        for i in range(self.num_balls):
            # Random position, keeping ball fully inside frame
            x = torch.empty(1).uniform_(r + 1, s - r - 1).item()
            y = torch.empty(1).uniform_(r + 1, s - r - 1).item()
            positions.append([x, y])

            # Random velocity: 1-3 pixels per frame, random direction
            speed = torch.empty(1).uniform_(1.0, 3.0).item()
            angle = torch.empty(1).uniform_(0, 2 * 3.14159).item()
            vx = speed * torch.tensor(angle).cos().item()
            vy = speed * torch.tensor(angle).sin().item()
            velocities.append([vx, vy])

            # Pick a color (cycle through palette)
            colors.append(self.COLORS[i % len(self.COLORS)])

        # Simulate and render each frame
        frames = []
        for _ in range(self.num_frames):
            frame = self._render_frame(positions, colors)
            frames.append(frame)
            # Update positions with velocity
            for j in range(self.num_balls):
                positions[j][0] += velocities[j][0]
                positions[j][1] += velocities[j][1]
                # Bounce off walls
                if positions[j][0] - r < 0:
                    positions[j][0] = r + (r - positions[j][0])
                    velocities[j][0] *= -1
                elif positions[j][0] + r > s:
                    positions[j][0] = (s - r) - (positions[j][0] + r - s)
                    velocities[j][0] *= -1
                if positions[j][1] - r < 0:
                    positions[j][1] = r + (r - positions[j][1])
                    velocities[j][1] *= -1
                elif positions[j][1] + r > s:
                    positions[j][1] = (s - r) - (positions[j][1] + r - s)
                    velocities[j][1] *= -1

        # Stack frames: (T, 3, H, W) -> (3, T, H, W)
        video = torch.stack(frames).permute(1, 0, 2, 3)
        # Convert from [0, 1] to [-1, 1]
        video = video * 2 - 1
        return video

    def _render_frame(self, positions, colors):
        """Render circles using distance fields. Returns (3, H, W) in [0, 1]."""
        s = self.image_size
        r = self.ball_radius
        frame = torch.zeros(3, s, s)

        # Coordinate grids
        yy = torch.arange(s).float().unsqueeze(1).expand(s, s)  # (H, W)
        xx = torch.arange(s).float().unsqueeze(0).expand(s, s)  # (H, W)

        for pos, color in zip(positions, colors):
            dist = torch.sqrt((xx - pos[0]) ** 2 + (yy - pos[1]) ** 2)
            mask = (dist <= r).float()
            for c in range(3):
                frame[c] += mask * color[c]

        return frame.clamp(0, 1)

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        return self.data[idx]
