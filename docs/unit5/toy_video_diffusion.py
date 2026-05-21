"""Minimal runnable toy video diffusion example for Unit 5.

This script trains a tiny diffusion model on synthetic moving-square videos.
It is intentionally small: the goal is to teach video tensor layout, diffusion
training, framewise denoising, and a simple temporal-convolution extension.
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MovingSquareDataset(Dataset):
    """Generate short videos of a colored square moving and bouncing.

    Each item returns a tensor with shape [T, C, H, W] in the range [-1, 1].
    The dataset is generated on the fly, so it does not need video files.
    """

    def __init__(
        self,
        length: int = 4096,
        num_frames: int = 16,
        image_size: int = 64,
        square_size: int = 10,
        min_speed: int = 1,
        max_speed: int = 3,
    ) -> None:
        self.length = length
        self.num_frames = num_frames
        self.image_size = image_size
        self.square_size = square_size
        self.min_speed = min_speed
        self.max_speed = max_speed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        generator = torch.Generator().manual_seed(index)
        limit = self.image_size - self.square_size

        x = torch.randint(0, limit + 1, (), generator=generator).item()
        y = torch.randint(0, limit + 1, (), generator=generator).item()

        vx = torch.randint(self.min_speed, self.max_speed + 1, (), generator=generator).item()
        vy = torch.randint(self.min_speed, self.max_speed + 1, (), generator=generator).item()
        if torch.rand((), generator=generator).item() < 0.5:
            vx = -vx
        if torch.rand((), generator=generator).item() < 0.5:
            vy = -vy

        color = torch.rand(3, 1, 1, generator=generator) * 0.8 + 0.2
        video = torch.zeros(self.num_frames, 3, self.image_size, self.image_size)

        for frame_idx in range(self.num_frames):
            frame = torch.zeros(3, self.image_size, self.image_size)
            frame[:, y : y + self.square_size, x : x + self.square_size] = color
            video[frame_idx] = frame

            x += vx
            y += vy

            if x < 0:
                x = -x
                vx = -vx
            elif x > limit:
                x = 2 * limit - x
                vx = -vx

            if y < 0:
                y = -y
                vy = -vy
            elif y > limit:
                y = 2 * limit - y
                vy = -vy

        video = video * 2.0 - 1.0
        return {"video": video}


def make_image_unet(image_size: int) -> UNet2DModel:
    return UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(32, 64, 64),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


class FramewiseVideoUNet(nn.Module):
    """Apply a 2D diffusion UNet independently to each frame."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.image_unet = make_image_unet(image_size)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T, H, W]
        batch, channels, frames, height, width = x.shape
        x_frames = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        frame_timesteps = timesteps[:, None].repeat(1, frames).reshape(batch * frames)
        pred = self.image_unet(x_frames, frame_timesteps).sample
        return pred.reshape(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4)


class TemporalConvVideoUNet(nn.Module):
    """Framewise 2D UNet followed by a small temporal residual correction."""

    def __init__(self, image_size: int) -> None:
        super().__init__()
        self.framewise = FramewiseVideoUNet(image_size)
        self.temporal = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv3d(32, 3, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        frame_pred = self.framewise(x, timesteps)
        temporal_residual = self.temporal(frame_pred)
        return frame_pred + temporal_residual


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = (frame.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    frame = (frame * 255).byte().permute(1, 2, 0).numpy()
    return Image.fromarray(frame)


def save_gif(video: torch.Tensor, path: Path, fps: int = 8) -> None:
    """Save a single video tensor [T, C, H, W] as GIF."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [tensor_to_pil(frame) for frame in video]
    duration_ms = int(1000 / fps)
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def save_frame_grid(video: torch.Tensor, path: Path) -> None:
    """Save all frames of one video as a horizontal contact sheet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [tensor_to_pil(frame) for frame in video]
    width, height = frames[0].size
    grid = Image.new("RGB", (width * len(frames), height))
    for idx, frame in enumerate(frames):
        grid.paste(frame, (idx * width, 0))
    grid.save(path)


def estimate_motion_score(video: torch.Tensor) -> float:
    """Return a simple adjacent-frame difference score for inspection."""

    # video: [T, C, H, W], range [-1, 1]
    return (video[1:] - video[:-1]).abs().mean().item()


@torch.no_grad()
def sample_video(
    model: nn.Module,
    scheduler: DDPMScheduler,
    shape: tuple[int, int, int, int, int],
    device: torch.device,
    num_inference_steps: int,
) -> torch.Tensor:
    model.eval()
    video = torch.randn(shape, device=device)
    scheduler.set_timesteps(num_inference_steps, device=device)

    for timestep in scheduler.timesteps:
        batch_timesteps = torch.full((shape[0],), int(timestep), device=device, dtype=torch.long)
        noise_pred = model(video, batch_timesteps)
        video = scheduler.step(noise_pred, timestep, video).prev_sample

    return video.clamp(-1, 1)


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MovingSquareDataset(
        length=args.dataset_size,
        num_frames=args.frames,
        image_size=args.image_size,
        square_size=args.square_size,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    preview = dataset[0]["video"]
    save_gif(preview, output_dir / "dataset_preview.gif", fps=args.fps)
    save_frame_grid(preview, output_dir / "dataset_preview_grid.png")

    if args.model == "framewise":
        model: nn.Module = FramewiseVideoUNet(args.image_size)
    elif args.model == "temporal":
        model = TemporalConvVideoUNet(args.image_size)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    model.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=args.train_timesteps, beta_schedule="squaredcos_cap_v2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    model.train()
    step = 0
    progress = tqdm(total=args.max_steps, desc="training")
    while step < args.max_steps:
        for batch in dataloader:
            clean = batch["video"].to(device)  # [B, T, C, H, W]
            clean = clean.permute(0, 2, 1, 3, 4).contiguous()  # [B, C, T, H, W]

            noise = torch.randn_like(clean)
            timesteps = torch.randint(
                0,
                scheduler.config.num_train_timesteps,
                (clean.shape[0],),
                device=device,
                dtype=torch.long,
            )
            noisy = scheduler.add_noise(clean, noise, timesteps)
            noise_pred = model(noisy, timesteps)
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % args.log_every == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

            if step % args.sample_every == 0 or step == args.max_steps - 1:
                sample = sample_video(
                    model,
                    scheduler,
                    shape=(1, 3, args.frames, args.image_size, args.image_size),
                    device=device,
                    num_inference_steps=args.inference_steps,
                )[0]
                sample_tchw = sample.permute(1, 0, 2, 3).contiguous()
                save_gif(sample_tchw, output_dir / f"sample_step_{step:04d}.gif", fps=args.fps)
                save_frame_grid(sample_tchw, output_dir / f"sample_step_{step:04d}_grid.png")
                score = estimate_motion_score(sample_tchw)
                (output_dir / "metrics.txt").open("a", encoding="utf-8").write(
                    f"step={step}, loss={loss.item():.6f}, frame_diff={score:.6f}\n"
                )
                model.train()

            step += 1
            progress.update(1)
            if step >= args.max_steps:
                break

    progress.close()
    torch.save(model.state_dict(), output_dir / f"{args.model}_toy_video_diffusion.pt")
    print(f"Saved outputs to {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tiny toy video diffusion model.")
    parser.add_argument("--model", choices=["framewise", "temporal"], default="framewise")
    parser.add_argument("--output-dir", default="unit5_outputs/toy_video_diffusion")
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset-size", type=int, default=2048)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--square-size", type=int, default=6)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--train-timesteps", type=int, default=1000)
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--sample-every", type=int, default=100)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
