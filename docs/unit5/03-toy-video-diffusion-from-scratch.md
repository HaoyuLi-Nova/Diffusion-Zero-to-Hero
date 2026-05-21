# Toy Video Diffusion from Scratch

本节是 Unit 5 第一阶段的核心实战：我们不用大模型，而是从零训练一个小型视频扩散模型，让它生成“彩色方块在画面中运动”的短视频。

本节配套的完整可运行脚本位于：

```text
unit5/toy_video_diffusion.py
```

你可以直接运行它，得到：

- `dataset_preview.gif`：合成数据样例；
- `dataset_preview_grid.png`：数据样例逐帧展开图；
- `sample_step_*.gif`：训练过程中的采样视频；
- `sample_step_*_grid.png`：采样视频逐帧展开图；
- `metrics.txt`：loss 和相邻帧差异；
- `framewise_toy_video_diffusion.pt` 或 `temporal_toy_video_diffusion.pt`：模型权重。

## 为什么这个实验值得做

视频生成最容易被误解成“重复生成很多张图像”。这个实验会让你亲眼看到：

- 把视频拆成逐帧图像处理，模型能学到单帧外观，但不天然理解运动；
- 加入一个非常小的 temporal convolution 后，模型至少有能力在相邻帧之间传递信息；
- 视频扩散的训练目标仍然是 DDPM 的 noise prediction，只是数据从 `[B, C, H, W]` 变成 `[B, C, T, H, W]`；
- 训练 loss 不是全部，必须保存 GIF/帧网格观察运动是否合理。

本实验的目标不是生成漂亮视频，而是建立真实视频模型的基本直觉。

## 环境准备

先安装仓库依赖：

```powershell
pip install -r requirements.txt
```

确认 PyTorch 可用：

```powershell
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
PY
```

如果你的机器没有 CUDA，也可以先用 CPU 跑一个很小的 smoke test，只是速度会慢。

## 直接运行 smoke test

这是最小可运行命令，用于确认脚本、数据生成、模型前向、训练一步、采样和 GIF 保存都能跑通：

```powershell
python unit5\toy_video_diffusion.py `
  --device cpu `
  --max-steps 1 `
  --sample-every 1 `
  --inference-steps 2 `
  --dataset-size 8 `
  --batch-size 2 `
  --image-size 16 `
  --square-size 4 `
  --frames 4 `
  --output-dir unit5_outputs\smoke_test
```

运行后应看到：

```text
unit5_outputs/smoke_test/
├── dataset_preview.gif
├── dataset_preview_grid.png
├── sample_step_0000.gif
├── sample_step_0000_grid.png
├── metrics.txt
└── framewise_toy_video_diffusion.pt
```

如果你看到 `ModuleNotFoundError: No module named 'torch'`，说明当前 Python 环境还没有安装 PyTorch。先回到上面的 `pip install -r requirements.txt`，或切换到你已有的 PyTorch 环境。

## 推荐训练命令

### 逐帧 baseline

先训练没有时间建模的 framewise baseline：

```powershell
python unit5\toy_video_diffusion.py `
  --model framewise `
  --image-size 32 `
  --square-size 6 `
  --frames 8 `
  --batch-size 16 `
  --max-steps 400 `
  --sample-every 100 `
  --inference-steps 50 `
  --output-dir unit5_outputs\framewise
```

如果有 CUDA，脚本会自动使用 GPU。也可以显式指定：

```powershell
python unit5\toy_video_diffusion.py --device cuda --model framewise --output-dir unit5_outputs\framewise
```

### Temporal convolution baseline

再训练加入 temporal convolution 的版本：

```powershell
python unit5\toy_video_diffusion.py `
  --model temporal `
  --image-size 32 `
  --square-size 6 `
  --frames 8 `
  --batch-size 16 `
  --max-steps 400 `
  --sample-every 100 `
  --inference-steps 50 `
  --output-dir unit5_outputs\temporal
```

你应该对比：

- `unit5_outputs/framewise/sample_step_0400.gif`
- `unit5_outputs/temporal/sample_step_0400.gif`

重点看方块颜色、位置、运动方向和背景是否稳定。不要只看最后一帧是否像图像。

## 脚本参数说明

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `framewise` | 可选 `framewise` 或 `temporal` |
| `--output-dir` | `unit5_outputs/toy_video_diffusion` | 输出 GIF、PNG、metrics 和权重的位置 |
| `--device` | 自动选择 | 留空时优先使用 CUDA，否则 CPU |
| `--dataset-size` | `2048` | 合成视频样本数量 |
| `--image-size` | `32` | 每帧分辨率 |
| `--square-size` | `6` | 移动方块边长 |
| `--frames` | `8` | 每个视频片段的帧数 |
| `--batch-size` | `16` | 训练 batch size |
| `--max-steps` | `400` | 训练步数 |
| `--learning-rate` | `1e-4` | AdamW 学习率 |
| `--train-timesteps` | `1000` | DDPM 训练噪声步数 |
| `--inference-steps` | `50` | 采样时使用的反向去噪步数 |
| `--sample-every` | `100` | 每隔多少步保存一次采样视频 |

显存不足时，优先降低：

1. `--batch-size`
2. `--frames`
3. `--image-size`
4. `--inference-steps`

## 数据集：真实可运行实现

脚本中的 `MovingSquareDataset` 不读取视频文件，而是在线生成短视频。每个样本返回：

```text
video: [T, C, H, W], float32, range [-1, 1]
```

核心代码如下，完整版本见 `unit5/toy_video_diffusion.py`：

```python
class MovingSquareDataset(Dataset):
    """Generate short videos of a colored square moving and bouncing.

    Each item returns a tensor with shape [T, C, H, W] in the range [-1, 1].
    The dataset is generated on the fly, so it does not need video files.
    """

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
```

这里用 `index` 作为随机种子，所以同一个 index 每次生成的视频一致。这让实验更容易复现，也方便你调试某一个样本。

## 可视化：保存 GIF 和帧网格

扩散训练不能只看 loss。视频任务必须保存动态结果。脚本里用 Pillow 保存 GIF，不需要额外安装 `imageio`：

```python
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
```

同时脚本会保存一张横向帧网格：

```python
def save_frame_grid(video: torch.Tensor, path: Path) -> None:
    """Save all frames of one video as a horizontal contact sheet."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [tensor_to_pil(frame) for frame in video]
    width, height = frames[0].size
    grid = Image.new("RGB", (width * len(frames), height))
    for idx, frame in enumerate(frames):
        grid.paste(frame, (idx * width, 0))
    grid.save(path)
```

GIF 用来看运动，帧网格用来看每帧细节。两者都要看。

## 模型 1：FramewiseVideoUNet

第一个模型复用 Diffusers 的 `UNet2DModel`。它把视频 `[B, C, T, H, W]` reshape 成 `[B*T, C, H, W]`，逐帧预测噪声：

```python
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
```

这个模型是真实可训练的，但它有一个故意保留的缺陷：它不知道第 1 帧和第 2 帧之间有什么关系。它只能学习“每一帧应该像移动方块图像”，不能稳定学习“方块如何连续移动”。

## 模型 2：TemporalConvVideoUNet

第二个模型在 framewise 预测之后加入一个小的 `Conv3d` residual block：

```python
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
```

这里的卷积核是 `(3, 1, 1)`，只沿时间维混合相邻帧，不扩大空间范围。它不是高级视频模型，但足以说明“时间维建模”这件事。

真实模型中的 3D UNet、temporal attention、motion module 都是在更大规模上解决类似问题：跨帧共享信息。

## 训练循环：真实 DDPM noise prediction

脚本使用 Diffusers 的 `DDPMScheduler`。训练目标和 Unit 1 的 DDPM 一样：给干净视频加噪，模型预测噪声。

关键代码：

```python
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
```

注意 shape 的转换：

```text
dataloader: [B, T, C, H, W]
model:      [B, C, T, H, W]
```

这是视频模型中非常常见的 bug 来源。每次写视频代码都应该打印 shape 或保存可视化样本确认。

## 采样循环：从噪声生成视频

采样从纯噪声视频开始：

```python
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
```

输出 shape 是：

```text
[B, C, T, H, W]
```

保存前转回：

```python
sample_tchw = sample[0].permute(1, 0, 2, 3).contiguous()
save_gif(sample_tchw, output_dir / f"sample_step_{step:04d}.gif", fps=args.fps)
```

## 简单指标：frame difference

脚本会记录一个非常简单的相邻帧差异：

```python
def estimate_motion_score(video: torch.Tensor) -> float:
    """Return a simple adjacent-frame difference score for inspection."""

    # video: [T, C, H, W], range [-1, 1]
    return (video[1:] - video[:-1]).abs().mean().item()
```

这个指标不能代表视频质量，但可以辅助观察：

- 差异接近 0：视频可能几乎静止；
- 差异特别大：可能有明显闪烁或跳变；
- 对 moving square，合理差异应集中在方块运动边缘。

真正判断仍然要打开 GIF 和帧网格。

## 如何阅读输出

训练输出目录中，建议按这个顺序检查：

1. `dataset_preview.gif`：确认数据本身是方块连续运动。
2. `dataset_preview_grid.png`：确认帧顺序、颜色、尺寸正常。
3. `sample_step_0000.gif`：随机初始化模型的输出，通常是噪声。
4. `sample_step_0100.gif`、`sample_step_0200.gif`：观察是否逐步出现方块结构。
5. `metrics.txt`：检查 loss 是否大致下降，frame_diff 是否异常。
6. framewise 与 temporal 的最终 GIF 对比：看 temporal 是否更稳定。

不要只挑最好看的一个 GIF。视频生成学习要保留失败样例，因为失败样例能告诉你模型到底缺什么。

## 常见问题

### 训练很慢

先降低配置：

```powershell
python unit5\toy_video_diffusion.py `
  --model framewise `
  --image-size 16 `
  --frames 4 `
  --batch-size 2 `
  --max-steps 20 `
  --inference-steps 5 `
  --output-dir unit5_outputs\debug
```

确认流程能跑通后，再增加分辨率、帧数和训练步数。

### GIF 里只有噪声

可能原因：

- 训练步数太少；
- `inference-steps` 太少；
- 学习率过高或过低；
- CPU smoke test 本来只用于检查流程，不用于看质量。

先用 400 到 1000 steps 训练，再判断模型能力。

### 方块出现但运动不稳定

这是 framewise baseline 的预期现象。它说明模型学到了单帧分布，但缺少时间建模。接着跑 `--model temporal`。

### 显存不足

优先降低：

```text
batch-size -> frames -> image-size -> inference-steps
```

不要一开始就用 `64x64`、`16` 帧、大 batch。toy 实验的价值是理解机制，不是追求分辨率。

## 与真实视频模型的对应关系

| 本实验 | 真实视频模型中的对应概念 |
| --- | --- |
| `[B, C, T, H, W]` | video latent / video tensor |
| framewise 2D UNet | 不含时间建模的逐帧 baseline |
| `Conv3d(kernel=(3,1,1))` | temporal convolution / motion block |
| 保存 GIF | 视频生成结果可视化 |
| frame difference | flickering / motion sanity check |
| moving square | object motion |

完成这个实验后，再看 AnimateDiff、Stable Video Diffusion、CogVideoX，会更容易理解它们为什么需要 motion module、temporal attention、video transformer、VAE slicing/tiling 和显存优化。

## 下一步

如果你要继续完善 Unit 5，可以基于这个脚本扩展：

1. 增加 Moving MNIST 数据集。
2. 在 temporal block 中加入 temporal attention。
3. 增加条件输入，例如速度方向 `[vx, vy]`。
4. 保存 MP4，而不仅是 GIF。
5. 增加一个 notebook，把脚本拆成交互式教学单元。

但第一步应该先把这个脚本跑通，并对比 framewise 和 temporal 两个版本的输出。
