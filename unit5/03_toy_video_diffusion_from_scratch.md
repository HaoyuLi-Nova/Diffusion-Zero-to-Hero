# Toy Video Diffusion from Scratch

本节设计一个最小的视频扩散实验。它的目标不是生成漂亮视频，而是让你清楚看到：如果模型不处理时间维，视频会出现什么问题；加入最小时间建模后，结果为什么会更稳定。

这也是后续进入 AnimateDiff、Stable Video Diffusion、CogVideoX 等真实模型前最重要的直觉训练。

## 实验目标

我们要训练一个模型生成短视频：

```text
16 frames, 64x64, moving square
```

输入是随机噪声，输出是一段方块运动视频。这个任务很小，但它包含视频生成的核心要素：

- 每一帧都要像一个合理图像；
- 方块身份和颜色要保持一致；
- 方块位置要随时间连续变化；
- 运动方向不能随机跳变；
- 背景不应该闪烁。

## 为什么不用真实视频起步

真实视频会引入太多变量：

- 视频压缩噪声；
- 镜头切换；
- 字幕和水印；
- 复杂背景；
- 多主体交互；
- caption 不准确；
- 分辨率和帧率不统一。

这些都不适合第一版教学。toy 数据集虽然简单，但能让我们明确知道 ground truth 运动规律。如果模型生成失败，我们更容易定位问题。

## 扩散流程回顾

图像 DDPM 的训练目标通常是：

```text
x_0: clean data
epsilon: Gaussian noise
t: timestep
x_t = add_noise(x_0, epsilon, t)
model predicts epsilon from x_t and t
loss = MSE(predicted_epsilon, epsilon)
```

视频扩散可以使用同样目标，只是 `x_0` 从图像换成视频：

```text
x_0: clean video [B, C, T, H, W]
epsilon: Gaussian noise with same shape
t: timestep
x_t: noisy video
model(x_t, t) -> predicted noise
loss = MSE(predicted_noise, epsilon)
```

最小训练循环仍然是：

```python
for batch in dataloader:
    clean_video = batch["video"]             # [B, T, C, H, W]
    clean_video = clean_video.permute(0, 2, 1, 3, 4)

    noise = torch.randn_like(clean_video)
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,))
    noisy_video = scheduler.add_noise(clean_video, noise, timesteps)

    noise_pred = model(noisy_video, timesteps)
    loss = F.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
```

从公式上看，视频扩散没有神秘变化。真正的变化在模型结构：模型必须处理 `[B, C, T, H, W]` 中的时间维。

## Baseline 1：逐帧 2D UNet

第一版 baseline 故意不做时间建模。它把视频拆成 `B*T` 张图像：

```text
input: [B, C, T, H, W]
permute / reshape: [B*T, C, H, W]
2D UNet
reshape back: [B, C, T, H, W]
```

伪代码：

```python
class FramewiseUNet(nn.Module):
    def __init__(self, image_unet):
        super().__init__()
        self.image_unet = image_unet

    def forward(self, x, timesteps):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        x_frames = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)

        # same timestep for every frame in the same video
        t_frames = timesteps[:, None].repeat(1, t).reshape(b * t)

        pred = self.image_unet(x_frames, t_frames)
        pred = pred.reshape(b, t, c, h, w).permute(0, 2, 1, 3, 4)
        return pred
```

这个模型可能学会生成“有方块的帧”，但它没有机制保证第 `t` 帧和第 `t+1` 帧连续。预期问题包括：

- 方块位置跳动；
- 方块颜色变化；
- 背景噪声闪烁；
- 运动方向不稳定。

这不是失败，而是教学目标：它证明逐帧图像生成不等于视频生成。

## Baseline 2：加入最小 temporal convolution

下一步加入一个非常小的 temporal module。最简单做法是在若干层特征上做 3D convolution，或者在输入/输出附近做时间维混合。

概念伪代码：

```python
class TinyTemporalBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )

    def forward(self, x):
        return x + self.net(x)
```

这里的卷积核是 `(3, 1, 1)`，表示只沿时间维混合相邻帧，不扩大空间卷积范围。

它的作用是让模型在预测当前帧噪声时看到前后帧信息。对于移动方块任务，这通常能改善：

- 方块颜色一致性；
- 位置连续性；
- 背景稳定性；
- 运动方向连续性。

## Baseline 3：简单 temporal attention

如果要更接近真实视频模型，可以加入 temporal attention。直觉是：对每个空间位置，让不同时间帧之间互相注意。

概念上：

```text
feature: [B, C, T, H, W]
rearrange -> [B*H*W, T, C]
apply self-attention over T
rearrange back -> [B, C, T, H, W]
```

伪代码：

```python
class TinyTemporalAttention(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x):
        # x: [B, C, T, H, W]
        b, c, t, h, w = x.shape
        tokens = x.permute(0, 3, 4, 2, 1).reshape(b * h * w, t, c)
        tokens_norm = self.norm(tokens)
        out, _ = self.attn(tokens_norm, tokens_norm, tokens_norm)
        tokens = tokens + out
        return tokens.reshape(b, h, w, t, c).permute(0, 4, 3, 1, 2)
```

这个模块比 temporal convolution 更贵，但表达能力更强。它能学习长一点的时间依赖，而不只看相邻帧。

第一版实现时不一定要做 attention。建议先完成 framewise baseline 和 temporal convolution，再考虑 attention。

## 模型对比计划

第一阶段最重要的不是单个模型，而是对比：

| 模型 | 时间建模 | 预期结果 |
| --- | --- | --- |
| Framewise 2D UNet | 无 | 单帧可能像样，但运动不稳定 |
| 2D UNet + Temporal Conv | 局部时间混合 | 方块运动更连续，颜色更稳定 |
| 2D UNet + Temporal Attention | 跨帧注意力 | 更强一致性，但显存更高 |

每个模型都在同一 toy dataset 上训练，使用相同 scheduler、timesteps、batch size 和采样配置。这样才能判断时间模块是否真的有帮助。

## 采样过程

采样时从纯噪声视频开始：

```python
video = torch.randn(batch_size, channels, frames, height, width).to(device)

for t in scheduler.timesteps:
    with torch.no_grad():
        noise_pred = model(video, t)
    video = scheduler.step(noise_pred, t, video).prev_sample
```

最后把 `[B, C, T, H, W]` 转回 `[B, T, C, H, W]` 并保存：

```python
video = video.permute(0, 2, 1, 3, 4)
```

注意：采样阶段最容易暴露模型是否学到了时间结构。训练 loss 下降，但采样视频仍然闪烁，是视频模型中很常见的现象。

## 观察指标

第一阶段不追求复杂指标，先做可解释的人工和简单数值检查。

### 人眼检查

每次保存 GIF 或 MP4，观察：

- 方块颜色是否保持；
- 方块大小是否稳定；
- 方块运动是否连续；
- 是否出现突然跳跃；
- 背景是否闪烁；
- 是否生成多个方块；
- 是否出现方块消失。

### 帧差图

计算相邻帧差：

```python
diff = (video[:, 1:] - video[:, :-1]).abs().mean(dim=2)
```

对于移动方块，帧差应该主要集中在方块运动边缘。如果整个背景都有高差异，说明闪烁明显。

### 轨迹估计

对于 moving square，可以用阈值找到方块中心：

```python
mask = frame.mean(dim=0) > threshold
y, x = mask.nonzero(as_tuple=True)
center = torch.stack([x.float().mean(), y.float().mean()])
```

然后画出中心随时间变化的轨迹。好模型应该产生平滑轨迹。

## 训练配置建议

第一版推荐：

```text
data: moving square
frames: 16
resolution: 64
channels: 3
batch size: 16
train timesteps: 1000
sampling steps: 50 or 100
optimizer: AdamW
learning rate: 1e-4
prediction target: epsilon
loss: MSE
```

如果显存不足：

- 降到 `32x32`
- 降到 `8` 帧
- batch size 降到 `4` 或 `8`
- 先只跑 framewise baseline

## 课程 notebook 的建议结构

后续 notebook 可以这样组织：

```text
01 imports and config
02 make moving square dataset
03 visualize dataset samples
04 define scheduler
05 define framewise model
06 train framewise model
07 sample and save GIF
08 add temporal convolution
09 train temporal model
10 compare framewise vs temporal
11 failure analysis
```

这比直接展示一个完整大模型更适合作为“从零开始”教程。

## 失败模式与解释

| 失败现象 | 可能原因 | 处理方式 |
| --- | --- | --- |
| 方块位置随机跳 | 没有时间建模或训练不足 | 加 temporal conv/attention |
| 方块颜色变化 | 模型没有保持主体 identity | 增加时间模块，降低数据随机性 |
| 背景闪烁 | 逐帧预测不稳定 | 加 temporal module，检查采样步数 |
| 方块消失 | 模型没有学好数据分布 | 训练更久，简化数据 |
| 生成多个方块 | 数据或模型容量不匹配 | 降低学习率，增加训练样本 |
| 运动太小 | stride 或数据速度太低 | 增大 velocity |
| 运动太跳 | velocity 太大或帧数太少 | 降低 velocity 或增加 fps |

## 与真实视频模型的对应关系

toy video diffusion 中的模块，对应真实视频模型中的概念：

| Toy 实验 | 真实模型概念 |
| --- | --- |
| `[B, C, T, H, W]` video tensor | video latent |
| framewise 2D UNet | 逐帧图像扩散 baseline |
| temporal convolution | 3D UNet / temporal block |
| temporal attention | video attention / temporal transformer |
| moving square trajectory | object motion |
| frame difference | flickering / temporal consistency check |

理解这些对应关系后，再看 AnimateDiff、Stable Video Diffusion 或 CogVideoX，就不会只停留在 pipeline 调用层面。

## 本节小结

Toy Video Diffusion 的价值在于建立可控实验环境：

1. 数据简单，运动规律明确。
2. 模型小，可以快速迭代。
3. baseline 对比清楚。
4. 失败模式容易解释。
5. 概念能直接迁移到真实视频生成模型。

下一阶段可以在此基础上补一个完整 notebook，实现 moving square 数据集、framewise baseline、temporal convolution baseline，并输出 GIF 对比。
