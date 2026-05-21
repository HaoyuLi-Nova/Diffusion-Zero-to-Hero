# 视频张量与数据集

进入视频生成前，必须先弄清楚视频如何变成张量。很多视频模型训练失败，不是因为扩散公式错了，而是因为帧采样、尺寸、归一化、时间顺序或 batch 组织出了问题。

本节目标是建立一个可靠的数据处理心智模型，为后续 toy video diffusion 和真实视频 pipeline 做准备。

## 视频文件到张量

一个视频文件通常包含：

- 编码格式：如 H.264、H.265、AV1。
- 容器格式：如 `.mp4`、`.mov`、`.webm`。
- 帧率：如 8 fps、24 fps、30 fps。
- 分辨率：如 512x512、720p、1080p。
- 音频轨：视频生成训练中通常先忽略。

模型不能直接读取 `.mp4`。我们需要把视频解码成帧序列：

```text
video file
-> decode frames
-> sample T frames
-> resize / crop
-> normalize
-> tensor [T, C, H, W]
```

多个样本组成 batch 后：

```text
batch video tensor: [B, T, C, H, W]
```

有些模型或算子更喜欢 channel-first 的 video layout：

```text
batch video tensor: [B, C, T, H, W]
```

两种布局都常见，关键是全项目保持一致，并在进入模型前明确转换。

## 两种常见张量布局

### `[B, T, C, H, W]`

这种布局更符合人类直觉：

- `B`：batch size
- `T`：frames
- `C`：channels
- `H`：height
- `W`：width

当你在 dataset 和 dataloader 中处理视频时，这种形式更容易读：

```python
video.shape  # [batch, frames, channels, height, width]
```

### `[B, C, T, H, W]`

这种布局更适合 3D convolution 和很多视频模型：

```python
video = video.permute(0, 2, 1, 3, 4)
video.shape  # [batch, channels, frames, height, width]
```

如果后续用 3D UNet、temporal convolution 或 video VAE，经常会看到 `[B, C, T, H, W]`。

### 课程建议

在 Unit 5 中建议：

- dataset 输出 `[T, C, H, W]`
- dataloader 组合后为 `[B, T, C, H, W]`
- 进入模型前再转成 `[B, C, T, H, W]`

这样便于教学，也便于和常见 PyTorch 视频处理代码对齐。

## 帧采样

视频通常很长，但训练时不能把完整视频全部送进模型。我们会从视频中抽取固定长度的 clip。

### 连续采样

从某个起点开始，取连续 `T` 帧：

```text
frames[start : start + T]
```

优点是运动最连续。缺点是如果原视频 fps 很高，相邻帧变化可能太小，模型学到的运动弱。

### 步长采样

每隔 `stride` 帧取一帧：

```text
frames[start : start + T * stride : stride]
```

例如：

```text
T = 16, stride = 2
```

表示从原视频中覆盖 32 帧的时间跨度，但只取 16 帧。这样能看到更明显的运动。

### 均匀采样

在一段视频里均匀取 `T` 帧。适合推理或展示，但训练时如果跨越太长，可能把镜头切换也采进去。

### 随机窗口采样

训练时常用：

1. 随机选择一个视频；
2. 随机选择一个起点；
3. 按固定 `T` 和 `stride` 取 clip；
4. 做 resize/crop/flip 等增强。

这能增加数据多样性，但要避免采到跨镜头转场。

## FPS 与播放速度

fps 决定视频播放时每秒显示多少帧。训练时要区分：

- 原始视频 fps；
- 训练采样后的有效 fps；
- 保存生成视频时使用的 fps。

例如原视频是 24 fps，你用 `stride=3` 采样，那么训练 clip 的有效时间间隔相当于 8 fps。保存生成视频时如果仍用 24 fps 播放，动作会显得更快。

教学项目中建议统一：

```text
frames = 16
fps = 8
resolution = 64 or 128
```

这样便于观察，也便于控制显存。

## 尺寸处理

视频模型对尺寸非常敏感。常见处理包括：

- resize：直接缩放到目标尺寸。
- center crop：保留中心区域。
- random crop：训练时增加数据变化。
- pad：保持比例，用边界填充。

toy video diffusion 建议直接使用合成数据，天然生成固定尺寸，例如：

```text
64x64, 16 frames
```

真实视频数据则建议先做离线预处理，避免训练时反复解码和 resize 导致速度慢。

## 像素归一化

扩散模型通常使用 `[-1, 1]` 范围的输入：

```python
video = video.float() / 255.0
video = video * 2.0 - 1.0
```

显示时再转回 `[0, 1]` 或 `[0, 255]`：

```python
video = (video + 1.0) / 2.0
video = video.clamp(0, 1)
```

如果归一化不一致，模型训练会明显不稳定。比如训练时用 `[-1, 1]`，采样显示时忘记转回，就会看到发黑或发灰的结果。

## Toy 数据集设计

第一阶段建议先用合成数据，而不是互联网视频。合成数据的好处是：

- 不需要下载大文件；
- 没有版权和隐私问题；
- 运动规律可控；
- 失败原因更容易分析；
- 可以精确生成 ground truth。

### 移动方块

最小数据集：一个彩色方块在黑色背景上移动。

可控制变量：

- 方块颜色；
- 方块大小；
- 初始位置；
- 运动方向；
- 运动速度；
- 是否反弹；
- 是否加入背景噪声。

这个任务适合验证模型是否学会基本位移。

### 移动圆点

圆点比方块边缘更平滑，适合观察模糊和轨迹。

可以设置：

- 匀速直线；
- 加速；
- 圆周运动；
- 多个圆点交叉运动。

### Moving MNIST

Moving MNIST 是经典 toy video dataset：数字在画面中移动并反弹。它比方块更复杂，因为数字形状有语义结构。

适合观察：

- 数字身份是否保持；
- 边缘是否闪烁；
- 运动是否连续；
- 反弹是否合理。

### 简单相机运动

可以生成一个大背景图，然后裁剪窗口随时间移动，模拟相机 pan 或 zoom。

这能帮助理解 camera motion 与 object motion 的区别：

- object motion：物体在画面中移动；
- camera motion：整个视角在移动。

## Dataset 输出约定

建议 toy dataset 的 `__getitem__` 输出：

```python
{
    "video": video,        # [T, C, H, W], float32, range [-1, 1]
    "label": label,        # optional
    "metadata": metadata,  # optional, e.g. velocity, color, trajectory
}
```

如果后续要做条件生成，可以加入：

```python
{
    "video": video,
    "condition": {
        "velocity": [vx, vy],
        "start": [x0, y0],
        "class": class_id,
        "trajectory": points,
    }
}
```

但第一版不要过早复杂化。先让无条件 toy video diffusion 跑通。

## 最小数据生成伪代码

下面是移动方块数据集的核心逻辑：

```python
def make_moving_square(
    num_frames=16,
    image_size=64,
    square_size=8,
    velocity=(2, 1),
):
    video = torch.zeros(num_frames, 3, image_size, image_size)

    x = torch.randint(0, image_size - square_size, ()).item()
    y = torch.randint(0, image_size - square_size, ()).item()
    vx, vy = velocity
    color = torch.rand(3, 1, 1)

    for t in range(num_frames):
        frame = torch.zeros(3, image_size, image_size)
        frame[:, y:y + square_size, x:x + square_size] = color
        video[t] = frame

        x += vx
        y += vy

        if x < 0 or x > image_size - square_size:
            vx = -vx
            x += 2 * vx
        if y < 0 or y > image_size - square_size:
            vy = -vy
            y += 2 * vy

    video = video * 2.0 - 1.0
    return video
```

这个函数生成的是 `[T, C, H, W]`。进入模型前可变成：

```python
video = video.unsqueeze(0)              # [B, T, C, H, W]
video = video.permute(0, 2, 1, 3, 4)    # [B, C, T, H, W]
```

## 保存和可视化

训练视频模型时，必须经常保存采样结果。只看 loss 不够，因为 loss 降低不代表运动合理。

建议保存：

- 单帧网格：快速看图像质量；
- GIF：快速看运动；
- MP4：更接近真实发布；
- 帧差图：观察运动和闪烁。

最小可视化方式：

```python
def to_display(video):
    # video: [T, C, H, W], range [-1, 1]
    video = (video + 1.0) / 2.0
    video = video.clamp(0, 1)
    video = (video * 255).byte()
    return video.permute(0, 2, 3, 1)  # [T, H, W, C]
```

保存 GIF 时可以使用 `imageio`：

```python
import imageio.v3 as iio

frames = to_display(video).cpu().numpy()
iio.imwrite("sample.gif", frames, duration=125)  # 8 fps
```

## 常见数据错误

| 问题 | 现象 | 检查方法 |
| --- | --- | --- |
| 时间顺序反了 | 动作倒放或训练不稳定 | 显示原始 clip |
| 维度 permute 错 | 颜色异常、形状异常 | 打印 shape 并可视化第一帧 |
| 归一化不一致 | 图像发黑、发灰或过曝 | 检查 min/max |
| 帧采样跨度太大 | 动作跳跃 | 降低 stride |
| 帧采样跨度太小 | 视频几乎静止 | 增大 stride |
| crop 太激进 | 主体经常被裁掉 | 保存预处理后的样本 |
| 混入镜头切换 | 模型学到突然跳变 | 做镜头切分或过滤 |

## 推荐的第一版配置

为了让后续 toy video diffusion 更容易复现，建议第一版固定：

```text
dataset: moving square
frames: 16
channels: 3
resolution: 64x64
batch size: 16 or 32
pixel range: [-1, 1]
layout in dataset: [T, C, H, W]
layout in model: [B, C, T, H, W]
```

这组配置足够小，适合在普通 GPU 上测试，也可以在 CPU 上跑小规模调试。

## 本节小结

视频数据处理的核心不是“读取 mp4”，而是稳定地产生模型期望的固定形状张量：

```text
[B, T, C, H, W] -> [B, C, T, H, W]
```

同时要保证帧顺序、采样间隔、归一化和可视化都正确。下一节会基于这个数据约定，设计一个从零开始的 toy video diffusion。
