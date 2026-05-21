# 视频生成前置知识

本节回答一个核心问题：如果你已经理解图像扩散模型，进入视频生成时到底需要补什么？

简短答案是：视频生成不是把图像生成重复 `T` 次。它需要同时建模空间结构、时间连续性、主体一致性和运动规律。

本单元不是纯概念阅读。读完本节后，你可以运行：

```powershell
python unit5\toy_video_diffusion.py `
  --model framewise `
  --image-size 32 `
  --frames 8 `
  --batch-size 16 `
  --max-steps 400 `
  --sample-every 100 `
  --output-dir unit5_outputs\framewise
```

它会训练一个逐帧视频扩散 baseline。后续再运行 `--model temporal`，对比加入时间卷积后的视频稳定性。

## 从图像扩散到视频扩散

在图像扩散中，模型学习从噪声恢复图像。以 Stable Diffusion 为例，扩散通常发生在 latent space 中：

```text
prompt
-> tokenizer / text encoder
-> noisy latent [B, 4, H/8, W/8]
-> UNet denoising
-> scheduler step
-> VAE decoder
-> image [B, 3, H, W]
```

视频扩散把“单张图像”换成“帧序列”：

```text
prompt or image condition
-> text/image encoder
-> noisy video latent [B, C, T, H, W]
-> video denoiser
-> scheduler step
-> video decoder
-> frames [B, T, 3, H_out, W_out]
```

多出来的 `T` 是时间维。它带来两个直接后果：

1. 数据量成倍增加。16 帧视频大致相当于 16 张图像一起处理。
2. 模型必须知道不同帧属于同一个样本，而不是互不相关的图片。

如果模型只在每一帧上独立做图像生成，它也许能生成每帧都不错的图片，但视频会闪烁、不连贯、不像同一个事件。

## 视频生成任务的几种形式

视频生成不是单一任务。学习时应区分以下几类：

| 任务 | 输入 | 输出 | 学习难度 | 典型用途 |
| --- | --- | --- | --- | --- |
| Unconditional Video Generation | 无条件或类别标签 | 短视频 | 中 | 理解基础生成过程 |
| Text-to-Video | 文本 prompt | 视频 | 高 | 文本驱动的内容创作 |
| Image-to-Video | 初始图像 | 视频 | 中 | 让静态图动起来 |
| Video-to-Video | 输入视频 + 条件 | 新视频 | 高 | 风格转换、编辑、增强 |
| Controllable Video Generation | 文本 + 姿态/深度/轨迹等 | 可控视频 | 高 | 动作、结构、镜头控制 |

本课程建议从 **Image-to-Video** 和 toy video diffusion 入门。原因是输入图像固定了主体外观和风格，读者更容易观察“运动是否合理”和“时间是否稳定”。

Text-to-Video 更开放，但难度更高：模型既要决定画面内容，又要决定动作和镜头变化。初学者如果直接从 T2V 大模型开始，常常只能学会调 prompt，而不理解模型为什么失败。

## 视频生成的核心难点

### 1. 时序一致性

时序一致性指相邻帧之间的内容、结构、纹理、光照和身份保持连贯。

常见失败包括：

- 人脸在不同帧中变形。
- 衣服花纹闪烁。
- 背景边缘抖动。
- 物体突然消失或复制。
- 前景和背景运动方向不一致。

图像生成只需要“这一张图像看起来合理”。视频生成还要保证“这一帧和上一帧共同构成合理运动”。

### 2. 运动建模

视频需要生成动作，而不只是生成多张类似图片。动作包括：

- 主体运动：人走路、车移动、动物奔跑。
- 局部运动：头发飘动、水波、火焰、表情变化。
- 相机运动：推近、拉远、平移、环绕。
- 场景变化：天气变化、光照变化、物体交互。

运动建模要求模型理解时间方向和速度。只生成“相似但略有变化的图”不等于生成视频。

### 3. 主体一致性

Text-to-Video 中，主体通常由 prompt 描述，例如“a red car driving on a mountain road”。模型需要在所有帧中保持：

- 同一辆红车；
- 大致一致的车型和颜色；
- 合理的位置变化；
- 背景视角和道路结构连续。

主体一致性差时，视频看起来像一组随机相似图片，而不是同一个镜头。

### 4. 显存与计算成本

视频 latent 的 token 数比图像大很多。假设图像 latent 是：

```text
[B, 4, 64, 64]
```

16 帧视频 latent 可能是：

```text
[B, 4, 16, 64, 64]
```

token 数直接乘以 16。若模型使用时空 attention，显存成本会进一步上升。因此视频生成常常需要：

- 降低分辨率；
- 限制帧数；
- 使用更小 batch size；
- 开启 CPU offload；
- 使用 VAE slicing / tiling；
- 使用混合精度；
- 对长视频分段生成或后处理。

### 5. 数据质量

视频数据比图像数据更难处理。一个视频样本不仅需要画面质量好，还需要：

- 镜头切分合理；
- 运动连续；
- caption 描述准确；
- fps 和长度统一；
- 水印、字幕、转场、黑边尽量少；
- 避免多镜头混剪被当成一个连续样本。

如果训练数据中充满突然剪辑，模型可能学到“视频可以随时跳变”，生成时就更容易闪烁或断裂。

## 常见视频模型设计

不同视频扩散模型会用不同方式引入时间维。你不需要一开始掌握所有细节，但需要知道它们在解决同一件事：让模型跨帧共享信息。

### 逐帧 2D UNet baseline

最简单的 baseline 是把每一帧当图像处理：

```text
[B, C, T, H, W]
-> reshape to [B*T, C, H, W]
-> 2D UNet
-> reshape back to [B, C, T, H, W]
```

优点是容易实现，可以复用图像扩散代码。缺点是模型看不到时间关系，容易闪烁。

这个 baseline 很适合教学，因为它能清楚展示“缺少时间建模会发生什么”。

### 3D UNet

3D UNet 把卷积或注意力扩展到时间维：

```text
2D conv: kernel over H, W
3D conv: kernel over T, H, W
```

它能同时处理空间和时间信息，但显存更高，训练更重。

### Temporal Attention

Temporal attention 让同一空间位置或同一 token 在不同帧之间交换信息。直觉上，它让模型回答：

- 这个物体在上一帧在哪里？
- 当前帧应该如何移动？
- 哪些纹理和身份信息应该保持？

许多现代视频模型都会结合 spatial attention 和 temporal attention。

### Motion Module

一种实用路线是在已有图像生成模型上加 motion module。这样可以保留图像模型的外观生成能力，再额外学习运动。AnimateDiff 就是这类思想的代表之一。

对学习者来说，这条路线很重要：它说明视频生成不一定要从零训练一个巨大模型，也可以在图像模型基础上加入时间建模模块。

### Video Transformer

更大规模的视频生成模型常使用 Transformer 或 DiT 类结构，把视频表示为时空 token 序列。它的扩展性强，但对数据、算力和工程实现要求更高。

Unit 5 第一阶段不会实现 Video Transformer，但已经提供 `toy_video_diffusion.py`，用可运行的 framewise baseline 和 temporal convolution baseline 保留相同的核心概念：视频可以被看成一组时空 token，需要跨时间交换信息。

## 入门时应该先学什么

推荐顺序：

1. 用 toy data 理解 `[B, T, C, H, W]`。
2. 实现逐帧 2D baseline，观察闪烁。
3. 加入简单 temporal module，观察一致性变化。
4. 用 Image-to-Video pipeline 理解真实模型如何利用输入图像固定主体。
5. 用 CogVideoX 等 Text-to-Video pipeline 理解 prompt、帧数、显存和采样参数。
6. 最后进入可控视频生成和评估。

这个顺序比“直接运行最大的视频模型”更慢一点，但更适合真正学会视频生成。

## 最小术语表

| 术语 | 含义 |
| --- | --- |
| Frame | 视频中的单帧图像 |
| FPS | 每秒帧数，决定播放速度 |
| Clip | 一段固定长度的视频片段 |
| Temporal Consistency | 帧间主体、纹理、结构和运动保持一致 |
| Flickering | 闪烁，常指纹理或背景在帧间抖动 |
| Motion Module | 在图像模型上增加的运动建模模块 |
| Image-to-Video | 输入图像，生成动态视频 |
| Text-to-Video | 输入文本，生成视频 |
| Video Latent | 视频在压缩潜空间中的表示 |
| Camera Motion | 相机推拉、平移、旋转、环绕等运动 |

## 本节小结

视频生成的关键问题可以压缩为一句话：

> 图像扩散学习“每一张图像应该长什么样”，视频扩散还要学习“这些图像如何在时间上组成同一个事件”。

下一节会从最基础的视频张量和数据集开始，说明如何把视频变成模型能处理的训练样本。
