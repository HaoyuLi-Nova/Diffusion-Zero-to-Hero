# Unit 5：视频生成从零开始

Unit 5 是本课程新增的视频生成主线。它不是 Unit 4 中“视频生成概览”的简单扩写，而是从图像扩散过渡到视频扩散的一套完整入门路径：先理解视频数据和时间维度，再用一个小型 toy video diffusion 建立直觉，最后再进入 Image-to-Video、Text-to-Video 和可控视频生成。

本阶段先完成前三个基础文档，目标是让读者在不依赖大模型和高显存环境的情况下，先弄清楚视频生成到底比图像生成多了什么。

## 为什么视频生成需要单独成课

图像扩散模型通常处理的是一张图像或一批图像，核心张量可以理解为：

```text
image: [B, C, H, W]
latent image: [B, C_latent, H/8, W/8]
```

视频生成处理的是帧序列，核心张量多了时间维：

```text
video: [B, T, C, H, W] 或 [B, C, T, H, W]
latent video: [B, C_latent, T, H/8, W/8]
```

这个额外的 `T` 不是普通的 batch 维度。它表示同一个场景、主体和运动过程在时间上的展开。模型不仅要生成每一帧好看的图像，还要保证相邻帧之间的主体、纹理、光照、动作和相机运动连贯。

如果把视频简单拆成独立图片逐帧生成，通常会出现：

- 主体身份漂移：同一个人、动物或物体在不同帧里长相变化。
- 背景闪烁：墙面、天空、纹理、边缘在帧间抖动。
- 动作断裂：运动方向忽然改变，速度不连续。
- 构图不稳：镜头位置、主体大小、视角在无意识地跳动。
- 文本不服从：prompt 描述的是一个动作，但视频只生成了静态图像抖动。

因此视频生成的关键不是“把图像模型运行很多次”，而是让模型理解并生成时间结构。

## 学习目标

完成 Unit 5 第一阶段后，你应该能回答：

1. 视频生成为什么比图像生成更难？
2. `[B, T, C, H, W]` 和 `[B, C, T, H, W]` 的区别是什么？
3. 为什么训练和推理时要控制帧数、分辨率、fps、stride？
4. 什么是时序一致性，常见失败模式有哪些？
5. toy video diffusion 应该如何设计，才能说明视频扩散的核心问题？
6. 为什么真实视频模型通常会使用 3D UNet、temporal attention、motion module 或 video transformer？

## 推荐学习顺序

| 顺序 | 文档 | 目标 |
| --- | --- | --- |
| 1 | [视频生成前置知识](01-video-generation-prerequisites.md) | 从图像扩散过渡到视频扩散，理解任务定义与核心难点 |
| 2 | [视频张量与数据集](02-video-tensors-and-datasets.md) | 学会把视频变成模型可用的张量，理解采样、帧率、窗口和预处理 |
| 3 | [Toy Video Diffusion](03-toy-video-diffusion-from-scratch.md) | 用最小实验建立视频扩散直觉，理解时间建模为什么必要 |

第一阶段完成后，再进入后续实战：

| 后续主题 | 作用 |
| --- | --- |
| Image-to-Video with Diffusers | 从输入图像出发生成短视频，学习成本低，主体保持更稳定 |
| Text-to-Video with CogVideoX | 使用现代开源 T2V 模型理解真实 pipeline、显存优化和参数控制 |
| Temporal Consistency and Evaluation | 系统记录失败模式，而不是只挑好看的样例 |
| Controllable Video Generation | 进入 pose、depth、trajectory、camera motion 等控制信号 |

## 与前面单元的关系

Unit 5 默认你已经理解：

- Unit 1：forward diffusion、reverse denoising、scheduler、noise prediction。
- Unit 2：guidance 和 conditioning 的基本概念。
- Unit 3：Stable Diffusion 的 VAE、text encoder、UNet、latent space、CFG。
- Unit 4：DDIM inversion、音频扩散、视频生成概览和现代模型路线。

如果你只关心视频生成，最低前置路径是：

```text
Unit 1 Diffusers 入门
-> Unit 3 Stable Diffusion 入门
-> Unit 5 视频生成从零开始
```

Unit 2 和 Unit 4 可以边学边回看。特别是当你进入可控视频生成时，Unit 2 的 conditioning 和 Unit 3 的 cross-attention 会非常重要。

## 第一阶段的工程边界

第一阶段不直接训练大型文生视频模型。原因是：

- 显存门槛高，普通读者很难复现。
- 大模型 pipeline 容易让人只会调参数，不理解视频维度。
- 如果没有 toy 任务，读者很难判断失败来自模型、数据、prompt、scheduler 还是显存压缩策略。

第一阶段推荐使用：

- 小尺寸：`32x32` 或 `64x64`
- 短序列：`8` 到 `16` 帧
- 简单数据：移动方块、移动圆点、Moving MNIST、合成轨迹
- 小模型：先用 2D UNet baseline，再加入最小 temporal module

这种设置生成效果不会惊艳，但它能清楚说明视频扩散的核心问题：时间维度必须被建模。

## 后续扩展方向

完成第一阶段后，Unit 5 会继续补：

1. Image-to-Video：从 Stable Video Diffusion / I2V pipeline 入门。
2. Text-to-Video：从 CogVideoX 的 Diffusers pipeline 入门。
3. 显存优化：`torch_dtype`、CPU offload、VAE slicing/tiling、帧数和分辨率控制。
4. 评估：prompt 记录、失败模式表、帧间差异、主体一致性、人眼检查协议。
5. 控制：pose-to-video、depth-to-video、trajectory control、camera control。

这条线的最终目标不是只会运行一个视频模型，而是能看懂并改造开源视频扩散项目。
