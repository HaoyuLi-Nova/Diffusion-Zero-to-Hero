# 现代扩散模型路线图

原课程很好地覆盖了 DDPM、Diffusers、微调、Stable Diffusion、DDIM inversion 和音频扩散。但扩散模型生态在 2023 年之后变化很快：低成本微调、结构控制、更大模型、快速采样、Transformer 架构和视频生成已经成为实际应用中的核心能力。

这份路线图用于衔接课程之后的学习。

## 1. 低成本个性化：LoRA 与 PEFT

DreamBooth 能让模型学习新主体，但全量微调显存高、容易过拟合、模型文件大。现代实践中更常用 **LoRA（Low-Rank Adaptation）**：

- 只训练少量低秩矩阵，显存和存储成本低。
- 可叠加多个 LoRA 控制风格、角色、服装或场景。
- 更适合个人 GPU 和公开分享。

建议学习顺序：

1. 先理解 DreamBooth 的主体绑定与唯一标识符。
2. 再学习 LoRA 如何注入 attention/linear 层。
3. 比较全量微调、Textual Inversion、DreamBooth LoRA 的效果与成本。

推荐后续新增 notebook：

- `extensions/lora_finetuning.ipynb`
- 数据准备、caption 策略、rank/alpha、学习率、过拟合判断

## 2. 结构控制：ControlNet、T2I-Adapter、IP-Adapter

仅靠 prompt 很难稳定控制姿态、构图、边缘和空间关系。现代图像生成常用额外条件控制：

- **ControlNet**：用 Canny、depth、pose、segmentation 等结构图控制生成。
- **T2I-Adapter**：更轻量的条件适配器。
- **IP-Adapter**：用参考图像控制主体、风格或构图。

建议学习重点：

- ControlNet 不替代 prompt，而是补充「结构约束」。
- 条件强度太高会限制创意，太低则失去控制。
- 多 ControlNet 可以叠加，但显存和调参复杂度会上升。

推荐后续新增 notebook：

- `extensions/controlnet_basics.ipynb`
- Canny/depth/openpose 条件图生成
- prompt 与 control strength 的网格对比

## 3. 更强基础模型：SDXL 与后续模型

Stable Diffusion 1.x/2.x 是理解体系结构的好入口，但现代应用常用更强基础模型：

- **SDXL**：更强 prompt 理解、更高分辨率、base/refiner 设计。
- **Turbo/Lightning/LCM**：更少步数的快速生成。
- **Flux / Flow-based models**：新一代 flow/rectified-flow 路线，采样速度和质量都在提升。

学习建议：

- 不要只比较「哪张图好看」，要比较 prompt 遵循、细节稳定性、手部/文字/复杂关系和速度。
- 学会区分 base model、checkpoint、LoRA、VAE、scheduler 的责任边界。

## 4. 快速采样：LCM、Turbo 与蒸馏

unit4 已介绍渐进蒸馏。现代快速生成常见路线包括：

- Progressive Distillation
- Consistency Models
- Latent Consistency Models
- Adversarial Diffusion Distillation
- Rectified Flow / Flow Matching 的少步采样

实践中要关注：

- 少步模型速度快，但细节与可控性可能下降。
- 蒸馏模型通常有推荐的 scheduler、steps 和 guidance scale。
- 不是所有 LoRA/ControlNet 都能无缝适配少步模型。

## 5. Transformer 与 Flow：DiT、Flow Matching

传统扩散图像模型以 UNet 为主。近年趋势是：

- 用 Transformer 替代 UNet，如 DiT。
- 在 latent patch/token 上建模。
- 用 Flow Matching 或 Rectified Flow 学习从噪声到数据的连续变换。

概念连接：

- DDPM 学习逐步反向扩散。
- Flow Matching 学习连续向量场。
- DiT 更像把图像 token 序列交给 Transformer 处理，扩展性强。

建议先把 unit1、unit3 的 diffusion/scheduler/latent 理解扎实，再进入这些方向。

## 6. 视频生成

视频生成不是简单地逐帧图像生成。关键难点是：

- 时间一致性：主体不能每帧变脸或漂移。
- 运动建模：需要理解相机运动、物体运动和场景变化。
- 显存成本：视频 token 数远大于单张图。
- 数据质量：视频 caption 与镜头切分影响很大。

主流方向：

- **AnimateDiff**：在图像模型上加入 motion module。
- **Stable Video Diffusion**：图像到视频扩展。
- **CogVideoX / HunyuanVideo 等**：更大规模的文生视频模型。
- **Video ControlNet / pose-to-video**：更强动作与结构控制。

建议实践顺序：

1. 先做 image-to-video，因为输入图像固定主体和风格。
2. 再做 text-to-video。
3. 最后尝试带姿态、深度或轨迹控制的视频生成。

## 7. 评估生成质量

扩散模型项目不应只看单张 cherry-pick。建议从以下维度评估：

- Prompt 遵循度
- 视觉质量与伪影
- 多样性
- 可控性
- 推理速度
- 显存占用
- 失败模式
- 安全与版权风险

对于学习项目，建议每次实验固定 seed 并记录参数，保留失败样例。失败样例通常比成功样例更能帮助理解模型。

## 8. 建议的后续扩展目录

```text
extensions/
├── README.md
├── lora_finetuning.ipynb
├── controlnet_basics.ipynb
├── sdxl_and_fast_sampling.ipynb
├── video_generation_intro.ipynb
└── flow_matching_minimal.md
```

本轮先完成公开发布基础与学习路线，后续可按上述顺序逐步新增 notebook。
