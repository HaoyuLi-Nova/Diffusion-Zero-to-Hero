# 学习路线

这份路线面向希望系统掌握图像生成、音频生成、视频生成与扩散模型工程实践的学习者。建议先跑通最小示例，再逐步进入 Stable Diffusion 与现代控制方法。

## 路线总览

| 阶段 | 内容 | 建议投入 | 产出 |
|------|------|----------|------|
| 0. 准备 | Python、PyTorch、GPU 环境、Hugging Face Hub | 0.5 天 | 能运行 notebook、下载模型 |
| 1. 扩散基础 | unit1 | 1-2 天 | 能解释加噪、去噪、采样循环 |
| 2. 训练与控制 | unit2 | 2-3 天 | 能微调小模型，并理解 guidance/conditioning |
| 3. Stable Diffusion | unit3 | 2-4 天 | 能拆解 text-to-image、img2img、inpainting |
| 4. 个性化微调 | hackathon | 2-5 天 | 能训练 DreamBooth 或理解其局限 |
| 5. 进阶扩展 | unit4 | 2-4 天 | 理解 DDIM inversion、音频扩散、蒸馏与新架构 |
| 6. 现代主题 | [modern-diffusion-roadmap.md](modern-diffusion-roadmap.md) | 持续 | 跟进 LoRA、ControlNet、SDXL、视频生成 |

## 如果你是初学者

1. 先读 [unit0](unit0/index.md)，确认前置知识。
2. 跑通 [unit1/01_introduction_to_diffusers.ipynb](unit1/01-introduction-to-diffusers.md)，重点理解 scheduler 如何加噪与去噪。
3. 再读 [unit1/02_diffusion_models_from_scratch.ipynb](unit1/02-diffusion-models-from-scratch.md)，不要纠结模型效果，关注训练循环。
4. 学 [unit2](unit2/index.md) 时先理解「模型预测什么」和「guidance 如何改变采样方向」。
5. 进入 [unit3](unit3/index.md) 前，确保你能解释 UNet、VAE、文本编码器分别负责什么。

## 如果你主要关心图像生成应用

推荐路径：

1. unit1 的 Diffusers 入门
2. unit3 的 Stable Diffusion 入门
3. unit4 的 DDIM 反演
4. hackathon 的 DreamBooth
5. 现代扩展路线中的 LoRA、ControlNet、IP-Adapter、SDXL

实践建议：

- 先学会用 pipeline 生成、修改、保存图像。
- 再学习 prompt、negative prompt、CFG scale、scheduler、seed 对结果的影响。
- 最后再进入个性化微调和结构控制。

## 如果你主要关心研究理解

推荐路径：

1. unit1 从零实现
2. unit2 条件生成示例
3. unit3 的 latent diffusion 与 cross-attention
4. unit4 的蒸馏、DDIM inversion、DiT/MaskGIT
5. 阅读现代路线中的 Flow Matching、Rectified Flow、DiT

建议配套阅读：

- DDPM: Denoising Diffusion Probabilistic Models
- DDIM: Denoising Diffusion Implicit Models
- Latent Diffusion Models
- Classifier-Free Diffusion Guidance
- Scalable Diffusion Models with Transformers

## 如果你主要关心视频生成

本仓库当前视频部分以 unit4 理论导读为主，尚未加入完整视频生成 notebook。建议先完成：

1. unit3：理解 Stable Diffusion 的潜空间与文本条件。
2. unit4：阅读视频生成概览。
3. modern roadmap：学习 AnimateDiff、Stable Video Diffusion、CogVideoX、时序一致性与显存优化。

视频生成比图像生成更吃显存与存储，建议从短视频、低分辨率、小 batch 开始。

## 每学完一个单元应能回答的问题

- unit1：扩散模型为什么从噪声开始？scheduler 的作用是什么？
- unit2：guidance 与 conditioning 有什么区别？什么时候需要微调？
- unit3：Stable Diffusion 为什么在 latent 中扩散？CFG scale 为什么会影响提示词遵循度？
- unit4：DDIM inversion 如何支持图像编辑？音频为什么可以转成频谱图来生成？
- hackathon：DreamBooth 为什么容易过拟合？低显存时有哪些替代方案？

## 学习节奏建议

不要追求一次跑通所有 notebook。扩散模型学习更像调试一个系统：每次只改变一个变量，记录 prompt、seed、scheduler、steps、CFG scale 和显存占用。形成自己的实验日志，比单纯看生成结果更重要。
