# 扩散模型术语表

本术语表采用「中文（English）」形式，方便阅读论文、Diffusers 文档和课程 notebook。

## 基础概念

| 中文 | English | 简要说明 |
|------|---------|----------|
| 扩散模型 | Diffusion Model | 通过逐步加噪训练模型，再从噪声逐步去噪生成样本的生成模型 |
| 前向过程 | Forward Process | 训练时逐步向真实数据添加噪声的过程 |
| 反向过程 | Reverse Process | 推理时从噪声逐步去噪得到样本的过程 |
| 去噪 | Denoising | 预测并移除输入中的噪声 |
| 时间步 | Timestep | 扩散过程中的离散步骤，通常代表噪声强度 |
| 噪声日程 | Noise Schedule | 每个时间步添加多少噪声的规则 |
| 采样器/调度器 | Sampler / Scheduler | 控制反向去噪更新方式的组件 |
| 训练目标 | Training Objective | 常见为预测噪声、预测干净样本或预测 velocity |

## 常见模型与组件

| 中文 | English | 简要说明 |
|------|---------|----------|
| UNet | UNet | 扩散模型中常见的去噪网络，具备多尺度编码-解码结构 |
| 变分自编码器 | Variational Autoencoder, VAE | Stable Diffusion 中用于图像与 latent 互相转换 |
| 潜空间 | Latent Space | 压缩后的表示空间，计算成本低于像素空间 |
| 文本编码器 | Text Encoder | 将 prompt 转为向量条件的模型，如 CLIP text encoder |
| 交叉注意力 | Cross-Attention | 让图像特征关注文本 token 的机制 |
| 管道 | Pipeline | Diffusers 中把 tokenizer、text encoder、UNet、VAE、scheduler 组合起来的高层接口 |

## 生成控制

| 中文 | English | 简要说明 |
|------|---------|----------|
| 条件生成 | Conditional Generation | 给模型额外条件，如类别、文本、深度图、边缘图 |
| 引导 | Guidance | 在采样过程中调整预测方向，使结果更符合目标 |
| 无分类器引导 | Classifier-Free Guidance, CFG | 使用有条件与无条件预测的差值强化提示词方向 |
| 引导尺度 | Guidance Scale | CFG 的放大系数，过高可能导致过饱和或失真 |
| 负向提示词 | Negative Prompt | 告诉模型避免出现的内容或风格 |
| 图生图 | Image-to-Image, img2img | 以已有图像为起点，通过加噪和去噪生成变体 |
| 图像修复 | Inpainting | 使用 mask 指定需要重绘的区域 |
| 深度条件生成 | Depth-to-Image | 用深度图控制图像结构 |

## 训练与微调

| 中文 | English | 简要说明 |
|------|---------|----------|
| 微调 | Fine-tuning | 在新数据上继续训练已有模型 |
| 全量微调 | Full Fine-tuning | 更新模型全部或大部分参数，成本高但表达力强 |
| DreamBooth | DreamBooth | 用少量图片让模型学习特定主体或风格 |
| 文本反演 | Textual Inversion | 学习新的 token embedding 表示一个概念 |
| 低秩适配 | LoRA | 用少量低秩参数适配模型，显存与存储成本低 |
| 参数高效微调 | PEFT | 只训练少量新增参数的微调方法集合 |
| 过拟合 | Overfitting | 模型记住训练样本，泛化或可控性变差 |

## 进阶采样与加速

| 中文 | English | 简要说明 |
|------|---------|----------|
| DDPM | Denoising Diffusion Probabilistic Models | 经典随机扩散采样框架 |
| DDIM | Denoising Diffusion Implicit Models | 可确定性采样，常用于更快采样和 inversion |
| 反演 | Inversion | 将真实图像映射回扩散采样轨迹或 latent 噪声 |
| 渐进蒸馏 | Progressive Distillation | 用教师模型训练少步数学生模型 |
| 一致性模型 | Consistency Model | 学习跨噪声水平的一致映射以加速生成 |
| LCM | Latent Consistency Model | latent 空间的一致性模型，常用于少步生成 |
| Turbo 模型 | Turbo Model | 经蒸馏或特殊训练得到的快速生成模型 |

## 现代扩展

| 中文 | English | 简要说明 |
|------|---------|----------|
| ControlNet | ControlNet | 用边缘、深度、姿态等结构条件控制生成 |
| IP-Adapter | IP-Adapter | 使用参考图像作为图像提示，控制主体或风格 |
| SDXL | Stable Diffusion XL | 更大规模、更强文本理解与高分辨率能力的 SD 版本 |
| DiT | Diffusion Transformer | 使用 Transformer 替代 UNet 的扩散架构 |
| Flow Matching | Flow Matching | 学习从噪声到数据分布的连续流场 |
| Rectified Flow | Rectified Flow | 通过更直的传输路径提升生成效率的一类方法 |
| 视频扩散 | Video Diffusion | 生成帧序列，并关注时间一致性与运动建模 |
| 时序一致性 | Temporal Consistency | 视频中相邻帧内容、结构和风格保持连贯 |

## 常见缩写

| 缩写 | 全称 |
|------|------|
| SD | Stable Diffusion |
| CFG | Classifier-Free Guidance |
| VAE | Variational Autoencoder |
| CLIP | Contrastive Language-Image Pretraining |
| LDM | Latent Diffusion Model |
| LoRA | Low-Rank Adaptation |
| PEFT | Parameter-Efficient Fine-Tuning |
| SVD | Stable Video Diffusion |
