# 单元 3：Stable Diffusion

欢迎学习 Hugging Face 扩散模型课程第三单元！你将认识强大的扩散模型 Stable Diffusion（SD），并探索其能力边界。

## 开始本单元 🚀

建议步骤：

- 通过 [课程订阅表单](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) 订阅更新。
- 阅读下方本单元核心概念。
- 打开 [**Stable Diffusion 入门** 笔记本](#动手笔记本)，实践常见用法。
- 使用 [**hackathon** 目录](../hackathon/index.md) 中的 **DreamBooth** 笔记本微调自己的 SD 模型并分享（黑客松已结束，仍可学习与实践）。
- （可选）观看 [_Stable Diffusion 深度剖析_ 视频](https://www.youtube.com/watch?app=desktop&v=0_BBRNYInx8) 及 [配套笔记本](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb)（Fast.ai 课程 [*Stable Diffusion from the Foundations*](https://www.fast.ai/posts/part2-2022.html) 的补充材料）。

📢 别忘了加入 [Discord](https://huggingface.co/join/discord)，在 `#diffusion-models-class` 频道交流。

## 简介

![Stable Diffusion 生成示例](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/sd_demo_images.jpg)  
_使用 Stable Diffusion 生成的示例图像_

Stable Diffusion 是强大的**文本条件潜空间扩散模型**（下文会解释这些术语）。它能根据文本描述生成惊人图像，因而在互联网上广受欢迎。本单元将解析其原理并展示更多用法。

## 潜空间扩散（Latent Diffusion）

图像越大，计算开销越大，尤其在自注意力中，计算量随输入数量二次增长：128×128 相对 64×64 像素多 4 倍，自注意力层内存与计算约多 16 倍。这对高分辨率生成是难题。

![潜空间扩散示意图](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/latent_diffusion_diagram.png)  
_示意图来自 [Latent Diffusion 论文](http://arxiv.org/abs/2112.10752)_

**潜空间扩散**用变分自编码器（VAE）将图像**压缩**到更小的空间尺寸。图像含大量冗余，VAE 可学习用小得多的**潜表示**重建高保真图像。SD 所用 VAE 将 3 通道图像编码为 4 通道潜变量，空间各维约缩小 8 倍：512×512 输入 → 4×64×64 潜变量。

在潜表示而非全分辨率像素上做扩散，可获得小图训练的诸多好处（更低显存、更浅 UNet、更快生成），最后再解码为高分辨率结果，显著降低训练与推理成本。

## 文本条件（Text Conditioning）

单元 2 介绍了向 UNet 注入额外信息以控制生成，即**条件生成**。给定加噪图像，模型需结合**文本描述**等线索预测去噪结果。推理时，我们输入期望画面的描述与纯噪声起点，模型尽力将噪声「去噪」成与描述一致的图像。

![文本编码流程](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/text_encoder.png)  
_文本经编码器变为 UNet 可用的嵌入（encoder_hidden_states）_

需要把文本转为能刻画语义的数值表示。SD 使用基于 **CLIP** 的预训练 Transformer 文本编码器。提示词先分词，再经 CLIP 文本编码器得到每个 token 的向量（SD 1.x 为 768 维，SD 2.x 为 1024 维）。为统一长度，提示词会填充/截断为 77 个 token，故条件张量形状约为每提示 77×1024。

![UNet 中的条件注入](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/unet.png)

如何将条件送入 UNet？答案是**交叉注意力**：UNet 各层含交叉注意力，空间位置可「关注」文本 token，在预测时融入提示信息；与时间步条件等一起在多尺度注入。

## 无分类器引导（Classifier-Free Guidance, CFG）

即便文本条件设计得很强，模型在预测时仍常更依赖带噪图像而非提示——许多 caption 与图像仅弱相关，模型学会了「不要太信描述」。推理时若完全不跟提示，生成结果可能与描述无关。

![不同 CFG 尺度](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/cfg_example.jpeg)  
_提示词 "An oil painting of a collie in a top hat"，CFG 尺度 0、1、2、10（从左到右）_

**无分类器引导（CFG）** 的做法：训练时有时将文本条件置空，迫使模型学会无条件去噪；推理时做两次预测——有提示与无提示——用两者之差、按**引导尺度**放大，使结果更朝文本条件预测方向偏移。尺度越大，通常越贴合描述（过高可能过饱和或失真）。

## 其他条件类型：超分、修复与深度

可为 SD 增加其他条件，例如：

- [**Depth-to-Image**](https://huggingface.co/stabilityai/stable-diffusion-2-depth)：额外输入深度信息，推理时传入目标深度图（可由单独模型估计），生成结构相似的图像。  
- **超分辨率**：以低分辨率图为条件生成高分辨率（如 [SD Upscaler](https://huggingface.co/stabilityai/stable-diffusion-x4-upscaler)）。  
- **Inpainting**：传入掩码，指定需重绘区域，非掩码区域保持不变。

## 使用 DreamBooth 微调

![DreamBooth 示意图](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/hackathon/dreambooth_teaser.jpg)  
_图片来自 [DreamBooth 项目页](https://dreambooth.github.io/)（基于 Imagen）_

**DreamBooth** 用于微调文生图模型以学习新概念（特定物体或风格），最初针对 Imagen，很快适配 [Stable Diffusion](https://huggingface.co/docs/diffusers/training/dreambooth)。效果可非常出色（许多 AI 头像即由此类服务产生），但对超参敏感；请参阅 [hackathon 笔记本](../hackathon/dreambooth.md) 与 [HF 博客调查](https://huggingface.co/blog/dreambooth)。

## 动手笔记本

| 章节 | 本地笔记本 |
|:-----|:-----------|
| Stable Diffusion 入门 | [stable-diffusion-introduction.md](stable-diffusion-introduction.md) |
| DreamBooth 黑客松 | [../hackathon/dreambooth.md](../hackathon/dreambooth.md) |
| Stable Diffusion 深度剖析（可选，英文） | [Fast.ai 笔记本](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) |

DreamBooth 需要较大算力；在 Colab / Kaggle 请使用 **GPU** 运行时。

**Stable Diffusion 入门** 用 🤗 Diffusers 简要演示管道生成与修改图像。

**DreamBooth 黑客松笔记本** 展示如何在自己的图像上微调 SD，学习新风格或概念。

**深度剖析** 笔记本与视频逐步拆解典型生成管道，并讨论各阶段的可改造点。

## 项目时间

可跟随 **DreamBooth** 笔记本在指定主题上训练模型；黑客松已结束，仍可练习并上传到 Hub。详情见 [hackathon/README.md](../hackathon/index.md)。

## 延伸阅读

- [High-Resolution Image Synthesis with Latent Diffusion Models](http://arxiv.org/abs/2112.10752) — SD 背后的潜扩散方法  
- [CLIP](https://openai.com/blog/clip/) — 连接图文；SD 使用 CLIP 文本编码器。另见 [OpenCLIP 介绍](https://wandb.ai/johnowhitaker/openclip-benchmarking/reports/Exploring-OpenCLIP--VmlldzoyOTIzNzIz)（SD 2 使用开源 CLIP 变体之一）  
- [GLIDE](https://arxiv.org/abs/2112.10741) — 早期文本条件与 CFG 工作  

发现更多资源？欢迎反馈补充。

## Stable Diffusion 版本脉络

| 版本 | 特点 | 学习重点 |
|------|------|----------|
| SD 1.x | 生态最成熟，示例最多 | 理解基础 pipeline、LoRA、ControlNet 生态 |
| SD 2.x | 使用 OpenCLIP，部分模型加入 depth 等能力 | 关注文本编码器变化与模型兼容性 |
| SDXL | 更大模型、更高分辨率、更强提示词理解 | 理解 base/refiner、双文本编码器与高分辨率工作流 |
| 快速模型 | Turbo、Lightning、LCM 等 | 理解蒸馏、少步采样与质量/速度权衡 |

本单元使用 SD 1.x/2.x 体系讲清核心组件。掌握 VAE、UNet、text encoder、scheduler 的职责后，再迁移到 SDXL 或其他现代模型会更自然。

## 常见 Pipeline 的关系

- **text-to-image**：只输入文本，从随机 latent 开始生成。
- **image-to-image**：输入图像先加噪，再按新 prompt 去噪，适合风格迁移和变体生成。
- **inpainting**：额外输入 mask，只重绘指定区域。
- **depth-to-image**：额外输入深度图，保持大体结构。
- **ControlNet / IP-Adapter**：不是本单元原始内容，但可以理解为更强的条件输入机制。

实际项目中，prompt 通常只负责语义方向；构图、姿态、边缘、深度、参考主体等更稳定的控制，应交给 ControlNet、IP-Adapter 或后续视频/3D 条件方法。

## 安全与版权提醒

Stable Diffusion 能生成高度逼真的图像。公开项目中应避免引导生成违法、有害、侵犯隐私或冒充真实人物的内容。训练个性化模型时，优先使用你拥有权利的数据；涉及人脸、商标、艺术家风格或商业素材时，需要额外谨慎。
