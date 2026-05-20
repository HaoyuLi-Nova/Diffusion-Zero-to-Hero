# 单元 4：扩散模型进阶

欢迎学习 Hugging Face 扩散模型课程第四单元！本单元将概览扩散模型在最新研究中的诸多改进与扩展。相比前几单元，**代码比重更低**，旨在为你后续自学提供路线图。

## 开始本单元 🚀

建议步骤：

- 通过 [课程订阅表单](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) 订阅，以便新课程发布时收到通知。  
- 阅读下方各主题概览。  
- 通过链接中的视频与论文深入感兴趣的方向。  
- 学习演示笔记本，并阅读文末「接下来做什么」获取项目建议。

📢 别忘了加入 [Discord](https://huggingface.co/join/discord)，在 `#diffusion-models-class` 频道讨论与分享作品。

## 目录

- [单元 4：扩散模型进阶](#单元-4扩散模型进阶)
  - [开始本单元 🚀](#开始本单元-)
  - [目录](#目录)
  - [通过蒸馏加速采样](#通过蒸馏加速采样)
  - [训练方面的改进](#训练方面的改进)
  - [更强的生成与编辑控制](#更强的生成与编辑控制)
  - [视频](#视频)
  - [音频](#音频)
  - [新架构与新思路——走向「迭代精炼」](#新架构与新思路走向迭代精炼)
  - [动手笔记本](#动手笔记本)
  - [接下来做什么？](#接下来做什么)

## 通过蒸馏加速采样

**渐进蒸馏（Progressive distillation）** 利用已有扩散模型训练**推理步数更少**的新版本：学生模型从教师权重初始化；训练中教师走两步采样，学生用一步逼近该预测。可反复进行，上一轮学生成为下一轮教师。结果通常只需 **4 或 8 步** 即可得到尚可的样本（远少于原教师）。核心机制见 [提出该思想的论文](http://arxiv.org/abs/2202.00512)：

![渐进蒸馏示意图](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/progressive_distillation.png)

_渐进蒸馏示意图（来自 [论文](http://arxiv.org/abs/2202.00512)）_

该「教师教学生」思路可扩展到**引导蒸馏**：教师使用无分类器引导，学生根据额外输入（目标引导尺度）在单步内复现等效输出，进一步减少高质量采样所需的模型前向次数。[此视频](https://www.youtube.com/watch?v=ZXuK6IRJlnk) 有概览。

注：可使用蒸馏版 Stable Diffusion，见 [Diffusers 文档](https://huggingface.co/docs/diffusers/main/en/using-diffusers/distilled_sd)。

主要参考文献：

- [Progressive Distillation For Fast Sampling Of Diffusion Models](http://arxiv.org/abs/2202.00512)  
- [On Distillation Of Guided Diffusion Models](http://arxiv.org/abs/2210.03142)  

## 训练方面的改进

近年来出现了多种提升扩散模型训练的技巧；本节概括近期论文中的核心思想。研究仍在快速演进，若有值得补充的论文欢迎反馈。

![ERNIE-ViLG 2.0 论文图 2](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/ernie_vilg_mode.png)  
_图片来自 [ERNIE-ViLG 2.0 论文](http://arxiv.org/abs/2210.15257)_

主要方向包括：

- 调节噪声日程、损失加权与采样轨迹以提高训练效率。Karras 等人的 [Elucidating the Design Space of Diffusion-Based Generative Models](http://arxiv.org/abs/2206.00364) 对设计空间有精彩分析。  
- 在多种宽高比上训练，见 [课程发布会视频](https://www.youtube.com/watch?v=g6tIUrMvOec)。  
- **级联扩散**：先低分辨率模型，再一个或多个超分模型（DALL·E 2、Imagen 等）。  
- **更强条件**：丰富文本嵌入（[Imagen](https://arxiv.org/abs/2205.11487) 使用 T5 大模型）或多种条件组合（[eDiffi](http://arxiv.org/abs/2211.01324)）。  
- **知识增强**：在训练中引入预训练图像描述、目标检测等，生成更信息化 caption（[ERNIE-ViLG 2.0](http://arxiv.org/abs/2210.15257)）。  
- **混合去噪专家（MoDE）**：为不同噪声水平训练不同「专家」，见上图。

主要参考文献：

- [Elucidating the Design Space of Diffusion-Based Generative Models](http://arxiv.org/abs/2206.00364)  
- [eDiffi: Text-to-Image Diffusion Models with an Ensemble of Expert Denoisers](http://arxiv.org/abs/2211.01324)  
- [ERNIE-ViLG 2.0](http://arxiv.org/abs/2210.15257)  
- [Imagen](https://arxiv.org/abs/2205.11487)（[演示站](https://imagen.research.google/)）  

## 更强的生成与编辑控制

除训练改进外，采样与推理阶段也有大量创新，可为现有扩散模型增加新能力。

![eDiffi「用文字作画」示例](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/ediffi_paint_with_words.png)  
_[eDiffi](http://arxiv.org/abs/2211.01324) 的 paint-with-words 生成结果_

视频 [《用扩散模型编辑图像》](https://www.youtube.com/watch?v=zcG7tG3xS3s) 概览了主要方法，大致可分为四类：

1. **加噪后用新提示去噪** — `img2img` 管道的核心，并有多种扩展：  
   - [SDEdit](https://sde-image-editing.github.io/)、[MagicMix](https://magicmix.github.io/)  
   - **DDIM 反演**：沿采样轨迹「反向」而非随机加噪，控制更精细 — 见本仓库 [01-ddim-inversion.md](01-ddim-inversion.md)  
   - [Null-text Inversion](https://null-text-inversion.github.io/)：逐步优化 CFG 的无条件文本嵌入，实现高质量文本编辑  

2. **在 (1) 基础上用掩码限定编辑区域**：  
   - [Blended Diffusion](https://omriavrahami.com/blended-diffusion-page/)  
   - [基于 CLIPSeg 的文本 inpainting 演示](https://huggingface.co/spaces/nielsr/text-based-inpainting)  
   - [DiffEdit](https://arxiv.org/abs/2210.11427)：用扩散模型自身生成编辑掩码  
   - [SmartBrush](https://arxiv.org/abs/2212.05034)：针对掩码引导 inpainting 的微调  

3. **交叉注意力控制**：利用 UNet 交叉注意力控制编辑空间位置：  
   - [Prompt-to-Prompt](https://arxiv.org/abs/2208.01626)；[应用于 SD 的实践](https://wandb.ai/wandb/cross-attention-control/reports/Improving-Generative-Images-with-Instructions-Prompt-to-Prompt-Image-Editing-with-Cross-Attention-Control--VmlldzoyNjk2MDAy)  
   - eDiffi 的 paint-with-words（见上图）  

4. **在单张图像上微调（过拟合）后生成**：  
   - [Imagic](https://arxiv.org/abs/2210.09276)  
   - [UniTune](https://arxiv.org/abs/2210.09477)  

[InstructPix2Pix](https://arxiv.org/abs/2211.09800) 用上述编辑技术构建合成数据对（编辑指令由 GPT-3.5 生成），训练能按自然语言指令编辑图像的模型。

## 视频

![Imagen Video 示例帧](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/imagen_video_frames.png)  
_[Imagen Video 示例视频](https://imagen.research.google/video/) 的静帧_

视频可视为图像序列，扩散核心思想可直接应用。近期工作关注合适架构（如对整个序列操作的 3D UNet）与高效处理视频数据。高帧率数据量远大于静态图，常见流程是先低分辨率、低帧率生成，再经时空超分得到最终高质量视频。

主要参考文献：

- [Video Diffusion Models](https://video-diffusion.github.io/)  
- [Imagen Video 论文](https://imagen.research.google/video/paper.pdf)  

## 音频

![Riffusion 频谱图](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/riffusion_spectrogram.png)  
_由 Riffusion 生成的频谱图（[来源](https://www.riffusion.com/about)）_

虽有直接在波形上做扩散的工作（如 [DiffWave](https://arxiv.org/abs/2009.09761)），目前较成功的是将音频转为**频谱图**（spectrogram），当作 2D「图像」训练常见图像扩散模型，再转回音频。[Riffusion](https://www.riffusion.com/) 在 Stable Diffusion 上微调以文本条件生成频谱图 — [在线体验](https://www.riffusion.com/)。

音频生成领域发展极快；撰写英文原版 README 时一周内即有大量新工作（下列带 * 者）：

主要参考文献：

- [DiffWave](https://arxiv.org/abs/2009.09761)  
- [Riffusion](https://www.riffusion.com/about)（[代码](https://github.com/riffusion/riffusion)）  
- *[MusicLM](https://google-research.github.io/seanet/musiclm/examples/)  
- *[RAVE2](https://github.com/acids-ircam/RAVE)；*[AudioLDM](https://twitter.com/LiuHaohe/status/1619119637660327936?s=20&t=jMkPWBFuAH19HI9m5Sklmg)  
- *[Noise2Music](https://noise2music.github.io/)  
- *[Make-An-Audio](https://text-to-audio.github.io/)  
- *[Moûsai](https://arxiv.org/abs/2301.11757)  

## 新架构与新思路——走向「迭代精炼」

![Cold Diffusion 论文图 1](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/cold_diffusion.png)  
_图片来自 [Cold Diffusion](http://arxiv.org/abs/2208.09392) 论文_

我们正逐渐超越狭义「扩散」模型，走向更一般的**迭代精炼**类模型：某种退化（如前向扩散中的高斯加噪）被逐步逆转以生成样本。[Cold Diffusion](http://arxiv.org/abs/2208.09392) 表明多种非高斯退化也可迭代「撤销」；基于 Transformer 的方法则用 token 替换或掩码作为加噪策略。

![MaskGIT 流程](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/maskgit_pipeline.png)  
_[MaskGIT](http://arxiv.org/abs/2202.04200) 流程_

许多模型中的 UNet 正被 Transformer 等架构替代：[DiT](https://www.wpeebles.com/DiT) 用 Transformer 替换 UNet；[Recurrent Interface Networks](https://arxiv.org/pdf/2212.11972.pdf) 追求更高效率；[MaskGIT](http://arxiv.org/abs/2202.04200)、[MUSE](http://arxiv.org/abs/2301.00704) 在图像 token 上工作；[Paella](https://arxiv.org/abs/2211.07292v1) 表明 UNet 在 token  regime 下同样有效。

新论文不断涌现，迭代精炼任务的上限仍待探索，值得继续跟进！

主要参考文献：

- [Cold Diffusion](http://arxiv.org/abs/2208.09392)  
- [Scalable Diffusion Models with Transformers (DiT)](https://www.wpeebles.com/DiT)  
- [MaskGIT](http://arxiv.org/abs/2202.04200)  
- [Muse](http://arxiv.org/abs/2301.00704)  
- [Paella](https://arxiv.org/abs/2211.07292v1)  
- [Recurrent Interface Networks](https://arxiv.org/pdf/2212.11972.pdf)；另见 [simple diffusion](https://arxiv.org/abs/2301.11093)（高分辨率训练中的噪声日程）  

## 动手笔记本

| 章节 | 本地笔记本 |
|:-----|:-----------|
| DDIM 反演 | [01-ddim-inversion.md](01-ddim-inversion.md) |
| 音频扩散 | [02-diffusion-for-audio.md](02-diffusion-for-audio.md) |

也可在 [官方仓库](https://github.com/huggingface/diffusion-models-class/tree/main/unit4) 通过 Colab / Kaggle 等运行英文环境。

本单元涵盖大量方向，许多都值得单独成课。目前可通过两个笔记本动手：

- **DDIM 反演**：用反演技术基于现有扩散模型编辑图像。  
- **音频扩散**：介绍频谱图，并演示在特定音乐流派上微调音频扩散模型的最小示例。  

## 接下来做什么？

截至目前，这是本课程的最后一个单元——之后的路由你自行探索！欢迎在 Hugging Face [Discord](https://huggingface.co/join/discord) 提问并分享作品，期待看到你的创作 🤗。

## 现代扩展建议

如果你已经完成本单元，下一步建议沿四条主线继续学习：

| 方向 | 为什么重要 | 建议主题 |
|------|------------|----------|
| 个性化微调 | 真实项目常需要学习特定主体或风格 | LoRA、DreamBooth LoRA、Textual Inversion |
| 结构控制 | prompt 很难稳定控制姿态、边缘和构图 | ControlNet、T2I-Adapter、IP-Adapter |
| 快速生成 | 交互式创作需要低延迟 | LCM、Turbo、Lightning、蒸馏模型 |
| 视频生成 | 扩散模型的重要应用方向 | AnimateDiff、Stable Video Diffusion、CogVideoX、时序一致性 |

从研究角度，建议继续关注 **DiT**、**Flow Matching**、**Rectified Flow** 与多模态视频模型。它们与本单元的「迭代精炼」思想一脉相承，但在架构和训练目标上更接近 2024 年之后的主流路线。更完整的后续路线见 [modern-diffusion-roadmap.md](../modern-diffusion-roadmap.md)。
