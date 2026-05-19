# 单元 1：扩散模型入门

欢迎学习 Hugging Face 扩散模型课程第一单元！在本单元中，你将了解扩散模型的基本原理，并使用 🤗 Diffusers 库创建自己的扩散模型。

## 开始本单元 🚀

建议步骤：

- 通过 [课程订阅表单](https://huggingface.us17.list-manage.com/subscribe?u=7f57e683fa28b51bfc493d048&id=ef963b4162) 订阅，以便在新内容发布时收到通知。
- 阅读下方入门材料，并浏览感兴趣的延伸阅读。
- 打开下方的 **《Diffusers 入门》** 笔记本，用 🤗 Diffusers 将理论付诸实践。
- 使用笔记本或配套训练脚本训练并分享你自己的扩散模型。
- （可选）若希望了解最小化从零实现及设计取舍，可学习 **《从零实现扩散模型》** 笔记本。
- （可选）观看 [本单元概览视频](https://www.youtube.com/watch?v=09o5cv6u76c)（非正式讲解）。

📢 别忘了加入 [Discord](https://huggingface.co/join/discord)，在 `#diffusion-models-class` 频道讨论资料并分享作品。

## 什么是扩散模型？

扩散模型是「生成模型」家族中较新的一员。生成建模的目标是：给定若干训练样本，学习**生成**与训练数据相似但非简单复制的新数据（如图像或音频）。优秀的生成模型应产出**多样**且像训练分布的输出。扩散模型如何做到？下面以图像生成为例说明。

<p align="center">
    <img src="https://user-images.githubusercontent.com/10695622/174349667-04e9e485-793b-429a-affe-096e8199ad5b.png" width="800"/>
    <br>
    <em> 图片来自 DDPM 论文（https://arxiv.org/abs/2006.11239）。 </em>
<p>

扩散模型成功的关键在于**迭代**去噪过程：生成从随机噪声开始，经多步逐步细化，最终得到输出图像。每一步，模型估计如何从当前输入走向「完全去噪」的版本。由于每步只做小改动，早期（预测最终输出极难时）的估计误差可在后续步骤中纠正。

训练相对直观，大致循环以下步骤：

1. 从训练集加载图像  
2. 以不同强度添加噪声（模型需对极噪与接近清晰的输入都能较好去噪）  
3. 将加噪输入送入模型  
4. 评估去噪效果  
5. 据此更新模型权重  

推理时，从纯随机输入出发，反复经模型小步更新。实践中还有多种采样方法，力求用更少步数生成高质量图像。

单元 1 的笔记本会逐步演示上述过程。单元 2 将介绍通过条件（如类别标签）与引导（guidance）控制输出；单元 3、4 将探索 Stable Diffusion 等强大模型。

## 动手笔记本

| 章节 | 本地笔记本 |
|:-----|:-----------|
| Diffusers 入门 | [01_introduction_to_diffusers.ipynb](01_introduction_to_diffusers.ipynb) |
| 从零实现扩散模型 | [02_diffusion_models_from_scratch.ipynb](02_diffusion_models_from_scratch.ipynb) |

也可在 [官方仓库](https://github.com/huggingface/diffusion-models-class/tree/main/unit1) 通过 Colab / Kaggle / Gradient / Studio Lab 运行英文原版。

**《Diffusers 入门》** 使用 diffusers 库展示训练与采样的完整流程，你将学会在自选数据上创建、训练并采样扩散模型，并能阅读、修改示例训练脚本。笔记本还介绍本单元的主要练习：共同探索不同规模下的「训练配方」。

**《从零实现扩散模型》** 用尽量精简的 PyTorch 从零实现加噪、建模、训练与采样，再与 diffusers 版本对比，理解各组件与设计决策，便于阅读新实现时快速抓住要点。

## 项目时间

掌握基础后，尝试训练一个或多个扩散模型！《Diffusers 入门》笔记本末尾有一些建议。请把结果、训练配方与发现分享给社区，一起摸索最佳训练方式。

## 延伸阅读

- [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) — DDPM 理论与代码的深度导读  
- Hugging Face 文档：[无条件图像生成训练](https://huggingface.co/docs/diffusers/training/unconditional_training)  
- AI Coffee Break：扩散模型 — https://www.youtube.com/watch?v=344w5h24-h8  
- Yannic Kilcher：DDPM — https://www.youtube.com/watch?v=W-O7AZNzbzQ  

发现更多优质资源？欢迎反馈，我们会补充到列表中。

## 进阶理解：扩散模型与其他生成模型

| 模型族 | 生成方式 | 优势 | 常见限制 |
|--------|----------|------|----------|
| VAE | 学习压缩潜变量并解码 | 训练稳定、潜空间可解释 | 样本可能偏模糊 |
| GAN | 生成器与判别器对抗训练 | 单步生成快、图像锐利 | 训练不稳定、模式崩塌 |
| Autoregressive | 按顺序预测 token/像素 | 概率建模清晰 | 高分辨率采样慢 |
| Diffusion | 从噪声逐步去噪 | 训练稳定、质量高、可控性强 | 多步采样成本较高 |

学习本单元时，重点不是追求最漂亮的生成结果，而是看懂三件事：噪声如何加入、模型预测什么、scheduler 如何把预测转成下一步样本。

## 采样器速览

- **DDPM**：经典随机采样，适合理解扩散模型训练目标。
- **DDIM**：可确定性采样，步数更少，并支持后续的 inversion 思路。
- **DPM-Solver / UniPC / Euler 系列**：现代文生图常用的快速采样器，通常在 Stable Diffusion 生态中使用。
- **LCM / Turbo**：蒸馏或一致性训练后的少步采样路线，适合实时预览，但要使用对应模型与推荐参数。

后续学习 Stable Diffusion 时，建议固定 prompt 与 seed，只改变 scheduler、steps、CFG scale，对比它们如何影响细节、构图与速度。
