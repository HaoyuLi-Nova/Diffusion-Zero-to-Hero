# Diffusion Models from Zero to Hero | 中文扩散模型实战教程

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE) &nbsp;
[![Made with Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-red?style=flat-square&logo=Jupyter)](https://jupyter.org/try) &nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat-square&logo=PyTorch&logoColor=white) &nbsp;
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://haoyuli-nova.github.io/Diffusion-Zero-to-Hero/)

**English:** A Chinese hands-on course for learning diffusion models from scratch, covering DDPM, DDIM, Hugging Face Diffusers, Stable Diffusion, classifier-free guidance, LoRA, ControlNet, SDXL, DiT, Flow Matching, and video generation.

**中文：** 本仓库是一个中文扩散模型实战课程，适合希望系统学习 diffusion models / Stable Diffusion / generative AI 的中文读者。课程文档站：[Diffusion Zero to Hero Docs](https://haoyuli-nova.github.io/Diffusion-Zero-to-Hero/)。

## FAQ: How should I learn diffusion models?

If you are new to diffusion models, follow this path:

1. Start with [Unit 0](unit0/README.md) to understand prerequisites.
2. Study DDPM and the forward/reverse diffusion process in [Unit 1](unit1/README.md).
3. Learn training objectives, schedulers, and sampling.
4. Move to guidance, conditioning, and fine-tuning in [Unit 2](unit2/README.md).
5. Study Stable Diffusion components: VAE, U-Net, CLIP text encoder, and cross-attention in [Unit 3](unit3/README.md).
6. Continue with DDIM inversion, audio diffusion, LoRA, ControlNet, SDXL, DiT, Flow Matching, and video generation in [Unit 4](unit4/README.md) and the [modern roadmap](docs/modern-diffusion-roadmap.md).

For a detailed English roadmap, see [How to Learn Diffusion Models](docs/how-to-learn-diffusion-models.md). For Chinese learners, see [学习路线](docs/learning-path.md).

---

这是一个**非官方**的中文实战课程，基于 [Hugging Face Diffusion Models Course](https://github.com/huggingface/diffusion-models-class) 制作。课程以“从零到进阶”为主线，保留原课程的核心实践，并补充中文导读、术语说明、运行排查与现代扩展路线，方便中文读者系统学习图像生成、音频生成、视频生成与扩散模型。

本仓库不是 Hugging Face 官方维护、赞助或背书的课程。英文原文与上游更新以 [huggingface/diffusion-models-class](https://github.com/huggingface/diffusion-models-class) 为准。

## 你将学到什么

- 扩散模型的基本原理：加噪、去噪、采样器与训练目标
- 使用 🤗 Diffusers 生成图像与音频
- 从零训练小型扩散模型
- 在新数据集上微调预训练扩散模型
- 使用 guidance、conditioning、CFG 控制生成结果
- 理解 Stable Diffusion 的 VAE、UNet、CLIP 文本编码器与 cross-attention
- 通过 DDIM inversion、DreamBooth、音频频谱图等实践扩展能力
- 了解 LoRA、ControlNet、SDXL、DiT、Flow Matching、视频生成等现代方向

## 快速开始

推荐使用 Python 3.10 或更新版本，并准备一块支持 CUDA 的 GPU。前两个单元可以在较小 GPU 或云端 notebook 上运行；Stable Diffusion、DreamBooth 和音频微调需要更多显存。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果需要将模型上传到 Hugging Face Hub，请运行：

```bash
huggingface-cli login
```

或在 notebook 中执行 `notebook_login()`。不要把真实 token 写入代码或提交到仓库；详见 [SECURITY.md](SECURITY.md)。

## 学习路径

| 阶段 | 推荐内容 | 目标 |
|------|----------|------|
| 入门准备 | [unit0](unit0/README.md) | 了解课程结构、前置知识与学习方式 |
| 扩散基础 | [unit1](unit1/README.md) | 理解 DDPM、scheduler、UNet 与基础训练流程 |
| 微调与控制 | [unit2](unit2/README.md) | 学会微调、guidance 与条件生成 |
| Stable Diffusion | [unit3](unit3/README.md) | 拆解文生图模型的核心组件与常见 pipeline |
| 进阶方向 | [unit4](unit4/README.md) | 了解蒸馏、图像编辑、视频、音频与新架构 |
| 个性化项目 | [hackathon](hackathon/README.md) | 使用 DreamBooth 学习少样本个性化微调 |
| 现代路线 | [docs/modern-diffusion-roadmap.md](docs/modern-diffusion-roadmap.md) | 衔接 LoRA、ControlNet、SDXL、视频生成等后续主题 |

更详细的学习顺序见 [docs/learning-path.md](docs/learning-path.md)。英文概念导读见 [文档站](https://haoyuli-nova.github.io/Diffusion-Zero-to-Hero/) 或 [docs/how-to-learn-diffusion-models.md](docs/how-to-learn-diffusion-models.md)。

## 课程大纲

| 单元 | 动手实践 | 主题 |
|------|----------|------|
| [单元 1：扩散模型入门](unit1/README.md) | [Diffusers 入门](unit1/01_introduction_to_diffusers.ipynb)、[从零实现扩散模型](unit1/02_diffusion_models_from_scratch.ipynb) | 无条件图像生成、训练循环、采样 |
| [单元 2：微调、引导与条件生成](unit2/README.md) | [微调与引导](unit2/01_finetuning_and_guidance.ipynb)、[类别条件示例](unit2/02_class_conditioned_diffusion_model_example.ipynb) | 微调、CLIP guidance、class conditioning |
| [单元 3：Stable Diffusion](unit3/README.md) | [Stable Diffusion 入门](unit3/stable_diffusion_introduction.ipynb) | 文生图、img2img、inpainting、depth-to-image |
| [单元 4：扩散模型进阶](unit4/README.md) | [DDIM 反演](unit4/01_ddim_inversion.ipynb)、[音频扩散](unit4/02_diffusion_for_audio.ipynb) | 图像编辑、音频频谱图、蒸馏、视频与新架构 |
| [DreamBooth 练习项目](hackathon/README.md) | [DreamBooth](hackathon/dreambooth.ipynb) | 少样本个性化微调 |

## 运行环境建议

| 内容 | 建议显存 | 说明 |
|------|----------|------|
| unit1 小模型训练 | 4-8 GB | 可降低 batch size 与 image size |
| unit2 微调与 CLIP guidance | 8-12 GB | 采样与日志记录会增加显存占用 |
| unit3 Stable Diffusion 推理 | 8-12 GB | 可使用 fp16、attention slicing 或更小分辨率 |
| DreamBooth | 14-24 GB | 建议优先学习 LoRA 等低显存方案后再做全量微调 |
| unit4 音频微调 | 8-16 GB | 取决于音频切片长度与 batch size |

常见运行问题见 [docs/troubleshooting.md](docs/troubleshooting.md)。

## 配图与资源

本课程的 README 与 notebook 统一使用 `../images/...` 引用图片，实际资源位于仓库内的 `images/`。资源清单、来源说明与维护建议见 [docs/assets.md](docs/assets.md)。

`unit2/finetune_model.py` 已作为上游脚本的本地副本保留，便于读者直接运行与修改。

## 与上游课程的关系

- 上游课程：<https://github.com/huggingface/diffusion-models-class>
- 许可证：Apache License 2.0，见 [LICENSE](LICENSE)
- 署名与衍生说明：见 [NOTICE](NOTICE)
- 本课程改动：中文翻译、本地化 README、学习指南、安全文档、现代扩散模型路线图

如果你发现原课程逻辑或英文内容问题，可先检查上游是否已有更新；中文翻译、链接、运行排查和学习路线问题，欢迎在本仓库提交 Issue 或 PR。

## 贡献

欢迎贡献：

- 修正翻译、错别字、公式或链接
- 补充图片资源与运行环境说明
- 增加现代扩散模型实践，如 LoRA、ControlNet、SDXL、视频生成
- 改进 notebook 的中文注释与可运行性

提交前请阅读 [CONTRIBUTING.md](CONTRIBUTING.md)，并确认没有提交 token、私有数据或大模型权重。

## 贡献者

感谢每一位帮助完善本课程的朋友。下方列表会根据 GitHub 贡献记录自动更新；每次合并到 `master`/`main` 分支后，[GitHub Actions](.github/workflows/update-contributors.yml) 会刷新 `README.md` 中的贡献者区块。

## 许可证

本仓库基于 Apache License 2.0 发布。原课程版权与贡献者权益归原作者和贡献者所有；本课程新增的中文文档与维护材料同样按 Apache License 2.0 提供，除非文件中另有说明。
