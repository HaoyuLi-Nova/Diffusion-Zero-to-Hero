# Diffusion Models from Zero to Hero

这是一个面向中文读者的扩散模型实战课程，内容从基础 DDPM 与 Diffusers 入门开始，逐步进入微调、引导、条件生成、Stable Diffusion、DDIM 反演、音频扩散、视频生成和 DreamBooth 个性化微调。

文档站点的目标是把仓库中的 Jupyter Notebook 转成更适合 GitHub Pages 阅读的课程页面。每个页面都保留了原 notebook 的讲解顺序与代码单元，读者可以先在网页上通读概念，再回到 notebook 中动手运行。

## 推荐学习顺序

| 阶段 | 内容 | 目标 |
| --- | --- | --- |
| Unit 0 | [课程准备](unit0/index.md) | 熟悉课程结构、运行环境与前置知识 |
| Unit 1 | [Diffusers 入门](unit1/01-introduction-to-diffusers.md)、[从零实现扩散模型](unit1/02-diffusion-models-from-scratch.md) | 理解加噪、去噪、scheduler、UNet 与基础训练流程 |
| Unit 2 | [微调与引导](unit2/01-finetuning-and-guidance.md)、[类别条件扩散模型](unit2/02-class-conditioned-diffusion-model.md) | 学会微调、guidance、CLIP guidance 与 class conditioning |
| Unit 3 | [Stable Diffusion 入门](unit3/stable-diffusion-introduction.md) | 拆解 VAE、text encoder、UNet、scheduler 与常见 pipeline |
| Unit 4 | [DDIM 反演](unit4/01-ddim-inversion.md)、[音频扩散](unit4/02-diffusion-for-audio.md) | 学习图像编辑、反演、频谱图音频生成与进阶方向 |
| Unit 5 | [视频生成从零开始](unit5/index.md) | 理解视频张量、时序一致性、toy video diffusion 与后续 I2V/T2V 路线 |
| Hackathon | [DreamBooth](hackathon/dreambooth.md) | 用少量图像微调 Stable Diffusion，完成个性化生成练习 |

## 运行方式

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

在 Windows PowerShell 中可改用：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Stable Diffusion、DreamBooth 和音频微调通常需要 GPU。若只想阅读概念，直接浏览文档即可；若要复现实验，请优先打开对应 notebook。

## 文档维护

Notebook 对应的 Markdown 页面由 `scripts/notebook_to_docs.py` 生成。修改 notebook 后可运行：

```bash
python scripts/notebook_to_docs.py
```

这会同步 `docs/unit*/` 与 `docs/hackathon/` 下的课程页面。文档中的展示图片统一引用仓库根目录 `images/` 中的资源，避免在 `docs/` 中维护重复图片。
