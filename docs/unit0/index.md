# Hugging Face 扩散模型课程

在这门免费课程中，你将：

- 👩‍🎓 学习扩散模型背后的理论
- 🧨 使用流行的 🤗 Diffusers 库生成图像与音频
- 🏋️‍♂️ 从零开始训练自己的扩散模型
- 📻 在新数据集上微调现有扩散模型
- 🗺 探索条件生成与引导（guidance）
- 🧑‍🔬 构建自定义扩散模型管道（pipeline）

## 前置要求

本课程需要较好的 Python 水平，以及深度学习与 PyTorch 的基础。若尚未具备，可参考：

- Python：https://www.udacity.com/course/introduction-to-python--ud1110
- PyTorch 深度学习入门：https://www.udacity.com/course/deep-learning-pytorch--ud188
- PyTorch 60 分钟入门：https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html

若要将模型上传到 Hugging Face Hub，需要注册账号（免费）：https://huggingface.co/join

## 课程大纲

课程共四个单元。每单元包含理论部分（README 与论文/资源链接）以及两个 *notebook*：

- **单元 1：扩散模型入门** — 🤗 Diffusers 介绍与从零实现
- **单元 2：微调与引导** — 在新数据上微调扩散模型并添加引导
- **单元 3：Stable Diffusion** — 探索强大的文本条件潜空间扩散模型
- **单元 4：扩散模型进阶** — 将扩散推向更高阶的技术

各单元中文导读与笔记本见本仓库对应目录：`unit1/` … `unit4/`。

## 课程作者

[**Jonathan Whitaker**](https://huggingface.co/johnowhitaker) 是 [answer.ai](https://www.answer.ai/) 的数据科学家/AI 研究员，热衷于教学与课程设计，目前专注于生成式 AI 等多模态方向。个人主页：[johnowhitaker.dev](https://johnowhitaker.dev/)。

[**Lewis Tunstall**](https://huggingface.co/lewtun) 是 Hugging Face 的机器学习工程师，致力于开发开源工具并让其易于社区使用；亦是 O’Reilly 著作 [*Natural Language Processing with Transformers*](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) 的合著者。

## 常见问题

- **完成课程是否有证书？**  
  目前本课程不提供证书。Hugging Face 生态系统的认证计划仍在筹备中，敬请关注。

- **每单元大约需要多少时间？**  
  设计上每章约 1 周、每周约 6–8 小时。你可按自己的节奏完成。

- **有问题在哪里提问？**  
  在课程页面点击「*Ask a question*」，或加入 [Hugging Face Discord](https://discord.com/invite/JfAtkvEtRb)，在 `#diffusion-models-class` 频道提问。

- **课程代码在哪里？**  
  本仓库各单元目录下的 `.ipynb` 即为动手笔记本；也可在页面顶部通过 Colab 等入口运行（参见官方仓库说明）。

- **如何为课程做贡献？**  
  中文翻译、链接、运行问题和扩展资料请优先在本仓库提交 Issue 或 PR，并参考 [贡献指南](https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/CONTRIBUTING.md)。如果是上游英文课程本身的 bug，可再向 [diffusion-models-class](https://github.com/huggingface/diffusion-models-class) 反馈。

- **能否复用本课程内容？**  
  可以。课程采用宽松的 [Apache 2.0 许可证](https://www.apache.org/licenses/LICENSE-2.0.html)。使用时需注明出处、提供许可证链接，并说明是否修改。引用示例 BibTeX：

```
@misc{huggingfacecourse,
  author = {Hugging Face},
  title = {The Hugging Face Diffusion Models Course, 2022},
  howpublished = "\url{https://huggingface.co/course}",
  year = {2022},
  note = "[Online; accessed <today>]"
}
```

## 开始学习

准备好后，请前往 [单元 1](../unit1/index.md) 开始课程。

## 中文学习版说明

`Diffusion Zero to Hero 中文实战课` 是基于上游课程的非官方中文学习版，除翻译外还会逐步补充术语表、故障排查和现代扩散模型路线。建议先按原课程顺序学习，再阅读 [学习路线](../learning-path.md) 与 [现代扩散模型路线图](../modern-diffusion-roadmap.md)。
