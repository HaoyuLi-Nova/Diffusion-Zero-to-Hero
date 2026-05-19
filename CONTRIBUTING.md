# 贡献指南

感谢你愿意改进 `Diffusion Zero to Hero 中文实战课`。这个仓库的目标是：在尊重上游课程的基础上，为中文读者提供可运行、可维护、可持续更新的扩散模型学习资料。

## 可以贡献什么

- 修正翻译、术语、错别字和链接
- 补齐 `images/` 资源或修复图片引用
- 改进 notebook 的中文注释与运行说明
- 补充环境配置、显存优化、常见错误排查
- 新增现代扩散模型教程，如 LoRA、ControlNet、SDXL、视频生成、Flow Matching
- 同步上游 `huggingface/diffusion-models-class` 的新增内容

## 翻译与写作规范

- 术语尽量保持中英并列，首次出现时写作「中文（English）」。
- 保留论文、模型、库名和函数名的英文原名，例如 `StableDiffusionPipeline`、`DDIMScheduler`。
- 面向学习者解释「为什么」和「什么时候用」，不要只逐句翻译英文原文。
- 如果内容来自上游课程、论文或博客，请保留原链接与图注来源。
- 大段新内容尽量放在 `docs/` 或对应单元 README，避免打断 notebook 的教学节奏。

## Notebook 规范

- 提交前清理输出、执行计数和 widget 状态。
- 不要提交本地运行产生的大图片、模型权重、checkpoint、W&B 日志或私有数据。
- 保留可复现的安装说明；需要 GPU 的地方写明显存建议。
- 不要硬编码 token。使用 `notebook_login()` 或 `huggingface-cli login`。

## 安全要求

提交前请确认：

- 没有 `.env`、`HF_TOKEN`、`hf_...`、W&B key 或其他密钥。
- 没有个人路径、私有数据集路径、内网地址或不可公开的模型地址。
- 没有大体积模型文件（如 `.ckpt`、`.safetensors`、`.pt`）。

可用下面的命令做基础扫描：

```bash
rg -i "hf_[A-Za-z0-9]{20,}|HF_TOKEN|password|secret|token|/home/" .
```

## 与上游同步

本仓库基于 <https://github.com/huggingface/diffusion-models-class>。同步上游时建议：

1. 先记录上游变更来源（commit、PR 或文件路径）。
2. 保留原作者署名、论文链接、许可证说明。
3. 将英文更新翻译成本仓库统一风格。
4. 在 `CHANGELOG.md` 中记录同步内容。

## 提交 PR 前检查

- [ ] 文档链接可点击，路径大小写正确
- [ ] notebook 已清理输出和 widget metadata
- [ ] 没有提交密钥、私有数据或大模型权重
- [ ] 新增图片放在 `images/`，并保留来源说明
- [ ] 新增依赖已写入 `requirements.txt` 或对应文档
- [ ] 涉及上游内容时已保留来源链接与许可证说明
- [ ] 不要手动编辑 `README.md` 中 `<!-- contributors:start -->` 与 `<!-- contributors:end -->` 之间的内容

合并到主分支后，GitHub Actions 会根据贡献记录自动更新 README 中的贡献者列表。

若希望头像和用户名正确显示，请在本机配置与 GitHub 账号关联的 git 信息，例如：

```bash
git config user.name "HaoyuLi-Nova"
git config user.email "你的 GitHub 已验证邮箱或 noreply 邮箱"
```

可在 <https://github.com/settings/emails> 查看可用邮箱。
