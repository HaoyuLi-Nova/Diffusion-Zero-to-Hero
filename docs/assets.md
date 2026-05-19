# Diffusion Zero to Hero 图片与资源清单

本课程的 README 与 notebook 使用相对路径 `../images/...` 引用配图。实际资源目录为仓库内的 `images/`。

如果图片缺失，README 会出现破图；部分 notebook 还会因为读取本地图片而报错。

## 需要的目录

```text
images/
├── hackathon/
├── unit1/
├── unit2/
├── unit3/
└── unit4/
```

## 资源清单

### unit1

- `images/unit1/diffusers_library.jpg`
- `images/unit1/huggingface_token_settings.png`
- `images/unit1/unet_model.png`
- `images/unit1/unet_diag.png`

### unit2

- `images/unit2/finetune_sample_generations.png`
- `images/unit2/gradio_demo.png`

### unit3

- `images/unit3/sd_demo_images.jpg`
- `images/unit3/latent_diffusion_diagram.png`
- `images/unit3/text_encoder.png`
- `images/unit3/unet.png`
- `images/unit3/cfg_example.jpeg`
- `images/unit3/image.png`
- `images/unit3/mask.png`
- `images/unit3/vae_diagram.png`
- `images/unit3/inpaint_from_scratch.png`
- `images/unit3/inpaint_w_border.jpg`

### unit4

- `images/unit4/progressive_distillation.png`
- `images/unit4/ernie_vilg_mode.png`
- `images/unit4/ediffi_paint_with_words.png`
- `images/unit4/imagen_video_frames.png`
- `images/unit4/riffusion_spectrogram.png`
- `images/unit4/cold_diffusion.png`
- `images/unit4/maskgit_pipeline.png`
- `images/unit4/ddim_sampling_timesteps.png`
- `images/unit4/torchaudio_feature_extractions.png`

### hackathon

- `images/hackathon/dreambooth_teaser.jpg`
- `images/hackathon/dreambooth_high_level.png`
- `images/hackathon/dreambooth_novel_views.png`
- `images/hackathon/dreambooth_property_modification.png`

## 来源说明

这些图片主要来自上游课程、论文图、Hugging Face 文档资源或原论文/项目页面。同步图片时应保留原图注和来源链接；如果图片来自论文或第三方项目，不应移除其 attribution。

## 发布建议

- 小体积教学图片可以直接提交到 `images/`。
- 大体积数据、模型权重和生成输出不要提交到 Git。
- 如果未来改为 CDN 或 Git LFS，请同步更新 README 与 notebook 中的路径。
