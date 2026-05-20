# How to Learn Diffusion Models

This course is organized as a hands-on path: read the documentation page first, then run the matching notebook when you are ready to experiment.

## 1. Start with the Core Loop

Begin with [Unit 1](unit1/index.md). The key ideas are:

- forward diffusion adds noise to clean data;
- the model learns to predict denoising information;
- the scheduler turns model predictions into the next sample;
- sampling starts from noise and iteratively refines it.

Use [Diffusers 入门](unit1/01-introduction-to-diffusers.md) to see the library workflow, then read [从零实现扩散模型](unit1/02-diffusion-models-from-scratch.md) to understand the minimal PyTorch version.

## 2. Learn Control and Adaptation

Move to [Unit 2](unit2/index.md) after you understand DDPM-style training and sampling.

- [微调与引导](unit2/01-finetuning-and-guidance.md) shows how to adapt a pretrained model and modify the sampling loop with guidance.
- [类别条件扩散模型](unit2/02-class-conditioned-diffusion-model.md) shows a small class-conditioned MNIST model.

The practical distinction is simple: fine-tuning changes the model weights, guidance changes the sampling behavior, and conditioning trains the model to use extra inputs.

## 3. Decompose Stable Diffusion

Read [Stable Diffusion 入门](unit3/stable-diffusion-introduction.md) when you are ready for text-to-image systems.

Focus on the component roles:

- VAE compresses images into latent space and decodes latents back to pixels;
- tokenizer and text encoder turn prompts into conditioning embeddings;
- UNet predicts denoising information in latent space;
- scheduler controls the sampling trajectory;
- CFG compares conditional and unconditional predictions to strengthen prompt following.

## 4. Explore Editing and Other Modalities

[Unit 4](unit4/index.md) extends the course beyond basic image generation.

- [DDIM 反演](unit4/01-ddim-inversion.md) explains how inversion supports image editing.
- [音频扩散](unit4/02-diffusion-for-audio.md) treats spectrograms as images and converts the result back to audio.

After this, use [现代扩散模型路线图](modern-diffusion-roadmap.md) to continue toward LoRA, ControlNet, SDXL, DiT, Flow Matching and video generation.

## 5. Build a Personalization Project

Use [DreamBooth](hackathon/dreambooth.md) as a project-style exercise. Keep a small, clean image set, record training parameters, and compare both successful and failed generations. In real projects, consider LoRA-style DreamBooth after understanding the full fine-tuning version.
