<!-- This page is generated from the matching notebook by scripts/notebook_to_docs.py. -->

> 原始 Notebook：[unit3/stable_diffusion_introduction.ipynb](https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/unit3/stable_diffusion_introduction.ipynb)

# 简介

本笔记本将介绍如何使用现有管道（Pipeline）通过 Stable Diffusion 创建和修改图像的基础知识。我们还将简要介绍管道中的核心组件，更深入的探索将在深度解析笔记本中进行。具体来说，我们将涵盖以下内容：
- 使用 `StableDiffusionPipeline` 从文本生成图像，并尝试调整可用参数
- 了解管道中部分核心组件的实际运行机制
    - 使模型成为「潜空间扩散模型」的 VAE（变分自编码器）
    - 处理文本提示词的分词器（Tokenizer）和文本编码器
    - UNet 网络本身
    - 调度器（Scheduler），以及探索不同类型的调度器
- 使用管道组件复现采样循环
- 使用 Img2Img 管道编辑现有图像
- 使用图像修复（Inpainting）和 Depth2Img 管道

❓如有任何问题，请在 Hugging Face Discord 服务器的 `#diffusion-models-class` 频道中提问。如果你还没有注册，可以通过以下链接加入：https://huggingface.co/join/discord

# 环境准备

```python
# %pip install -Uq diffusers ftfy accelerate
# Installing transformers from source for now since we need the latest version for Depth2Img
# %pip install -Uq git+https://github.com/huggingface/transformers
import os
from dotenv import load_dotenv
load_dotenv("../.env", override=True)
```

```python
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline, 
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline, 
    StableDiffusionDepth2ImgPipeline
    )       

# We'll use a couple of demo images later in the notebook
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")

# load images for inpainting example
img_path="../images/unit3/image.png"
mask_path="../images/unit3/mask.png"
img_height = 512
img_width = 512

init_image = Image.open(img_path).convert("RGB").resize((img_height, img_width))
mask_image = Image.open(mask_path).convert("RGB").resize((img_height, img_width))
# Set device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
```

# 从文本生成图像

让我们加载一个 Stable Diffusion 管道（Pipeline），看看它能实现什么功能。Stable Diffusion 有多个不同版本，在撰写本文时最新版本为 2.1。如果你想探索旧版本，只需将模型 ID 替换为对应的模型即可（例如，你可以尝试 `CompVis/stable-diffusion-v1-4`，或者从 [DreamBooth 概念库](https://huggingface.co/sd-dreambooth-library) 中选择一个模型）。

```python
# Load the pipeline
model_id = "sd2-community/stable-diffusion-2-1-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to(device)
```

如果你的 GPU 显存不足，可以通过以下几种方法降低显存占用：
- 加载 FP16（半精度）版本（并非所有系统都支持）。使用该版本时，如果你单独调试管道的各个组件，可能还需要将张量转换为 `torch.float16` 类型：

  `pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16).to(device)`

- 启用注意力切片（attention slicing）功能。该功能会略微降低运行速度，但能有效减少 GPU 显存占用：

 `pipe.enable_attention_slicing()`
- 降低生成图像的分辨率

管道加载完成后，我们可以使用以下代码根据提示词生成图像：

```python
# Set up a generator for reproducibility
generator = torch.Generator(device=device).manual_seed(42)

# Run the pipeline, showing some of the available arguments
pipe_output = pipe(
    prompt="Palette knife painting of an autumn cityscape", # What to generate
    negative_prompt="Oversaturated, blurry, low quality", # What NOT to generate
    height=480, width=640,     # Specify the image size
    guidance_scale=8,          # How strongly to follow the prompt
    num_inference_steps=35,    # How many steps to take
    generator=generator        # Fixed random seed
)

# View the resulting image
pipe_output.images[0]
```

**练习：** 试着运行上方代码块，自定义提示词并调整各项参数，观察参数变化对出图效果的影响。更换随机种子，或是直接删掉 `generator` 参数，就能每次生成不一样的图片。

重点可调参数说明：
- 宽高参数用于设定生成图像尺寸，数值必须是 **8 的倍数**，才能保证 VAE 模块正常运行（后续章节会详细讲解）。
- 采样步数影响出图质量，默认 50 步效果较佳；日常调试实验最少可降至 20 步，运行速度更快。
- 反向提示词会在无分类器引导（CFG）流程中生效，是精准控图的实用手段。该参数可省略，不过多数使用者都会像示例一样，在反向提示词里填写不想出现的画面特征。
- `guidance_scale` 参数用来控制无分类器引导（CFG）强度。数值越高，生成画面越贴合提示词；但数值过高容易导致画面色彩过浓、观感变差。

想要寻找优质提示词灵感，可以参考 [Stable Diffusion Prompt Book](https://stability.ai/sdv2-prompt-book)。

下方代码块可以直观看出调高引导强度带来的画面变化：

```python
#@markdown comparing guidance scales:
cfg_scales = [1.1, 8, 12] #@param
prompt = "A collie with a pink hat" #@param
fig, axs = plt.subplots(1, len(cfg_scales), figsize=(16, 5))
for i, ax in enumerate(axs):
  im = pipe(prompt, height=480, width=480,
    guidance_scale=cfg_scales[i], num_inference_steps=35,
    generator=torch.Generator(device=device).manual_seed(42)).images[0]
  ax.imshow(im); ax.set_title(f'CFG Scale {cfg_scales[i]}');
```

自行调整上述数值，尝试不同的引导强度与提示词。审美效果因人而异，不过在我看来，**8 到 12** 这个区间的参数效果，优于过高或过低的数值。

# 流水线组件

我们当前使用的 `StableDiffusionPipeline`，比之前章节学过的 `DDPMPipeline` 结构更为复杂。除了 UNet 与调度器之外，该管道还集成了诸多其他核心组件：

```python
print(list(pipe.components.keys())) # List components
```

为了更好地理解整个管道的运行原理，我们先来逐一拆解并实操各个组件，之后再将它们整合起来，亲手复现整套管道的运行逻辑。

### VAE（变分自编码器）

![VAE 示意图](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/vae_diagram.png)

VAE（变分自编码器）是一类模型，它能够将输入数据编码为压缩表征，再把这种「潜空间」表征还原成与原始输入高度相似的数据。使用 Stable Diffusion 生成图像时，我们先在 VAE 的**潜空间**中执行扩散过程**生成潜变量**，最后再**解码**得到最终成品图像。

以下代码可读取一张输入图像，借助 VAE 将其编码为潜空间特征，再重新解码还原图像：

```python
# Create some fake data (a random image, range (-1, 1))
images = torch.rand(1, 3, 512, 512).to(device) * 2 - 1 
print("Input images shape:", images.shape)

# Encode to latent space
with torch.no_grad():
  latents = 0.18215 * pipe.vae.encode(images).latent_dist.mean
print("Encoded latents shape:", latents.shape)

# Decode again
with torch.no_grad():
  decoded_images = pipe.vae.decode(latents / 0.18215).sample
print("Decoded images shape:", decoded_images.shape)
```

如你所见，一张 512×512 的图像会被压缩为 64×64 的潜空间表征（包含 4 个通道）。**每个空间维度都进行了 8 倍压缩**，这就是为什么生成图像的宽高必须是 8 的倍数。

处理这些信息密度极高的 4×64×64 潜变量，比直接处理 512 像素的大尺寸图像效率高得多，这使得扩散模型的训练和推理速度更快，资源消耗也更低。VAE 的解码过程并非完美无缺，但它的效果足够好，因此这点微小的质量损失通常是值得的。

**注意：** 上述代码示例中包含了 0.18215 这个缩放因子，这是为了与 Stable Diffusion 训练时的预处理方式保持一致。

### 分词器与文本编码器

![文本编码器示意图](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/text_encoder.png)

文本编码器的作用是将输入字符串（即提示词）转换为数值表征，作为条件输入传给 UNet。文本首先会通过管道的分词器（Tokenizer）转换为一系列词元（Token）。文本编码器的词汇表约包含 5 万个词元——任何不在词汇表中的单词都会被拆分为更小的子词（Sub-word）。

这些词元随后会输入文本编码器本身——这是一个 Transformer 模型，最初是作为 CLIP 的文本编码器训练的。我们期望这个预训练的 Transformer 模型已经学习到了丰富的文本表征，这些表征同样能为扩散任务提供有效支持。

让我们通过编码一个示例提示词来测试这个过程：先手动分词并输入文本编码器，再使用管道的 `encode_prompt` 方法展示完整流程，包括将文本长度填充/截断至 77 个词元的最大限制：

```python
# Tokenizing and encoding an example prompt manually

# Tokenize
input_ids = pipe.tokenizer(["A painting of a flooble"])['input_ids']
print("Input ID -> decoded token")
for input_id in input_ids[0]:
  print(f"{input_id} -> {pipe.tokenizer.decode(input_id)}")

# Feed through CLIP text encoder
input_ids = torch.tensor(input_ids).to(device)
with torch.no_grad():
  text_embeddings = pipe.text_encoder(input_ids)['last_hidden_state']
print("Text embeddings shape:", text_embeddings.shape)
```

```python
# Get the final text embeddings using the pipeline's encode_prompt function
text_embeddings = pipe._encode_prompt("A painting of a flooble", device, 1, True, '')
text_embeddings.shape
```

这些文本嵌入（也就是文本编码器模型中最后一个 Transformer 块的所谓「隐藏状态」）会作为 `forward` 方法的额外参数输入到 UNet 中，我们将在下一节中看到具体实现。

### UNet 网络

![UNet 示意图](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/unet.png)

UNet 接收带噪声的输入并预测噪声，这和我们在之前章节中见过的 UNet 原理一致。与之前的示例不同，这里的输入不是原始图像，而是图像的**潜空间表征**。除了时间步条件之外，这个 UNet 还会将提示词的**文本嵌入**作为额外输入。下面是它在一些模拟数据上进行预测的示例：

```python
# Dummy inputs
timestep = pipe.scheduler.timesteps[0]
latents = torch.randn(1, 4, 64, 64).to(device)
text_embeddings = torch.randn(1, 77, 1024).to(device)

# Model prediction
with torch.no_grad():
  unet_output = pipe.unet(latents, timestep, text_embeddings).sample
print('UNet output shape:', unet_output.shape) # Same shape as the input latents
```

### 调度器

调度器保存噪声日程，并根据模型预测把带噪样本更新为下一步的结果。默认调度器是 `PNDMScheduler`，你也可以使用其他调度器（例如 `LMSDiscreteScheduler`），只要它们以相同的配置初始化即可。

我们可以绘制噪声日程，查看噪声水平（基于 $\bar{\alpha}$）随时间步的变化：

```python
plt.plot(pipe.scheduler.alphas_cumprod, label=r'$\bar{\alpha}$')
plt.xlabel('Timestep (high noise to low noise ->)');
plt.title('Noise schedule');
plt.legend();
```

如果你想尝试不同的调度器，可以按如下方式替换：

```python
from diffusers import LMSDiscreteScheduler

# Replace the scheduler
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)

# Print the config
print('Scheduler config:', pipe.scheduler)

# Generate an image with this new scheduler
pipe(prompt="Palette knife painting of an winter cityscape", height=480, width=480,
     generator=torch.Generator(device=device).manual_seed(42)).images[0]
```

更多关于使用不同调度器的说明，请参阅[官方文档](https://huggingface.co/docs/diffusers/using-diffusers/schedulers)。

### 手写采样循环

在了解了上述各组件的运行方式之后，我们可以将它们组合起来，复现流水线的功能：

```python
guidance_scale = 8 #@param
num_inference_steps = 30 #@param
prompt = "Beautiful picture of a wave breaking" #@param
negative_prompt = "zoomed in, blurry, oversaturated, warped" #@param

# Encode the prompt
text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

# Create our random starting point
latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
latents *= pipe.scheduler.init_noise_sigma

# Prepare the scheduler
pipe.scheduler.set_timesteps(num_inference_steps, device=device)

# Loop through the sampling timesteps
for i, t in enumerate(pipe.scheduler.timesteps):

  # Expand the latents if we are doing classifier free guidance
  latent_model_input = torch.cat([latents] * 2)

  # Apply any scaling required by the scheduler
  latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

  # Predict the noise residual with the UNet
  with torch.no_grad():
    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

  # Perform guidance
  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
  noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

  # Compute the previous noisy sample x_t -> x_t-1
  latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# Decode the resulting latents into an image
with torch.no_grad():
  image = pipe.decode_latents(latents.detach())

# View
pipe.numpy_to_pil(image)[0]
```

在大多数情况下，直接使用现成的流水线会更方便；但拥有这段可修改的采样循环，对于理解并调整各组件的工作方式很有帮助。如果你想更深入地探索并修改这些代码与各组件，可以查看 [Stable Diffusion Deep Dive 笔记本](https://github.com/fastai/diffusion-nbs/blob/master/Stable%20Diffusion%20Deep%20Dive.ipynb) 和[配套视频](https://m.youtube.com/watch?v=0_BBRNYInx8)。

# 更多流水线

除了根据提示词生成图像之外，Stable Diffusion 还能做很多事！本节将演示几种有趣的流水线，让你初步了解 Stable Diffusion 的其他应用场景。其中部分示例需要下载新的模型；如果时间紧张，你可以只浏览现有输出，而不必亲自下载并运行所有模型。

## Img2Img（图生图）

在前面的示例中，我们从随机潜变量起步，通过完整的扩散采样循环从零生成图像。但我们不必每次都从零开始。Img2Img 流水线会先把已有图像编码为一组潜变量，再向潜变量中加入一定噪声并以此作为起点。加入的噪声量与去噪步数共同决定了 img2img 的「强度」（strength）。只加入少量噪声（低 strength）时，输出与原图差异很小；而加入最大噪声并运行完整去噪过程时，结果除整体结构相似外，几乎看不出原图痕迹。

为更好地理解 img2img 流程，我们先**手动实现**一遍。这有助于弄清潜变量如何编码、噪声如何加入以及扩散如何应用。完成手写实现后，我们再看看 diffusers 库提供的 `StableDiffusionImg2ImgPipeline` 如何更高效地实现相同功能。

下面演示如何手动实现 img2img 流水线：

### 手写 Img2Img 循环

```python
import numpy as np

# Encode init_image
init_image_tensor = torch.from_numpy(np.array(init_image).transpose(2, 0, 1)).float() / 255.0 # 0~255 => 0~1
init_image_tensor = 2.0 * init_image_tensor - 1.0 # 0~1 => -1~1
init_image_tensor = init_image_tensor.unsqueeze(0).to(device) # add batch dim.

with torch.no_grad():
    init_image_latents = pipe.vae.encode(init_image_tensor).latent_dist.sample() * pipe.vae.config.scaling_factor
```

```python
guidance_scale = 7.5 #@param
num_inference_steps = 30 #@param
strength = 0.6
prompt = "An oil painting of a man on a bench" #@param

# Encode the prompt
text_embeddings = pipe._encode_prompt(prompt, device, 1, True, '')

# Prepare the scheduler
pipe.scheduler.set_timesteps(num_inference_steps, device=device)

# Prepare latent variables
# We don't use all timesteps in the noise scheduler.
# Calculate a subset of timesteps based on `strength` to apply to the initial image.
init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
t_start = max(num_inference_steps - init_timestep, 0)
timesteps = pipe.scheduler.timesteps[t_start:]
# The first timestep of the new timesteps will be the starting point for adding noise to the initial image.
latent_timestep = timesteps[:1]

# Add noise to init_image_latents at the noise level specified by latent_timestep.
noise = torch.randn((1, 4, 64, 64), device=device, 
                    generator=torch.Generator(device=device).manual_seed(42))
latents = pipe.scheduler.add_noise(init_image_latents, noise, latent_timestep)

# Loop through the sampling timesteps
for i, t in enumerate(timesteps):

  # Expand the latents if we are doing classifier free guidance
  latent_model_input = torch.cat([latents] * 2)

  # Apply any scaling required by the scheduler
  latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

  # Predict the noise residual with the UNet
  with torch.no_grad():
    noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

  # Perform guidance
  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
  noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

  # Compute the previous noisy sample x_t -> x_t-1
  latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

# Decode latents
latents_norm = latents / pipe.vae.config.scaling_factor

with torch.no_grad():
    result_image = pipe.vae.decode(latents_norm).sample

result_image = (result_image / 2 + 0.5).clamp(0, 1).squeeze()
result_image = (result_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
result_image = Image.fromarray(result_image)

# View the result
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(result_image);axs[1].set_title('Result');
```

在手动实现 img2img 流程之后，让我们看看如何使用 diffusers 库提供的 `StableDiffusionImg2ImgPipeline` 更高效地获得相同结果。

该流水线不需要特殊模型；只要 `model_id` 与上文文生图示例相同，就无需下载新的权重文件。

### Img2Img 流水线

```python
# Loading an Img2Img pipeline
model_id = "sd2-community/stable-diffusion-2-1-base"
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id).to(device)
```

在「环境准备」一节中，我们加载了示例 `init_image` 供本演示使用；你也可以换成自己的图像。下面是流水线的实际运行效果：

```python
# Apply Img2Img
result_image = img2img_pipe(
    prompt="An oil painting of a man on a bench",
    image=init_image, # The starting image
    strength=0.6, # 0 for no change, 1.0 for max strength
).images[0]

# View the result
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(result_image);axs[1].set_title('Result');
```

**练习：** 动手试验这条流水线。尝试使用自己的图像，或调整不同的 strength 与提示词。你可以使用与文生图流水线相同的许多参数，因此不妨尝试不同的尺寸、步数等。

## Inpainting（局部重绘）

如果我们希望保留输入图像的某些部分，只在其他区域生成新内容，这就是「Inpainting（局部重绘）」。虽然可以用与前述演示相同的模型（通过 `StableDiffusionInpaintPipelineLegacy`）来实现，但使用专门微调的 Stable Diffusion 版本通常效果更好——它会同时接收带遮罩的图像和遮罩本身作为额外条件。遮罩图像应与输入图像尺寸相同：**白色**区域表示要替换的部分，**黑色**区域表示保持不变的部分。

为深入理解 inpainting 流程，我们先手动实现 `StableDiffusionInpaintPipelineLegacy` 背后的逻辑。这有助于从底层弄清 inpainting 的工作原理，以及 Stable Diffusion 如何处理输入。完成后，我们再对比专用微调流水线。下面演示如何手动实现 inpainting 流水线，并应用于「环境准备」一节中加载的示例图像与遮罩：

![手写 Inpainting 流程示意](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/inpaint_from_scratch.png)

### 手写 Inpainting 循环

```python
# Resize mask image
mask_image_latent_size = mask_image.resize((64,64))
mask_image_latent_size = torch.tensor( (np.array(mask_image_latent_size)[...,0] > 5).astype(np.float32) )
plt.imshow(mask_image_latent_size.numpy(), cmap='gray')

mask_image_latent_size = mask_image_latent_size.to(device)
mask_image_latent_size.shape
```

再次编写去噪循环。

```python
guidance_scale = 8 #@param
num_inference_steps = 30 #@param
prompt = "A small robot, high resolution, sitting on a park bench"
negative_prompt = "zoomed in, blurry, oversaturated, warped"
generator = torch.Generator(device=device).manual_seed(42)

# Encode the prompt
text_embeddings = pipe._encode_prompt(prompt, device, 1, True, negative_prompt)

# Create our random starting point
latents = torch.randn((1, 4, 64, 64), device=device, generator=generator)
latents *= pipe.scheduler.init_noise_sigma

# Prepare the scheduler
pipe.scheduler.set_timesteps(num_inference_steps, device=device)

for i, t in enumerate(pipe.scheduler.timesteps):
    # Expand the latents if we are doing classifier free guidance
    latent_model_input = torch.cat([latents] * 2)
    
    # Apply any scaling required by the scheduler
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Predict the noise residual with the UNet
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

    # Perform inpainting to fill in the masked areas
    if i < len(pipe.scheduler.timesteps)-1:
        # Add noise to the original image's latent at the previous timestep t-1
        noise = torch.randn(init_image_latents.shape, generator=generator, device=device, dtype=torch.float32)
        background = pipe.scheduler.add_noise(init_image_latents, noise, torch.tensor([pipe.scheduler.timesteps[i+1]])) 

        latents = latents*mask_image_latent_size # white in the areas
        background = background * (1-mask_image_latent_size) # black in the areas

        # Combine the generated and original image latents based on the mask
        latents += background

# Decode latents
latents_norm = latents / pipe.vae.config.scaling_factor

with torch.no_grad():
    inpainted_image = pipe.vae.decode(latents_norm).sample

inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).squeeze()
inpainted_image = (inpainted_image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
inpainted_image = Image.fromarray(inpainted_image)

inpainted_image
```

### Inpainting 流水线

在手动实现 inpainting 逻辑之后，让我们看看如何使用专为 inpainting 任务设计的微调流水线。下面演示如何加载该流水线，并应用于「环境准备」一节中加载的示例图像与遮罩：

![Inpainting 效果示例](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit3/inpaint_w_border.jpg)

```python
# Load the inpainting pipeline (requires a suitable inpainting model)
# pipe = StableDiffusionInpaintPipeline.from_pretrained("runwayml/stable-diffusion-inpainting")

# "runwayml/stable-diffusion-inpainting" is no longer available.
# Therefore, we are using the "stabilityai/stable-diffusion-2-inpainting" model instead.
pipe = StableDiffusionInpaintPipeline.from_pretrained("sd2-community/stable-diffusion-2-inpainting")
pipe = pipe.to(device)
```

```python
# Inpaint with a prompt for what we want the result to look like
prompt = "A small robot, high resolution, sitting on a park bench"
image = pipe(prompt=prompt, image=init_image, mask_image=mask_image).images[0]

# View the result
fig, axs = plt.subplots(1, 3, figsize=(16, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(mask_image);axs[1].set_title('Mask')
axs[2].imshow(image);axs[2].set_title('Result');
```

将 inpainting 与能自动生成遮罩的模型结合使用时，效果会格外强大。例如，[这个 Demo Space](https://huggingface.co/spaces/nielsr/text-based-inpainting) 使用名为 CLIPSeg 的模型，根据文本描述自动遮罩出需要替换的对象。

### 附注：管理模型缓存

探索不同的流水线与模型变体会占用大量磁盘空间。你可以用以下命令查看当前已下载的模型：

```bash
!ls ~/.cache/huggingface/hub/ # List the contents of the cache directory
```

请参阅[缓存相关文档](https://huggingface.co/docs/huggingface_hub/main/en/how-to-cache)，了解如何查看并有效管理缓存。

## Depth2Img（深度条件生成）

_输入图像、深度图与生成示例（图片来源：StabilityAI）_

Img2Img 很实用，但有时我们希望保留原图的构图，却完全改变颜色或纹理。要找到既能保留布局、又不保留原图色彩的 Img2Img strength 往往并不容易。

这时就需要另一个微调模型！该模型在生成时会额外注入深度信息作为条件。流水线内部使用深度估计模型生成深度图，再将其送入微调后的 UNet，从而有望在保留初始图像深度与结构的同时，填充全新的内容。

```python
# Load the Depth2Img pipeline (requires a suitable model)
pipe = StableDiffusionDepth2ImgPipeline.from_pretrained("sd2-community/stable-diffusion-2-depth")
pipe = pipe.to(device)
```

```python
# Inpaint with a prompt for what we want the result to look like
prompt = "An oil painting of a man on a bench"
image = pipe(prompt=prompt, image=init_image).images[0]

# View the result
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
axs[0].imshow(init_image);axs[0].set_title('Input Image')
axs[1].imshow(image);axs[1].set_title('Result');
```

请注意输出与 img2img 示例的对比——此处色彩变化更大，但整体结构仍忠实于原图。本例中效果并不理想，因为为了贴合狗的轮廓，人物被生成了极其怪异的解剖结构；但在其他场景下这种方法非常有用。若想了解该方法的「杀手级应用」，可参考[这条推文](https://twitter.com/CarsonKatri/status/1600248599254007810?s=20&t=BlzSK26sfqi2336SN0gKpQ)，其中展示了如何用深度模型为 3D 场景赋予纹理！

# 接下来做什么？

希望本节能让你感受到 Stable Diffusion 的多种用法！当你玩腻了本笔记本中的示例后，可以查看 **DreamBooth 黑客松** 笔记本，学习如何微调属于自己的 Stable Diffusion 版本，并与本文介绍的文生图或 img2img 流水线配合使用。

如果你好奇各组件的底层原理，可以打开 **Stable Diffusion Deep Dive** 笔记本，其中会有更深入的讲解，并展示一些额外技巧。

别忘了把你的创作分享给我们和社区！
