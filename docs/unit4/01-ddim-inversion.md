<!-- This page is generated from the matching notebook by scripts/notebook_to_docs.py. -->

> 原始 Notebook：[unit4/01_ddim_inversion.ipynb](https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/unit4/01_ddim_inversion.ipynb)

# DDIM 反演

本笔记本将探索**反演（inversion）**，了解它与采样的关系，并将其应用于使用 Stable Diffusion 编辑图像的任务。

## 你将学到什么

- DDIM 采样如何工作
- 确定性 vs 随机性采样器
- DDIM 反演背后的理论
- 使用反演编辑图像

让我们开始吧！

## 环境准备

```python
# %pip install -q transformers diffusers accelerate
import os
from dotenv import load_dotenv
load_dotenv("../.env", override=True)
```

```python
import torch
import requests
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from io import BytesIO
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
from torchvision import transforms as tfms
from diffusers import StableDiffusionPipeline, DDIMScheduler
```

```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda:1" if torch.cuda.is_available() else "cpu")
print(device)
```

## 加载现有管道

```python
# Load a pipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5").to(device)
```

```python
# Set up a DDIM scheduler
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
```

```python
# Sample an image to make sure it is all working
prompt = 'Beautiful DSLR Photograph of a penguin on the beach, golden hour'
negative_prompt = 'blurry, ugly, stock photo'
im = pipe(prompt, negative_prompt=negative_prompt).images[0]
im.resize((256, 256)) # Resize for convenient viewing
```

## DDIM 采样

在时刻 $t$，含噪图像 $x_t$ 是原始图像（$x_0$）与某种噪声（$\epsilon$）的混合。以下是 DDIM 论文中 $x_t$ 的公式，本节后续将引用：

$$ x_t = \sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon $$

$\epsilon$ 是方差为单位的高斯噪声
$\alpha_t$（'alpha'）在 DDPM 论文中令人困惑地被记为 $\bar{\alpha}$（'alpha_bar'）（!!），并由噪声调度器定义。在 Diffusers 中，alpha 调度器会被计算，其值存储在 `scheduler.alphas_cumprod` 中。我知道这很混乱！让我们绘制这些值，并记住在本笔记本其余部分我们将使用 DDIM 的记号。

```python
# Plot 'alpha' (alpha_bar in DDPM language, alphas_cumprod in Diffusers for clarity)
timesteps = pipe.scheduler.timesteps.cpu()
alphas = pipe.scheduler.alphas_cumprod[timesteps]
plt.plot(timesteps, alphas, label='alpha_t');
plt.legend();
```

最初（时间步 0，图的左侧）我们从干净图像、无噪声开始。$\alpha_t = 1$。随着时间步增大，我们最终得到几乎全是噪声的状态，$\alpha_t$ 趋近于 0。

在采样过程中，我们从时间步 1000 的纯噪声开始，缓慢走向时间步 0。为计算采样轨迹中的下一个 t（由于从高 t 到低 t，即 $x_{t-1}$），我们预测噪声（$\epsilon_\theta(x_t)$，即模型的输出），并用其计算预测的降噪图像 $x_0$。然后用该预测沿「指向 $x_t$ 的方向」移动一小段距离。最后，我们可以添加由 $\sigma_t$ 缩放的额外噪声。下面是论文中展示这一过程的相关片段：

![Screenshot from 2023-01-28 10-04-22.png](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/ddim_sampling_timesteps.png)

因此，我们有一个从 $x_t$ 到 $x_{t-1}$ 的方程，其中噪声量可控。而今天我们特别关心不添加任何额外噪声的情况——这将得到完全确定性的 DDIM 采样。让我们看看这在代码中是什么样子：

```python
# Sample function (regular DDIM)
@torch.no_grad()
def sample(prompt, start_step=0, start_latents=None,
           guidance_scale=3.5, num_inference_steps=30,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):
  
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Create a random starting point if we don't have one already
    if start_latents is None:
        start_latents = torch.randn(1, 4, 64, 64, device=device)
        start_latents *= pipe.scheduler.init_noise_sigma

    latents = start_latents.clone()

    for i in tqdm(range(start_step, num_inference_steps)):
    
        t = pipe.scheduler.timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)


        # Normally we'd rely on the scheduler to handle the update step:
        # latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

        # Instead, let's do it ourselves:
        prev_t = max(1, t.item() - (1000//num_inference_steps)) # t-1
        alpha_t = pipe.scheduler.alphas_cumprod[t.item()]
        alpha_t_prev = pipe.scheduler.alphas_cumprod[prev_t]
        predicted_x0 = (latents - (1-alpha_t).sqrt()*noise_pred) / alpha_t.sqrt()
        direction_pointing_to_xt = (1-alpha_t_prev).sqrt()*noise_pred
        latents = alpha_t_prev.sqrt()*predicted_x0 + direction_pointing_to_xt

    # Post-processing
    images = pipe.decode_latents(latents)
    images = pipe.numpy_to_pil(images)

    return images
```

```python
# Test our sampling function by generating an image
sample('Watercolor painting of a beach sunset', negative_prompt=negative_prompt, num_inference_steps=50)[0].resize((256, 256))
```

试着将代码与论文中的方程对应起来。注意 $\sigma$=0，因为我们只关心不添加额外噪声的情况，因此可以省略方程中的那些项。

## 反演

反演的目标是「逆转」采样过程。我们希望得到一个含噪潜变量，若将其作为常规采样流程的起点，则能重新生成原始图像。

这里我们加载一张图像作为初始图像，你也可以自己生成一张来使用。

```python
# https://www.pexels.com/photo/a-beagle-on-green-grass-field-8306128/
input_image =Image.open("../images/unit4/puppy.png").convert("RGB").resize((512, 512))
input_image
```

我们还将使用提示词进行反演，并包含 classifier-free guidance，因此请输入对图像的描述：

```python
input_image_prompt = "Photograph of a puppy on the grass"
```

接下来，需要将此 PIL 图像转换为一组潜变量，作为反演的起点：

```python
# Encode with VAE
with torch.no_grad(): latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)*2-1)
l = 0.18215 * latent.latent_dist.sample()
```

好了，进入精彩部分。该函数与上面的采样函数类似，但时间步方向相反：从 t=0 开始，向越来越大的噪声方向移动。我们不是更新潜变量使其降噪，而是估计预测噪声并用其**撤销**一次更新步骤，将它们从 t 移到 t+1。

```python
## Inversion
@torch.no_grad()
def invert(start_latents, prompt, guidance_scale=3.5, num_inference_steps=80,
           num_images_per_prompt=1, do_classifier_free_guidance=True,
           negative_prompt='', device=device):
  
    # Encode prompt
    text_embeddings = pipe._encode_prompt(
            prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
    )

    # Latents are now the specified start latents
    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)

    # Reversed timesteps <<<<<<<<<<<<<<<<<<<<
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps-1):

        # We'll skip the final iteration
        if i >= num_inference_steps - 1: continue

        t = timesteps[i]

        # Expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

        # Predict the noise residual
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # Perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        current_t = max(0, t.item() - (1000//num_inference_steps)) #t
        next_t = t # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1-alpha_t).sqrt()*noise_pred)*(alpha_t_next.sqrt()/alpha_t.sqrt()) + (1-alpha_t_next).sqrt()*noise_pred


        # Store
        intermediate_latents.append(latents)
            
    return torch.cat(intermediate_latents)
```

在小狗图片的潜变量表示上运行后，我们会得到反演过程中创建的所有中间潜变量：

```python
inverted_latents = invert(l, input_image_prompt, num_inference_steps=50)
inverted_latents.shape
```

可以查看最终的潜变量——希望它们能成为新采样尝试的含噪起点：

```python
# Decode the final inverted latents
with torch.no_grad():
  im = pipe.decode_latents(inverted_latents[-1].unsqueeze(0))
pipe.numpy_to_pil(im)[0]
```

可以使用常规的 __call__ 方法将这些反演后的潜变量传给管道：

```python
pipe(input_image_prompt, latents=inverted_latents[-1][None], num_inference_steps=50, guidance_scale=3.5).images[0]
```

但这里我们看到第一个问题：这**与我们开始的图像并不完全一致**！这是因为 DDIM 反演依赖一个关键假设：时刻 t 与 t+1 的噪声预测相同——当我们只在 50 或 100 个时间步上反演时，这并不成立。可以使用更多时间步以获得更准确的反演，也可以「取巧」：例如从采样过程的 20/50 步开始，使用反演时保存的对应中间潜变量：

```python
# The reason we want to be able to specify start step
start_step = 20
sample(input_image_prompt, start_latents=inverted_latents[-(start_step+1)][None], 
       start_step=start_step, num_inference_steps=50)[0]
```

与输入图像非常接近！为什么要这样做？希望是：若现在用新提示词采样，会得到与原始图像一致的图像，**仅在与新提示词相关的部分**发生变化。例如将 'puppy' 替换为 'cat'，应看到一只猫，草坪和背景几乎相同：

```python
# Sampling with a new prompt
start_step = 10
new_prompt = input_image_prompt.replace('puppy', 'cat')
sample(new_prompt, start_latents=inverted_latents[-(start_step+1)][None], 
       start_step=start_step, num_inference_steps=50)[0]
```

### 为什么不直接用 img2img？

何必费劲反演？不能直接向输入图像加噪声并用新提示词去噪吗？可以，但这会导致全局变化过大（若加很多噪声）或各处变化不足（若加较少噪声）。自己动手试试：

```python
start_step = 10
num_inference_steps = 50
pipe.scheduler.set_timesteps(num_inference_steps)
noisy_l = pipe.scheduler.add_noise(l, torch.randn_like(l), pipe.scheduler.timesteps[start_step])
sample(new_prompt, start_latents=noisy_l, start_step=start_step, num_inference_steps=num_inference_steps)[0]
```

注意草坪和背景的变化大得多。

# 整合在一起

将目前为止编写的代码封装成一个简单函数：接收一张图像和两个提示词，使用反演进行编辑：

```python
def edit(input_image, input_image_prompt, edit_prompt, num_steps=100, start_step=30, guidance_scale=3.5):
    with torch.no_grad(): latent = pipe.vae.encode(tfms.functional.to_tensor(input_image).unsqueeze(0).to(device)*2-1)
    l = 0.18215 * latent.latent_dist.sample()
    inverted_latents = invert(l, input_image_prompt, num_inference_steps=num_steps)
    final_im = sample(edit_prompt, start_latents=inverted_latents[-(start_step+1)][None], 
                      start_step=start_step, num_inference_steps=num_steps, guidance_scale=guidance_scale)[0]
    return final_im
```

实际效果：

```python
edit(input_image, 'A puppy on the grass', 'an old grey dog on the grass', num_steps=50, start_step=10)
```

```python
edit(input_image, 'A puppy on the grass', 'A blue dog on the lawn', num_steps=50, start_step=12, guidance_scale=6)
```

```python
# Exercise: Try this on some more images! Explore the different parameters.
```

## 更多步数 = 更好效果

若反演精度不足，可尝试使用更多步数（代价是运行时间更长）。要测试反演，可用相同提示词调用我们的编辑函数：

```python
# Inversion test with far more steps
edit(input_image, 'A puppy on the grass', 'A puppy on the grass', num_steps=350, start_step=1)
```

好多了！再试一次编辑：

```python
edit(input_image, 'A photograph of a puppy', 'A photograph of a grey cat', num_steps=150, start_step=30, guidance_scale=5.5)
```

```python
# source: https://www.pexels.com/photo/girl-taking-photo-1493111/
face =Image.open("../images/unit4/face.png").convert("RGB").resize((512, 512))
face
```

```python
edit(face, 'A photograph of a face', 'A photograph of a face with sunglasses', num_steps=250, start_step=30, guidance_scale=3.5)
```

```python
edit(face, 'A photograph of a face', 'Acrylic palette knife painting of a face, colorful', num_steps=250, start_step=65, guidance_scale=5.5)
```

# 接下来做什么？

掌握本笔记本的知识后，建议你研究 ['Null-text Inversion'](https://null-text-inversion.github.io/)，它在 DDIM 基础上通过在反演过程中优化 null text（无条件文本提示）来实现更准确的反演和更好的编辑。
