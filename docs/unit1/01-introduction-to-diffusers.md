<!-- This page is generated from the matching notebook by scripts/notebook_to_docs.py. -->

> 原始 Notebook：[unit1/01_introduction_to_diffusers.ipynb](https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/unit1/01_introduction_to_diffusers.ipynb)

# 🤗 Diffusers 入门

![diffusers_library](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit1/diffusers_library.jpg)

在本笔记本中，你将训练第一个扩散模型，用于**生成可爱的蝴蝶图像 🦋。** 在此过程中，你将了解 🤗 Diffusers 库的核心组件，为课程后续更高级的应用打下良好基础。

让我们开始吧！

## 你将学到什么

在本笔记本中，你将：

- 看到一个强大的自定义扩散模型管道（Pipeline）的实际运行效果（并了解如何制作自己的版本）
- 通过以下步骤创建你自己的迷你管道：
  - 回顾扩散模型的核心思想
  - 从 Hub 加载训练数据
  - 探索如何使用调度器（Scheduler）向数据添加噪声
  - 创建并训练 UNet 模型
  - 将各部分组装成可运行的管道
- 编辑并运行用于启动更长训练运行的脚本，该脚本将处理：
  - 通过 🤗 Accelerate 进行多 GPU 训练
  - 实验日志记录以跟踪关键统计信息
  - 将最终模型上传到 Hugging Face Hub

❓如有任何问题，请在 Hugging Face Discord 服务器的 `#diffusion-models-class` 频道中提问。如果你还没有注册，可以通过以下链接加入：https://huggingface.co/join/discord

## 前置要求

在开始本笔记本之前，你应该：

* 📖 阅读第 1 单元材料
* 🤗 在 Hugging Face Hub 上创建账户。如果还没有，可以在这里注册：https://huggingface.co/join

## 步骤 1：环境准备

运行以下单元格以安装 diffusers 库以及其他一些依赖项：

```python
%pip install -qq -U diffusers datasets transformers accelerate ftfy pyarrow==9.0.0
```

接下来，前往 https://huggingface.co/settings/tokens 创建一个具有写入权限的访问令牌（如果还没有的话）：

![Screenshot from 2022-11-10 12-23-34.png](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit1/huggingface_token_settings.png)

你可以使用命令行（`huggingface-cli login`）或运行以下单元格，用该令牌登录：

```python
from huggingface_hub import notebook_login

notebook_login()
```

然后你需要安装 Git-LFS 以上传模型检查点：

```python
%%capture
!sudo apt -qq install git-lfs
!git config --global credential.helper store
```

最后，让我们导入将要用到的库，并定义几个稍后在笔记本中会用到的便捷函数：

```python
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from PIL import Image


def show_images(x):
    """Given a batch of images x, make a grid and convert to PIL"""
    x = x * 0.5 + 0.5  # Map from (-1, 1) back to (0, 1)
    grid = torchvision.utils.make_grid(x)
    grid_im = grid.detach().cpu().permute(1, 2, 0).clip(0, 1) * 255
    grid_im = Image.fromarray(np.array(grid_im).astype(np.uint8))
    return grid_im


def make_grid(images, size=64):
    """Given a list of PIL images, stack them together into a line for easy viewing"""
    output_im = Image.new("RGB", (size * len(images), size))
    for i, im in enumerate(images):
        output_im.paste(im.resize((size, size)), (i * size, 0))
    return output_im


# Mac users may need device = 'mps' (untested)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

好了，一切准备就绪！

## Dreambooth：提前一窥后续内容

如果你在过去几个月里关注过 AI 相关的社交媒体，一定听说过 Stable Diffusion。它是一种强大的文本条件潜空间扩散模型（别担心，我们会学会这些术语的含义）。但它有一个缺陷：除非你我足够有名、照片遍布互联网，否则它并不知道我们长什么样。

Dreambooth 让我们能够创建自己的模型变体，额外学习特定人脸、物体或风格的知识。The Corridor Crew 用这项技术制作了精彩的视频，用一致的角色来讲述故事，很好地展示了这种方法能做什么：

```python
from IPython.display import YouTubeVideo

YouTubeVideo("W4Mcuh38wyM")
```

下面是一个示例，使用在 5 张流行儿童玩具「Mr Potato Head」照片上训练的[模型](https://huggingface.co/sd-dreambooth-library/mr-potato-head)。

首先，我们加载管道。这会从 Hub 下载模型权重等文件。由于仅为一行演示就要下载数 GB 数据，你可以跳过此单元格，直接欣赏示例输出！

```python
from diffusers import StableDiffusionPipeline

# Check out https://huggingface.co/sd-dreambooth-library for loads of models from the community
model_id = "sd-dreambooth-library/mr-potato-head"

# Load the pipeline
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    device
)
```

管道加载完成后，我们可以用以下方式生成图像：

```python
prompt = "an abstract oil painting of sks mr potato head by picasso"
image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5).images[0]
image
```

**练习：** 尝试使用不同的提示词。`sks` 令牌在此代表新概念的唯一标识符——如果去掉它会怎样？你也可以尝试改变采样步数（最少能降到多少？）和 `guidance_scale`，后者决定模型在多大程度上努力匹配提示词。

那个神奇的管道里发生了很多事情！课程结束时你会明白这一切如何运作。现在，让我们看看如何从头训练一个扩散模型。

## MVP（最小可行管道）

🤗 Diffusers 的核心 API 分为三个主要组件：
1. **Pipelines（管道）**：高级类，旨在以用户友好的方式，从流行的已训练扩散模型快速生成样本。
2. **Models（模型）**：用于训练新扩散模型的流行架构，*例如* [UNet](https://arxiv.org/abs/1505.04597)。
3. **Schedulers（调度器）**：在*推理*时从噪声生成图像的各种技术，以及在*训练*时生成带噪图像的技术。

管道非常适合终端用户，但如果你来参加这门课程，我们假定你想了解底层原理！因此，在本笔记本的剩余部分，我们将构建自己的管道，能够生成小型蝴蝶图片。下面是最终效果的演示：

```python
from diffusers import DDPMPipeline

# Load the butterfly pipeline
butterfly_pipeline = DDPMPipeline.from_pretrained(
    "johnowhitaker/ddpm-butterflies-32px"
).to(device)

# Create 8 images
images = butterfly_pipeline(batch_size=8).images

# View the result
make_grid(images)
```

也许不如 DreamBooth 示例那么惊艳，但我们是从头训练，且只使用了训练 Stable Diffusion 所用数据量的约 0.0001%。说到训练，回想本单元介绍中提到的，训练扩散模型大致如下：


1.   从训练数据中加载一些图像
2.   以不同强度添加噪声
3.   将带噪输入送入模型
4.   评估模型对这些输入的去噪效果
5.   利用该信息更新模型权重，然后重复

我们将在接下来几节中逐步探索这些步骤，直到完整的训练循环运行起来，然后探索如何从训练好的模型采样，以及如何将一切打包成管道以便轻松分享。让我们从数据开始……

## 步骤 2：下载训练数据集

在本示例中，我们将使用来自 Hugging Face Hub 的图像数据集。具体而言，是[这 1000 张蝴蝶图片的集合](https://huggingface.co/datasets/huggan/smithsonian_butterflies_subset)。这是一个非常小的数据集，因此我们还提供了几个更大数据集的注释行供选择。如果你更愿意使用自己的图像集合，也可以使用注释掉的代码示例从文件夹加载图片。

```python
import torchvision
from datasets import load_dataset
from torchvision import transforms

dataset = load_dataset("huggan/smithsonian_butterflies_subset", split="train")

# Or load images from a local folder
# dataset = load_dataset("imagefolder", data_dir="path/to/folder")

# We'll train on 32-pixel square images, but you can try larger sizes too
image_size = 32
# You can lower your batch size if you're running out of GPU memory
batch_size = 64

# Define data augmentations
preprocess = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),  # Resize
        transforms.RandomHorizontalFlip(),  # Randomly flip (data augmentation)
        transforms.ToTensor(),  # Convert to tensor (0, 1)
        transforms.Normalize([0.5], [0.5]),  # Map to (-1, 1)
    ]
)


def transform(examples):
    images = [preprocess(image.convert("RGB")) for image in examples["image"]]
    return {"images": images}


dataset.set_transform(transform)

# Create a dataloader from the dataset to serve up the transformed images in batches
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True
)
```

我们可以取一批图像并像这样查看其中一些：

```python
xb = next(iter(train_dataloader))["images"].to(device)[:8]
print("X shape:", xb.shape)
show_images(xb).resize((8 * 64, 64), resample=Image.NEAREST)
```

为保持本笔记本中的训练时间可控，我们使用 32 像素图像的小数据集。

## 步骤 3：定义调度器

我们的训练计划是：对这些输入图像添加噪声，然后将带噪图像送入模型。在推理时，我们将使用模型预测迭代地去除噪声。在 `diffusers` 中，这两个过程都由**调度器**处理。

噪声调度决定了在不同时间步添加多少噪声。下面展示如何使用 DDPM 训练和采样的默认设置创建调度器（基于论文 ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239)）：

```python
from diffusers import DDPMScheduler

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
```

DDPM 论文描述了一个腐蚀过程，在每个「时间步」添加少量噪声。给定某个时间步的 $x_{t-1}$，我们可以得到下一个（噪声稍大一点的）版本 $x_t$：<br><br>

$q(\mathbf{x}_t \vert \mathbf{x}_{t-1}) = \mathcal{N}(\mathbf{x}_t; \sqrt{1 - \beta_t} \mathbf{x}_{t-1}, \beta_t\mathbf{I}) \quad
q(\mathbf{x}_{1:T} \vert \mathbf{x}_0) = \prod^T_{t=1} q(\mathbf{x}_t \vert \mathbf{x}_{t-1})$<br><br>


也就是说，我们取 $x_{t-1}$，乘以 $\sqrt{1 - \beta_t}$，再加上按 $\beta_t$ 缩放的噪声。这个 $\beta$ 根据某种调度为每个 t 定义，并决定每个时间步添加多少噪声。我们不一定想执行 500 次该操作来得到 $x_{500}$，因此还有另一个公式，给定 $x_0$ 直接得到任意 t 的 $x_t$：<br><br>

$\begin{aligned}
q(\mathbf{x}_t \vert \mathbf{x}_0) &= \mathcal{N}(\mathbf{x}_t; \sqrt{\bar{\alpha}_t} \mathbf{x}_0, {(1 - \bar{\alpha}_t)} \mathbf{I})
\end{aligned}$ 其中 $\bar{\alpha}_t = \prod_{i=1}^T \alpha_i$，且 $\alpha_i = 1-\beta_i$<br><br>

数学符号总是看起来很吓人！好在调度器会帮我们处理这一切。我们可以绘制 $\sqrt{\bar{\alpha}_t}$（标记为 `sqrt_alpha_prod`）和 $\sqrt{(1 - \bar{\alpha}_t)}$（标记为 `sqrt_one_minus_alpha_prod`），查看输入 (x) 与噪声在不同时间步如何被缩放和混合：

```python
plt.plot(noise_scheduler.alphas_cumprod.cpu() ** 0.5, label=r"${\sqrt{\bar{\alpha}_t}}$")
plt.plot((1 - noise_scheduler.alphas_cumprod.cpu()) ** 0.5, label=r"$\sqrt{(1 - \bar{\alpha}_t)}$")
plt.legend(fontsize="x-large");
```

**练习：** 你可以通过在此处替换为注释掉的选项之一，探索在 beta_start、beta_end 和 beta_schedule 不同设置下该图如何变化：

```python
# One with too little noise added:
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_start=0.001, beta_end=0.004)
# The 'cosine' schedule, which may be better for small image sizes:
# noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
```

无论你选择了哪个调度器，现在都可以使用 `noise_scheduler.add_noise` 函数以不同强度添加噪声，如下所示：

```python
timesteps = torch.linspace(0, 999, 8).long().to(device)
noise = torch.randn_like(xb)
noisy_xb = noise_scheduler.add_noise(xb, noise, timesteps)
print("Noisy X shape", noisy_xb.shape)
show_images(noisy_xb).resize((8 * 64, 64), resample=Image.NEAREST)
```

再次尝试探索使用不同噪声调度和参数的效果。[这个视频](https://www.youtube.com/watch?v=fbLgFrlTnGU) 很好地更详细地解释了上述部分数学，是理解这些概念的绝佳入门。

## 步骤 4：定义模型

现在我们来到核心组件：模型本身。

大多数扩散模型使用 [U-net](https://arxiv.org/abs/1505.04597) 的某种变体架构，我们在这里也采用它。

![](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit1/unet_model.png)

简而言之：
- 模型让输入图像经过若干 ResNet 层块，每个块将图像尺寸减半
- 然后通过相同数量的块将其上采样回来
- 下采样路径中的特征通过跳跃连接（skip connection）链接到上采样路径中对应的层

该模型的一个关键特性是：它预测与输入相同尺寸的图像，这正是我们此处所需要的。

Diffusers 提供了便捷的 `UNet2DModel` 类，可在 PyTorch 中创建所需架构。

让我们为期望的图像尺寸创建一个 U-net。
注意 `down_block_types` 对应下采样块（上图中绿色），`up_block_types` 对应上采样块（上图中红色）：

```python
from diffusers import UNet2DModel

# Create a model
model = UNet2DModel(
    sample_size=image_size,  # the target image resolution
    in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",  # a regular ResNet upsampling block
    ),
)
model.to(device);
```

处理更高分辨率输入时，你可能希望使用更多下采样和上采样块，并仅在最低分辨率（底部）层保留注意力层以降低显存占用。稍后我们会讨论如何针对你的用例实验以找到最佳设置。

我们可以验证：传入一批数据和一些随机时间步，输出的形状与输入数据相同：

```python
with torch.no_grad():
    model_prediction = model(noisy_xb, timesteps).sample
model_prediction.shape
```

在下一节中，我们将看到如何训练该模型。

## 步骤 5：创建训练循环

该训练了！下面是一个典型的 PyTorch 优化循环：我们逐批遍历数据，并使用优化器每步更新模型参数——此处使用学习率为 0.0004 的 AdamW 优化器。

对每批数据，我们：
- 随机采样一些时间步
- 据此对数据加噪
- 将带噪数据送入模型
- 使用均方误差作为损失函数，比较模型预测与目标（此处为噪声）
- 通过 `loss.backward()` 和 `optimizer.step()` 更新模型参数

在此过程中，我们还会记录损失随时间的变化，以便稍后绘图。

注意：此代码运行近 10 分钟——如果赶时间，可以跳过下面两个单元格并使用预训练模型。或者，你可以探索通过上面的模型定义减少每层的通道数来加速训练。

[官方 diffusers 训练示例](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/training_example.ipynb) 在此数据集上以更高分辨率训练更大的模型，是了解不那么「极简」的训练循环的良好参考：

```python
# Set the noise scheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2"
)

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

losses = []

for epoch in range(30):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Get the model prediction
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)
        losses.append(loss.item())

        # Update the model parameters with the optimizer
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 5 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
```

绘制损失曲线可以看到，模型初期快速改进，然后以更慢的速度持续变好（若使用对数刻度（右侧图）会更明显）：

```python
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(losses)
axs[1].plot(np.log(losses))
plt.show()
```

作为运行上述训练代码的替代方案，你可以像这样使用管道中的模型：

```python
# Uncomment to instead load the model I trained earlier:
# model = butterfly_pipeline.unet
```

## 步骤 6：生成图像

如何用该模型得到图像？

### 选项 1：创建管道

```python
from diffusers import DDPMPipeline

image_pipe = DDPMPipeline(unet=model, scheduler=noise_scheduler)
```

```python
pipeline_output = image_pipe()
pipeline_output.images[0]
```

我们可以像这样将管道保存到本地文件夹：

```python
image_pipe.save_pretrained("my_pipeline")
```

查看文件夹内容：

```bash
!ls my_pipeline/
```

`scheduler` 和 `unet` 子文件夹包含重新创建这些组件所需的一切。例如，在 `unet` 文件夹中你会找到模型权重（`diffusion_pytorch_model.bin`）以及指定 UNet 架构的配置文件。

```bash
!ls my_pipeline/unet/
```

这些文件合在一起包含重新创建管道所需的一切。你可以手动将它们上传到 Hub 与他人分享管道，或在下一节查看通过 API 完成此操作的代码。

### 选项 2：编写采样循环

如果你查看管道的 forward 方法，就能看到运行 `image_pipe()` 时发生了什么：

```python
# ??image_pipe.forward
```

我们从随机噪声开始，按调度器时间步从噪声最多到最少运行，根据模型预测每步去除少量噪声：

```python
# Random starting point (8 random images):
sample = torch.randn(8, 3, 32, 32).to(device)

for i, t in enumerate(noise_scheduler.timesteps):

    # Get model pred
    with torch.no_grad():
        residual = model(sample, t).sample

    # Update sample with step
    sample = noise_scheduler.step(residual, t, sample).prev_sample

show_images(sample)
```

`noise_scheduler.step()` 函数完成适当更新 `sample` 所需的数学运算。有多种采样方法——在下一单元中，我们将看到如何为现有模型换入不同的采样器以加速图像生成，并更多地讨论从扩散模型采样的理论。

## 步骤 7：将模型推送到 Hub

在上面的示例中，我们将管道保存到了本地文件夹。要将模型推送到 Hub，需要一个模型仓库来推送文件。我们将根据想给模型起的 ID 确定仓库名称（可随意替换 `model_name`；只需包含你的用户名，这正是 `get_full_repo_name()` 函数所做的）：

```python
from huggingface_hub import get_full_repo_name

model_name = "sd-class-butterflies-32"
hub_model_id = get_full_repo_name(model_name)
hub_model_id
```

接下来，在 🤗 Hub 上创建模型仓库并推送我们的模型：

```python
from huggingface_hub import HfApi, create_repo

create_repo(hub_model_id)
api = HfApi()
api.upload_folder(
    folder_path="my_pipeline/scheduler", path_in_repo="", repo_id=hub_model_id
)
api.upload_folder(folder_path="my_pipeline/unet", path_in_repo="", repo_id=hub_model_id)
api.upload_file(
    path_or_fileobj="my_pipeline/model_index.json",
    path_in_repo="model_index.json",
    repo_id=hub_model_id,
)
```

最后要做的是创建一张漂亮的模型卡片，让我们的蝴蝶生成器在 Hub 上容易被找到（欢迎扩展和编辑描述！）：

````python
from huggingface_hub import ModelCard

content = f"""
---
license: mit
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

# Model Card for Unit 1 of the [Diffusion Models Class 🧨](https://github.com/huggingface/diffusion-models-class)

This model is a diffusion model for unconditional image generation of cute 🦋.

## Usage

```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('{hub_model_id}')
image = pipeline().images[0]
image
```
"""

card = ModelCard(content)
card.push_to_hub(hub_model_id)
````

现在模型已在 Hub 上，你可以在任何地方通过 `DDPMPipeline` 的 `from_pretrained()` 方法下载，如下所示：

```python
from diffusers import DDPMPipeline

image_pipe = DDPMPipeline.from_pretrained(hub_model_id)
pipeline_output = image_pipe()
pipeline_output.images[0]
```

太好了，可以用了！

# 使用 🤗 Accelerate 扩展规模

本笔记本用于学习目的，因此我尽量保持代码精简清晰。正因如此，我们省略了一些在更大数据上训练更大模型时可能想要的功能，例如多 GPU 支持、进度和示例图像日志、梯度检查点以支持更大批次、自动上传模型等。好在这些功能大多可在[此处的示例训练脚本](https://github.com/huggingface/diffusers/raw/main/examples/unconditional_image_generation/train_unconditional.py)中找到。

你可以像这样下载该文件：

```bash
!wget https://github.com/huggingface/diffusers/raw/main/examples/unconditional_image_generation/train_unconditional.py
```

打开文件，你会看到模型定义的位置以及可用的设置。我用以下命令运行了该脚本：

```python
# Let's give our new model a name for the Hub
model_name = "sd-class-butterflies-64"
hub_model_id = get_full_repo_name(model_name)
hub_model_id
```

```bash
!accelerate launch train_unconditional.py \
  --dataset_name="huggan/smithsonian_butterflies_subset" \
  --resolution=64 \
  --output_dir={model_name} \
  --train_batch_size=32 \
  --num_epochs=50 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision="no"
```

和之前一样，让我们将模型推送到 Hub 并创建一张漂亮的模型卡片（欢迎按需编辑！）：

````python
create_repo(hub_model_id)
api = HfApi()
api.upload_folder(
    folder_path=f"{model_name}/scheduler", path_in_repo="", repo_id=hub_model_id
)
api.upload_folder(
    folder_path=f"{model_name}/unet", path_in_repo="", repo_id=hub_model_id
)
api.upload_file(
    path_or_fileobj=f"{model_name}/model_index.json",
    path_in_repo="model_index.json",
    repo_id=hub_model_id,
)

content = f"""
---
license: mit
tags:
- pytorch
- diffusers
- unconditional-image-generation
- diffusion-models-class
---

# Model Card for Unit 1 of the [Diffusion Models Class 🧨](https://github.com/huggingface/diffusion-models-class)

This model is a diffusion model for unconditional image generation of cute 🦋.

## Usage

```python
from diffusers import DDPMPipeline

pipeline = DDPMPipeline.from_pretrained('{hub_model_id}')
image = pipeline().images[0]
image
```
"""

card = ModelCard(content)
card.push_to_hub(hub_model_id)
````

大约 45 分钟后，结果如下：

```python
pipeline = DDPMPipeline.from_pretrained(hub_model_id).to(device)
images = pipeline(batch_size=8).images
make_grid(images)
```

**练习：** 看看能否找到在尽可能短的时间内给出良好结果的训练/模型设置，并与社区分享你的发现。在脚本中四处查看，看能否理解代码，对任何看起来令人困惑的地方请求澄清。

# 进一步探索的方向

希望这能让你初步了解 🤗 Diffusers 库能做什么！一些可能的下一步：

- 在新数据集上尝试训练无条件扩散模型——如果你[自己创建数据集](https://huggingface.co/docs/datasets/image_dataset)则加分。你可以在 Hub 的 [HugGan 组织](https://huggingface.co/huggan)找到适合此任务的一些优质图像数据集。如果不想等很久才训练完，记得对它们进行下采样！
- 尝试 DreamBooth，使用[这个 Space](https://huggingface.co/spaces/multimodalart/dreambooth-training) 或[这个笔记本](https://colab.research.google.com/github/huggingface/notebooks/blob/main/diffusers/sd_dreambooth_training.ipynb) 创建你自己的定制 Stable Diffusion 管道
- 修改训练脚本，探索不同的 UNet 超参数（层数、通道数等）、不同的噪声调度等
- 查看 [从零开始的扩散模型](https://github.com/huggingface/diffusion-models-class/blob/main/unit1/02-diffusion-models-from-scratch.md) 笔记本，从另一个角度理解本单元涵盖的核心思想

祝你好运，第 2 单元见！
