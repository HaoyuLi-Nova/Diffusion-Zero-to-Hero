<!-- This page is generated from the matching notebook by scripts/notebook_to_docs.py. -->

> 原始 Notebook：[hackathon/dreambooth.ipynb](https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/hackathon/dreambooth.ipynb)

# DreamBooth 黑客松 🏆

欢迎来到 DreamBooth 黑客松！在本次竞赛中，你将**用少量自己的图像对 Stable Diffusion 进行微调，从而个性化模型。**我们将使用一种名为 [_DreamBooth_](https://arxiv.org/abs/2208.12242) 的技术：它可以把某个主体（例如你的宠物或最爱的一道菜）“植入”模型的输出域，这样在提示词中使用一个_唯一标识符_时，就能合成该主体。

让我们开始吧！

## 前置阅读

在深入本笔记本之前，建议你先阅读：

* [Unit 3 README](https://github.com/huggingface/diffusion-models-class/blob/main/unit3/README.md)：深入介绍 Stable Diffusion
* DreamBooth [博客文章](https://dreambooth.github.io/)：了解这项技术能做什么
* Hugging Face [博客文章](https://huggingface.co/blog/dreambooth)：使用 DreamBooth 微调 Stable Diffusion 的最佳实践

🚨 **注意：**本笔记本中的代码**至少需要 14GB GPU 显存**，并且是 🤗 Diffusers 提供的[官方训练脚本](https://github.com/huggingface/diffusers/tree/main/examples/dreambooth)的简化版本。它对大多数应用已经能产出不错的模型；若你至少有 24GB 显存，建议尝试类别先验保留损失（class preservation loss）、微调文本编码器等高级功能。更多细节请参阅 🤗 Diffusers [文档](https://huggingface.co/docs/diffusers/training/dreambooth)。

## 什么是 DreamBooth？

DreamBooth 是一种通过专门形式的微调，向 Stable Diffusion 教授新概念的技术。如果你在 Twitter 或 Reddit 上逛过，可能已经见过有人用它生成（常常很搞笑的）个人头像。例如，[Andrej Karpathy](https://karpathy.ai/) 变成牛仔会是什么样子（你可能需要运行下面的单元格才能看到输出）：

```python
%%html
<blockquote class="twitter-tweet"><p lang="en" dir="ltr">Stableboost auto-suggests a few hundred prompts by default but you can generate additional variations for any one prompt that seems to be giving fun/interesting results, or adjust it in any way: <a href="https://t.co/qWmadiXftP">pic.twitter.com/qWmadiXftP</a></p>&mdash; Andrej Karpathy (@karpathy) <a href="https://twitter.com/karpathy/status/1600578187141840896?ref_src=twsrc%5Etfw">December 7, 2022</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
```

DreamBooth 的工作方式如下：

* 收集约 10–20 张主体（例如你的狗）的输入图像，并定义一个指向该主体的唯一标识符 [V]。该标识符通常是虚构词，例如 `flffydog`，在推理时会植入不同的文本提示词中，从而把主体放到不同场景里。
* 将图像与包含唯一标识符和类别名称的文本提示词（例如本例中的 “A photo of a [V] dog”，即“一张 [V] 狗的照片”）一起提供给模型，对扩散模型进行微调。
* （可选）应用特殊的_类别特定先验保留损失（class-specific prior preservation loss）_：它利用模型对该类别已有的语义先验，通过在提示词中注入类别名称，鼓励生成属于该主体类别的多样化实例。实践中，这一步主要在人脸场景才真的需要；本次黑客松探索的主题可以跳过。

下图概览了 DreamBooth 技术：

![](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/hackathon/dreambooth_high_level.png)

### DreamBooth 能做什么？

除了把主体放到有趣的位置，DreamBooth 还可用于_**文本引导的视角合成（text-guided view synthesis）**_，即从不同视角观察主体，如下例所示：

![](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/hackathon/dreambooth_novel_views.png)

DreamBooth 还可以修改主体的属性，例如颜色，或混合不同动物物种！

![](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/hackathon/dreambooth_property_modification.png)

见识了 DreamBooth 的这些酷炫能力之后，让我们开始训练自己的模型吧！

## 第 1 步：环境准备

如果你在 Google Colab 或 Kaggle 上运行本笔记本，请运行下面的单元格以安装所需库：

```python
%pip install -qqU diffusers transformers bitsandbytes accelerate ftfy datasets
```

若在 Kaggle 上运行，你需要安装最新版 PyTorch 才能与 🤗 Accelerate 配合使用：

```python
# Uncomment and run if using Kaggle's notebooks. You may need to restart the notebook afterwards
# %pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

要将模型推送到 Hub 并出现在 [DreamBooth 排行榜](https://huggingface.co/spaces/dreambooth-hackathon/leaderboard) 上，还需要几个步骤。首先，在你的 Hugging Face 账户中创建一个具有_**写入权限**_的[访问令牌（access token）](https://huggingface.co/docs/hub/security-tokens)，然后运行下面的单元格并输入你的令牌：

```python
from huggingface_hub import notebook_login

notebook_login()
```

最后一步是安装 Git LFS：

```python
%%capture
!sudo apt -qq install git-lfs
!git config --global credential.helper store
```

## 第 2 步：选择主题

本次竞赛包含 5 个_主题_，每个主题对应以下类别之一的模型：

* **动物 🐨：** 让你的宠物或最喜欢的动物出现在雅典卫城、游泳，或飞向太空。
* **科学 🔬：** 生成星系、蛋白质等自然与医学科学领域的酷炫合成图像。
* **美食 🍔：** 在你最爱的一道菜或菜系上微调 Stable Diffusion。
* **风景 🏔：** 生成你最喜欢的山、湖或花园的美丽风景。
* **Wildcard 🔥：** 尽情发挥，为任意类别创建 Stable Diffusion 模型！

我们将**按主题对获赞最多的前 3 名模型颁奖**，欢迎你提交任意数量的模型！运行下面的单元格，通过下拉菜单选择要提交的主题：

```python
import ipywidgets as widgets

theme = "animal"
drop_down = widgets.Dropdown(
    options=["animal", "science", "food", "landscape", "wildcard"],
    description="Pick a theme",
    disabled=False,
)


def dropdown_handler(change):
    global theme
    theme = change.new


drop_down.observe(dropdown_handler, names="value")
display(drop_down)
```

```python
print(f"You've selected the {theme} theme!")
```

## 第 3 步：创建图像数据集并上传到 Hub

选定主题后，下一步是**为该主题创建图像数据集**并上传到 Hugging Face Hub：

* 你需要大约 **10–20 张**希望植入模型的主体图像。可以是自拍照，也可以从 [Unsplash](https://unsplash.com/) 等平台下载。你也可以浏览 Hub 上的[图像数据集](https://huggingface.co/datasets?task_categories=task_categories:image-classification&sort=downloads)寻找灵感。
* 为获得最佳效果，建议使用**不同角度和视角**的主体图像。

将图像收集到文件夹后，可通过 Hub UI 拖放上传。详见[本指南](https://huggingface.co/docs/datasets/upload_dataset#upload-with-the-hub-ui)，或观看下面的视频：

```python
from IPython.display import YouTubeVideo

YouTubeVideo("HaN6qCr_Afc")
```

你也可以使用 🤗 Datasets 的 `imagefolder` 在本地加载数据集，再推送到 Hub：

```python
from datasets import load_dataset

dataset = load_dataset("imagefolder", data_dir="your_folder_of_images")
# Push to Hub
dataset.push_to_hub("dreambooth-hackathon-images")
dataset = dataset['train']
```

创建数据集后，可用 `load_dataset()` 函数如下下载：

```python
from datasets import load_dataset

dataset_id = "lewtun/corgi"  # CHANGE THIS TO YOUR {hub_username}/{dataset_id}
dataset = load_dataset(dataset_id, split="train")
dataset
```

现在有了数据集，让我们定义一个辅助函数来查看其中几张图像：

```python
from PIL import Image


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols
    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


num_samples = 4
image_grid(dataset["image"][:num_samples], rows=1, cols=num_samples)
```

如果看起来不错，就可以进入下一步——为 DreamBooth 训练创建 PyTorch 数据集。

## 第 3 步：创建训练数据集

要为图像创建训练集，需要以下几个组件：

* _实例提示词（instance prompt）_：在训练开始时用于“预热”模型。大多数情况下，使用 “a photo of [identifier] [class noun]”（一张 [标识符] [类别名词] 的照片）效果就很好，例如对我们的可爱柯基照片使用 “a photo of ccorgi dog”。
    * **注意：** 建议为描述主体选择一个独特/虚构的词，例如 `ccorgi`，以免覆盖模型词表中的常见词。
* _分词器（tokenizer）_：将实例提示词转换为可输入 Stable Diffusion 文本编码器的 input ID。
* 一组_图像变换_：尤其是将图像缩放到统一尺寸，并将像素值归一化到统一的均值和标准差。

明确这些之后，我们先定义实例提示词：

```python
name_of_your_concept = "ccorgi"  # CHANGE THIS ACCORDING TO YOUR SUBJECT
type_of_thing = "dog"  # CHANGE THIS ACCORDING TO YOUR SUBJECT
instance_prompt = f"a photo of {name_of_your_concept} {type_of_thing}"
print(f"Instance prompt: {instance_prompt}")
```

接下来，需要实现 `__len__` 和 `__getitem__` 的 PyTorch `Dataset` 对象：

```python
from torch.utils.data import Dataset
from torchvision import transforms


class DreamBoothDataset(Dataset):
    def __init__(self, dataset, instance_prompt, tokenizer, size=512):
        self.dataset = dataset
        self.instance_prompt = instance_prompt
        self.tokenizer = tokenizer
        self.size = size
        self.transforms = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        example = {}
        image = self.dataset[index]["image"]
        example["instance_images"] = self.transforms(image)
        example["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            padding="do_not_pad",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        ).input_ids
        return example
```

很好。现在加载与原始 Stable Diffusion 模型文本编码器关联的 CLIP 分词器，并创建训练数据集，检查是否能正常工作：

```python
from transformers import CLIPTokenizer

# The Stable Diffusion checkpoint we'll fine-tune
model_id = "CompVis/stable-diffusion-v1-4"
tokenizer = CLIPTokenizer.from_pretrained(
    model_id,
    subfolder="tokenizer",
)

train_dataset = DreamBoothDataset(dataset, instance_prompt, tokenizer)
train_dataset[0]
```

## 第 4 步：定义 data collator

有了训练数据集，接下来要定义 _data collator（数据整理器）_。data collator 是一个函数：它收集批次中的样本并应用一定逻辑，形成可送入模型的单个张量。想深入了解，可看 [Hugging Face 课程](https://hf.co/course) 中的视频：

```python
YouTubeVideo("-RPeakdlHYo")
```

对 DreamBooth 而言，data collator 需要向模型提供分词器得到的 input ID，以及由图像堆叠而成的像素值张量。下面的函数即可胜任：

```python
import torch


def collate_fn(examples):
    input_ids = [example["instance_prompt_ids"] for example in examples]
    pixel_values = [example["instance_images"] for example in examples]
    pixel_values = torch.stack(pixel_values)
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    input_ids = tokenizer.pad(
        {"input_ids": input_ids}, padding=True, return_tensors="pt"
    ).input_ids

    batch = {
        "input_ids": input_ids,
        "pixel_values": pixel_values,
    }
    return batch
```

## 第 5 步：加载 Stable Diffusion 管道的组件

训练所需的拼图几乎齐了！正如你在 Unit 3 的 Stable Diffusion 笔记本中所见，管道由多个模型组成：

* **文本编码器**：将提示词转换为文本嵌入。这里使用 CLIP，因为 Stable Diffusion v1-4 训练时用的就是它。
* **VAE（变分自编码器）**：将图像压缩为潜表示（latents），并在推理时解压。
* **UNet**：对 VAE 的潜变量执行去噪。

可用 🤗 Diffusers 和 🤗 Transformers 如下加载这些组件：

```python
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPFeatureExtractor, CLIPTextModel

text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
```

## 第 6 步：微调模型

激动人心的部分来了——用 DreamBooth 训练模型！如 [Hugging Face 博客文章](https://huggingface.co/blog/dreambooth) 所示，最需要调整的超参数是学习率和训练步数。

一般来说，较低的学习率往往效果更好，但需要增加训练步数。下面的数值是不错的起点，你可能需要根据数据集进行调整：

```python
learning_rate = 2e-06
max_train_steps = 400
```

接下来，把训练所需的其他超参数包在 `Namespace` 对象里，便于配置训练运行：

```python
from argparse import Namespace

args = Namespace(
    pretrained_model_name_or_path=model_id,
    resolution=512, # Reduce this if you want to save some memory
    train_dataset=train_dataset,
    instance_prompt=instance_prompt,
    learning_rate=learning_rate,
    max_train_steps=max_train_steps,
    train_batch_size=1,
    gradient_accumulation_steps=1, # Increase this if you want to lower memory usage
    max_grad_norm=1.0,
    gradient_checkpointing=True,  # Set this to True to lower the memory usage
    use_8bit_adam=True,  # Use 8bit optimizer from bitsandbytes
    seed=3434554,
    sample_batch_size=2,
    output_dir="my-dreambooth",  # Where to save the pipeline
)
```

最后一步是定义 `training_function()`，封装训练逻辑，并交给 🤗 Accelerate 在 1 块或多块 GPU 上训练。若这是你第一次使用 🤗 Accelerate，可看这段视频快速了解它能做什么：

```python
YouTubeVideo("s7dy8QRgjJ0")
```

细节应与我们在 Unit 1 和 Unit 2 中从头训练扩散模型时所见类似：

```python
import math

import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import DDPMScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def training_function(text_encoder, vae, unet):

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    set_seed(args.seed)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        unet.parameters(),  # Only optimize unet
        lr=args.learning_rate,
    )

    noise_scheduler = DDPMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
    )

    train_dataloader = DataLoader(
        args.train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )

    # Move text_encode and vae to gpu
    text_encoder.to(accelerator.device)
    vae.to(accelerator.device)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )
    # Only show the progress bar once on each machine
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    progress_bar.set_description("Steps")
    global_step = 0

    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                    latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn(latents.shape).to(latents.device)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with torch.no_grad():
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                noise_pred = unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                loss = (
                    F.mse_loss(noise_pred, noise, reduction="none")
                    .mean([1, 2, 3])
                    .mean()
                )

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

            logs = {"loss": loss.detach().item()}
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    # Create the pipeline using the trained modules and save it
    if accelerator.is_main_process:
        print(f"Loading pipeline and saving to {args.output_dir}...")
        scheduler = PNDMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            skip_prk_steps=True,
            steps_offset=1,
        )
        pipeline = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=accelerator.unwrap_model(unet),
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=StableDiffusionSafetyChecker.from_pretrained(
                "CompVis/stable-diffusion-safety-checker"
            ),
            feature_extractor=feature_extractor,
        )
        pipeline.save_pretrained(args.output_dir)
```

函数定义好后，开始训练！根据数据集规模和 GPU 类型，可能需要 5 分钟到 1 小时：

```python
from accelerate import notebook_launcher

num_of_gpus = 1  # CHANGE THIS TO MATCH THE NUMBER OF GPUS YOU HAVE
notebook_launcher(
    training_function, args=(text_encoder, vae, unet), num_processes=num_of_gpus
)
```

若在单 GPU 上运行，可将下面的代码复制到新单元格并运行，为下一节释放一些显存。对多 GPU 机器，🤗 Accelerate 不允许_任何_单元格通过 `torch.cuda` 直接访问 GPU，因此不建议在多 GPU 情况下使用此技巧：

```python
with torch.no_grad():
    torch.cuda.empty_cache()
```

## 第 7 步：推理并查看生成结果

模型训练完成后，用它生成一些图像看看效果！首先从保存模型的输出目录加载管道：

```python
pipe = StableDiffusionPipeline.from_pretrained(
    args.output_dir,
    torch_dtype=torch.float16,
).to("cuda")
```

接下来生成几张图像。`prompt` 变量之后将用于设置 Hugging Face Hub 小组件的默认值，不妨多试几次找到合适的提示词。你也可以用 [CLIP Interrogator](https://huggingface.co/spaces/pharma/CLIP-Interrogator) 编写更丰富的提示词：

```python
# Pick a funny prompt here and it will be used as the widget's default 
# when we push to the Hub in the next section
prompt = f"a photo of {name_of_your_concept} {type_of_thing} in the Acropolis"

# Tune the guidance to control how closely the generations follow the prompt
# Values between 7-11 usually work best
guidance_scale = 7

num_cols = 2
all_images = []
for _ in range(num_cols):
    images = pipe(prompt, guidance_scale=guidance_scale).images
    all_images.extend(images)

image_grid(all_images, 1, num_cols)
```

## 第 8 步：将模型推送到 Hub

若对模型满意，最后一步是推送到 Hub，并在 [DreamBooth 排行榜](https://huggingface.co/spaces/dreambooth-hackathon/leaderboard) 上查看！

首先，需要为模型仓库定义名称。默认使用唯一标识符和类别名称，你也可以按需修改：

```python
# Create a name for your model on the Hub. No spaces allowed.
model_name = f"{name_of_your_concept}-{type_of_thing}"
```

接下来，简要描述你训练的模型类型，或任何你想分享的信息：

```python
# Describe the theme and model you've trained
description = f"""
This is a Stable Diffusion model fine-tuned on `{type_of_thing}` images for the {theme} theme.
"""
```

最后，运行下面的单元格，在 Hub 上创建仓库并推送所有文件，同时附带精美的 model card：

````python
# Code to upload a pipeline saved locally to the hub
from huggingface_hub import HfApi, ModelCard, create_repo, get_full_repo_name

# Set up repo and upload files
hub_model_id = get_full_repo_name(model_name)
create_repo(hub_model_id)
api = HfApi()
api.upload_folder(folder_path=args.output_dir, path_in_repo="", repo_id=hub_model_id)

content = f"""
---
license: creativeml-openrail-m
tags:
- pytorch
- diffusers
- stable-diffusion
- text-to-image
- diffusion-models-class
- dreambooth-hackathon
- {theme}
widget:
- text: {prompt}
---

# DreamBooth model for the {name_of_your_concept} concept trained by {api.whoami()["name"]} on the {dataset_id} dataset.

This is a Stable Diffusion model fine-tuned on the {name_of_your_concept} concept with DreamBooth. It can be used by modifying the `instance_prompt`: **{instance_prompt}**

This model was created as part of the DreamBooth Hackathon 🔥. Visit the [organisation page](https://huggingface.co/dreambooth-hackathon) for instructions on how to take part!

## Description

{description}

## Usage

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained('{hub_model_id}')
image = pipeline().images[0]
image
```
"""

card = ModelCard(content)
hub_url = card.push_to_hub(hub_model_id)
print(f"Upload successful! Model can be found here: {hub_url}")
print(
    f"View your submission on the public leaderboard here: https://huggingface.co/spaces/dreambooth-hackathon/leaderboard"
)
````

## 第 9 步：庆祝 🥳

恭喜，你已经训练了第一个 DreamBooth 模型！竞赛中你可以训练任意多个模型——重要的是**获赞最多的模型将赢得奖品**，别忘了广泛分享你的作品以获得更多投票！
