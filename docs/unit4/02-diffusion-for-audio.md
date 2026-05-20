<!-- This page is generated from the matching notebook by scripts/notebook_to_docs.py. -->

> 原始 Notebook：[unit4/02_diffusion_for_audio.ipynb](https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/unit4/02_diffusion_for_audio.ipynb)

# 音频扩散

在本笔记本中，我们将简要了解如何使用扩散模型生成音频。

## 你将学到：
- 计算机如何表示音频
- 在原始音频数据与频谱图之间转换的方法
- 如何使用自定义 collate 函数准备 DataLoader，将音频切片转换为频谱图
- 在特定音乐流派上微调现有音频扩散模型
- 将自定义管道上传到 Hugging Face Hub

说明：本笔记本主要用于教学——不保证我们的模型听起来很好 😉。

让我们开始吧！

## 环境准备与导入

```python
import os
from dotenv import load_dotenv
load_dotenv("../.env", override=True)
```

```text
True
```

```python
import torch, random
import numpy as np
import torch.nn.functional as F
from tqdm.auto import tqdm
from IPython.display import Audio
from matplotlib import pyplot as plt
from diffusers import DiffusionPipeline
from torchaudio import transforms as AT
from torchvision import transforms as IT
```

## 从预训练音频管道采样

首先按照 [Audio Diffusion 文档](https://huggingface.co/docs/diffusers/api/pipelines/audio_diffusion) 加载一个已有的音频扩散模型管道：

```python
# Load a pre-trained audio diffusion pipeline
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-instrumental-hiphop-256").to(device)
```

```text
Downloading (incomplete total...): 0.00B [00:00, ?B/s]
```

```text
Fetching 5 files:   0%|          | 0/5 [00:00<?, ?it/s]
```

```text
Loading pipeline components...:   0%|          | 0/3 [00:00<?, ?it/s]
```

```text
An error occurred while trying to fetch /home2/lihaoyu/Projects/Diffuser/diffusion-course/data/hub/models--teticio--audio-diffusion-instrumental-hiphop-256/snapshots/a63b0c7e794925f74f021d356c13bce47cb69264/unet: Error no file named diffusion_pytorch_model.safetensors found in directory /home2/lihaoyu/Projects/Diffuser/diffusion-course/data/hub/models--teticio--audio-diffusion-instrumental-hiphop-256/snapshots/a63b0c7e794925f74f021d356c13bce47cb69264/unet.
Defaulting to unsafe serialization. Pass `allow_pickle=False` to raise an error instead.
Expected types for unet: (<class 'diffusers.models.unets.unet_2d_condition.UNet2DConditionModel'>,), got <class 'diffusers.models.unets.unet_2d.UNet2DModel'>.
```

与前面单元使用的管道一样，我们可以像这样调用管道来生成样本：

```python
# Sample from the pipeline and display the outputs
output = pipe()
display(output.images[0])
display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))
```

这里，`rate` 参数指定音频的*采样率*；我们稍后会深入讨论。你还会注意到管道返回了多个结果。这是怎么回事？让我们仔细看看两个输出。

第一个是数据数组，表示生成的音频：

```python
# The audio array
output.audios[0].shape
```

第二个看起来像灰度图像：

```python
# The output image (spectrogram)
output.images[0].size
```

这提示了我们该管道的工作方式。音频并非直接用扩散生成——相反，管道拥有与 [Unit 1](https://github.com/huggingface/diffusion-models-class/tree/main/unit1) 中见到的无条件图像生成管道相同类型的 2D UNet，用于生成频谱图，再后处理为最终音频。

管道有一个额外组件处理这些转换，可通过 `pipe.mel` 访问：

```python
pipe.mel
```

## 从音频到图像再转回来

音频「波形」随时间编码原始音频样本——例如，这可能是麦克风接收到的电信号。处理这种「时域」表示可能比较棘手，因此常见做法是先转换为其他形式，通常称为频谱图。频谱图显示不同频率的强度（y 轴）随时间（x 轴）的变化：

```python
# Calculate and show a spectrogram for our generated audio sample using torchaudio
spec_transform = AT.Spectrogram(power=2)
spectrogram = spec_transform(torch.tensor(output.audios[0]))
print(spectrogram.min(), spectrogram.max())
log_spectrogram = spectrogram.log()
plt.imshow(log_spectrogram[0], cmap='gray');
```

刚生成的频谱图数值在 0.0000000000001 到 1 之间，大多数接近该范围的下端。这对可视化或建模并不理想——事实上我们必须对这些值取对数，才能得到能显示细节的灰度图。因此，我们通常使用一种特殊的频谱图，称为 Mel 频谱图，它通过对信号不同频率分量施加变换，旨在捕捉对人耳听觉重要的信息。

![torchaudio docs diagram](https://raw.githubusercontent.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/master/images/unit4/torchaudio_feature_extractions.png)
_来自 [torchaudio 文档](https://pytorch.org/audio/stable/transforms.html) 的部分音频变换_

幸运的是，我们甚至不必太担心这些变换——管道的 `mel` 功能会为我们处理这些细节。利用它，我们可以像这样将频谱图图像转换为音频：

```python
a = pipe.mel.image_to_audio(output.images[0])
a.shape
```

我们也可以先将原始音频数据加载，再调用 `audio_slice_to_image()` 函数，将音频数据数组转换为频谱图图像。较长的片段会自动切分为合适长度的块，以生成 256x256 的频谱图图像：

```python
pipe.mel.load_audio(raw_audio=a)
im = pipe.mel.audio_slice_to_image(0)
im
```

音频表示为一个很长的数字数组。要播放出来，还需要一个关键信息：采样率。我们用多少个样本（单个数值）来表示一秒的音频？

可以通过以下方式查看该管道训练时使用的采样率：

```python
sample_rate_pipeline = pipe.mel.get_sample_rate()
sample_rate_pipeline
```

若采样率指定错误，得到的音频会过快或过慢：

```python
display(Audio(output.audios[0], rate=44100)) # 2x speed
```

## 微调管道

现在我们对管道的工作方式有了大致了解，让我们在一些新音频数据上微调它！

该数据集是不同流派的音频片段集合，可以像这样从 Hub 加载：

```python
from datasets import load_dataset
dataset = load_dataset('lewtun/music_genres', split='train')
dataset
```

可以使用下面的代码查看数据集中的不同流派及每个流派包含的样本数量：

```python
for g in list(set(dataset['genre'])):
  print(g, sum(x==g for x in dataset['genre']))
```

数据集中的音频以数组形式存储：

```python
audio_array = dataset[0]['audio']['array']
sample_rate_dataset = dataset[0]['audio']['sampling_rate']
print('Audio array shape:', audio_array.shape)
print('Sample rate:', sample_rate_dataset)
display(Audio(audio_array, rate=sample_rate_dataset))
```

注意该音频的采样率更高——若要使用现有管道，需要对其进行「重采样」以匹配。片段也比管道预设的更长。幸运的是，使用 `pipe.mel` 加载音频时，会自动将片段切分为更小的部分：

```python
a = dataset[0]['audio']['array'] # Get the audio array
pipe.mel.load_audio(raw_audio=a) # Load it with pipe.mel
pipe.mel.audio_slice_to_image(0) # View the first 'slice' as a spectrogram
```

需要记得调整采样率，因为该数据集的每秒样本数是原来的两倍：

```python
sample_rate_dataset = dataset[0]['audio']['sampling_rate']
sample_rate_dataset
```

这里使用 torchaudio 的变换（导入为 AT）进行重采样，使用管道的 `mel` 将音频转为图像，使用 torchvision 的变换（导入为 IT）将图像转为张量。这样就得到一个函数，可将音频片段转为可用于训练的频谱图张量：

```python
resampler = AT.Resample(sample_rate_dataset, sample_rate_pipeline, dtype=torch.float32)
to_t = IT.ToTensor()

def to_image(audio_array):
  audio_tensor = torch.tensor(audio_array).to(torch.float32)
  audio_tensor = resampler(audio_tensor)
  pipe.mel.load_audio(raw_audio=np.array(audio_tensor))
  num_slices = pipe.mel.get_number_of_slices()
  slice_idx = random.randint(0, num_slices-1) # Pic a random slice each time (excluding the last short slice)
  im = pipe.mel.audio_slice_to_image(slice_idx) 
  return im
```

我们将 `to_image()` 函数作为自定义 collate 函数的一部分，把数据集变成可用于训练的 DataLoader。collate 函数定义如何将数据集的一批样本转换为最终可用于训练的数据批次。这里我们将每个音频样本转为频谱图图像，并将得到的张量堆叠在一起：

```python
def collate_fn(examples):
  # to image -> to tensor -> rescale to (-1, 1) -> stack into batch
  audio_ims = [to_t(to_image(x['audio']['array']))*2-1 for x in examples]
  return torch.stack(audio_ims)

# Create a dataset with only the 'Chiptune / Glitch' genre of songs
batch_size = 4 # 4 on colab, 12 on A100
chosen_genre = 'Electronic' # <<< Try training on different genres <<<
indexes = [i for i, g in enumerate(dataset['genre']) if g == chosen_genre]
filtered_dataset = dataset.select(indexes)
dl = torch.utils.data.DataLoader(filtered_dataset.shuffle(), batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
batch = next(iter(dl))
print(batch.shape)
```

**注意：除非 GPU 显存充足，否则需要使用较小的 batch size（例如 4）。**

## 训练循环

下面是一个简单的训练循环，在 DataLoader 上运行若干 epoch 以微调管道的 UNet。也可以跳过此单元格，用下一单元格中的代码加载管道。

```python
epochs = 3
lr = 1e-4

pipe.unet.train()
pipe.scheduler.set_timesteps(1000)
optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=lr)

for epoch in range(epochs):
    for step, batch in tqdm(enumerate(dl), total=len(dl)):
        
        # Prepare the input images
        clean_images = batch.to(device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, pipe.scheduler.num_train_timesteps, (bs,), device=clean_images.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        noisy_images = pipe.scheduler.add_noise(clean_images, noise, timesteps)

        # Get the model prediction
        noise_pred = pipe.unet(noisy_images, timesteps, return_dict=False)[0]

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward(loss)

        # Update the model parameters with the optimizer
        optimizer.step()
        optimizer.zero_grad()
```

```python
# OR: Load the version I trained earlier
pipe = DiffusionPipeline.from_pretrained("johnowhitaker/Electronic_test").to(device)
```

```python
output = pipe()
display(output.images[0])
display(Audio(output.audios[0], rate=22050))
```

```python
# Make a longer sample by passing in a starting noise tensor with a different shape
noise = torch.randn(1, 1, pipe.unet.sample_size[0], pipe.unet.sample_size[1]*4).to(device)
output = pipe(noise=noise)
display(output.images[0])
display(Audio(output.audios[0], rate=22050))
```

输出听起来未必惊艳，但这是一个开始 :)尝试调整学习率和 epoch 数，并在 Discord 上分享你的最佳结果，我们一起改进！

一些值得思考的方向：
- 我们使用的是 256px 方形频谱图，这限制了 batch size。能否从 128x128 频谱图恢复足够质量的音频？
- 我们用每次选取音频片段不同切片的方式代替随机图像增强，在训练很多 epoch 时，是否可以用其他增强方式改进？
- 还能如何用这种方法生成更长的片段？也许可以先生成 5 秒的起始片段，再用受 inpainting 启发的思路继续生成衔接的后续音频段……
- 在频谱图扩散的语境下，image-to-image 的等价物是什么？

## 推送到 Hub

对模型满意后，可以保存并推送到 Hub，供他人使用：

```python
from huggingface_hub import get_full_repo_name, HfApi, create_repo, ModelCard
```

```python
# Pick a name for the model
model_name = "audio-diffusion-electronic"
hub_model_id = get_full_repo_name(model_name)
```

```python
# Save the pipeline locally
pipe.save_pretrained(model_name)
```

```python
# Inspect the folder contents
!ls {model_name}
```

```python
# Create a repository
create_repo(hub_model_id)
```

```python
# Upload the files
api = HfApi()
api.upload_folder(
    folder_path=f"{model_name}/scheduler", path_in_repo="scheduler", repo_id=hub_model_id
)
api.upload_folder(
    folder_path=f"{model_name}/mel", path_in_repo="mel", repo_id=hub_model_id
)
api.upload_folder(folder_path=f"{model_name}/unet", path_in_repo="unet", repo_id=hub_model_id)
api.upload_file(
    path_or_fileobj=f"{model_name}/model_index.json",
    path_in_repo="model_index.json",
    repo_id=hub_model_id,
)
```

```python
# Push a model card
content = f"""
---
license: mit
tags:
- pytorch
- diffusers
- unconditional-audio-generation
- diffusion-models-class
---

# Model Card for Unit 4 of the [Diffusion Models Class 🧨](https://github.com/huggingface/diffusion-models-class)

This model is a diffusion model for unconditional audio generation of music in the genre {chosen_genre}

## Usage

<pre>
from IPython.display import Audio
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("{hub_model_id}")
output = pipe()
display(output.images[0])
display(Audio(output.audios[0], rate=pipe.mel.get_sample_rate()))
</pre>
"""

card = ModelCard(content)
card.push_to_hub(hub_model_id)
```

## 总结

希望本笔记本让你对音频生成的潜力有了初步感受。请查看本单元介绍中链接的一些参考资料，了解更高级的方法及其令人惊艳的生成样本！
