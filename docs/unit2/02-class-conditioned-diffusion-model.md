<!-- This page is generated from the matching notebook by scripts/notebook_to_docs.py. -->

> 原始 Notebook：[unit2/02_class_conditioned_diffusion_model_example.ipynb](https://github.com/HaoyuLi-Nova/Diffusion-Zero-to-Hero/blob/master/unit2/02_class_conditioned_diffusion_model_example.ipynb)

# 构建类别条件扩散模型
在本笔记本中，我们将演示向扩散模型添加条件信息的一种方法。具体而言，我们将在 MNIST 上训练一个类别条件扩散模型，延续[第一单元中的「从零开始」示例](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02-diffusion-models-from-scratch.md)，在推理时可以指定希望模型生成哪个数字。
如本单元简介所述，这只是向扩散模型添加额外条件信息的众多方法之一，因其相对简单而被选中。与第一单元的「从零开始」笔记本类似，本笔记本主要用于说明，若你愿意可以安全跳过。

## 环境准备与数据预处理

```python
%pip install -q diffusers
```

```python
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')
```

```python
# Load the dataset
dataset = torchvision.datasets.MNIST(root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor())

# Feed it into a dataloader (batch size 8 here just for demo)
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# View some examples
x, y = next(iter(train_dataloader))
print('Input shape:', x.shape)
print('Labels:', y)
plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys');
```

## 创建类别条件 UNet
我们将按以下方式注入类别条件：
- 创建标准的 `UNet2DModel`，并增加一些额外的输入通道
- 通过嵌入层将类别标签映射为形状为 `(class_emb_size)` 的学习向量
- 用 `net_input = torch.cat((x, class_cond), 1)` 将此信息作为额外通道拼接到 UNet 内部输入
- 将此 `net_input`（总共 `class_emb_size+1` 个通道）送入 UNet 得到最终预测
本示例将 `class_emb_size` 设为 4，但这完全任意——你可以尝试设为 1（看是否仍有效）、10（与类别数匹配），或用类别标签的简单 one-hot 编码直接替换学习的 `nn.Embedding`。
实现如下：

```python
class ClassConditionedUnet(nn.Module):
  def __init__(self, num_classes=10, class_emb_size=4):
    super().__init__()
    
    # The embedding layer will map the class label to a vector of size class_emb_size
    self.class_emb = nn.Embedding(num_classes, class_emb_size)

    # Self.model is an unconditional UNet with extra input channels to accept the conditioning information (the class embedding)
    self.model = UNet2DModel(
        sample_size=28,           # the target image resolution
        in_channels=1 + class_emb_size, # Additional input channels for class cond.
        out_channels=1,           # the number of output channels
        layers_per_block=2,       # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64), 
        down_block_types=( 
            "DownBlock2D",        # a regular ResNet downsampling block
            "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
            "AttnDownBlock2D",
        ), 
        up_block_types=(
            "AttnUpBlock2D", 
            "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
            "UpBlock2D",          # a regular ResNet upsampling block
          ),
    )

  # Our forward method now takes the class labels as an additional argument
  def forward(self, x, t, class_labels):
    # Shape of x:
    bs, ch, w, h = x.shape
    
    # class conditioning in right shape to add as additional input channels
    class_cond = self.class_emb(class_labels) # Map to embedding dimension
    class_cond = class_cond.view(bs, class_cond.shape[1], 1, 1).expand(bs, class_cond.shape[1], w, h)
    # x is shape (bs, 1, 28, 28) and class_cond is now (bs, 4, 28, 28)

    # Net input is now x and class cond concatenated together along dimension 1
    net_input = torch.cat((x, class_cond), 1) # (bs, 5, 28, 28)

    # Feed this to the UNet alongside the timestep and return the prediction
    return self.model(net_input, t).sample # (bs, 1, 28, 28)
```

若对形状或变换感到困惑，可添加 print 语句显示相关形状，确认是否符合预期。我也在部分中间变量上标注了形状，希望能让流程更清晰。

## 训练与采样
此前我们会写 `prediction = unet(x, t)`，现在在训练时添加正确的标签作为第三个参数（`prediction = unet(x, t, y)`），推理时可传入任意标签，若一切顺利，模型应生成与之匹配的图像。此处的 `y` 是 MNIST 数字标签，取值 0 到 9。
训练循环与[第一单元示例](https://github.com/huggingface/diffusion-models-class/blob/unit2/unit1/02-diffusion-models-from-scratch.md)非常相似。我们现在预测噪声（而非第一单元中的去噪图像），以匹配默认 DDPMScheduler 在训练时加噪和推理时生成样本所期望的目标。训练需要一段时间——加速可作为有趣的小项目，但大多数人可能只需浏览代码（乃至整个笔记本）而无需运行，因为我们只是在说明一个概念。

```python
# Create a scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
```

```python
#@markdown Training loop (10 Epochs):

# Redefining the dataloader to set the batch size higher than the demo of 8
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

# How many runs through the data should we do?
n_epochs = 10

# Our network 
net = ClassConditionedUnet().to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.Adam(net.parameters(), lr=1e-3) 

# Keeping a record of the losses for later viewing
losses = []

# The training loop
for epoch in range(n_epochs):
    for x, y in tqdm(train_dataloader):
        
        # Get some data and prepare the corrupted version
        x = x.to(device) * 2 - 1 # Data on the GPU (mapped to (-1, 1))
        y = y.to(device)
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, 999, (x.shape[0],)).long().to(device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)

        # Get the model prediction
        pred = net(noisy_x, timesteps, y) # Note that we pass in the labels y

        # Calculate the loss
        loss = loss_fn(pred, noise) # How close is the output to the noise

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())

    # Print out the average of the last 100 loss values to get an idea of progress:
    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')

# View the loss curve
plt.plot(losses)
```

训练完成后，我们可以传入不同标签作为条件来采样一些图像：

```python
#@markdown Sampling some different digits:

# Prepare random x to start from, plus some desired labels y
x = torch.randn(80, 1, 28, 28).to(device)
y = torch.tensor([[i]*8 for i in range(10)]).flatten().to(device)

# Sampling loop
for i, t in tqdm(enumerate(noise_scheduler.timesteps)):

    # Get model pred
    with torch.no_grad():
        residual = net(x, t, y)  # Again, note that we pass in our labels y

    # Update sample with step
    x = noise_scheduler.step(residual, t, x).prev_sample

# Show the results
fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.imshow(torchvision.utils.make_grid(x.detach().cpu().clip(-1, 1), nrow=8)[0], cmap='Greys')
```

就是这样！我们现在可以对生成的图像施加一定控制了。
希望你喜欢这个示例。一如既往，欢迎在 Discord 中提问。

```python
# Exercise (optional): Try this with FashionMNIST. Tweak the learning rate, batch size and number of epochs.
# Can you get some decent-looking fashion images with less training time than the example above?
```
