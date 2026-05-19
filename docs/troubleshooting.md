# 常见问题排查

这份文档覆盖运行 `Diffusion Zero to Hero 中文实战课` 时最常见的问题。扩散模型 notebook 对 GPU、依赖版本、网络和 Hugging Face 登录状态都比较敏感，遇到问题时建议先从这里排查。

## 安装与环境

### `ModuleNotFoundError`

先确认已经在仓库根目录安装依赖：

```bash
pip install -r requirements.txt
```

如果你在 Colab、Kaggle 或 Studio Lab 中运行，建议在 notebook 开头重新安装核心库：

```python
%pip install -U diffusers transformers accelerate datasets huggingface_hub
```

### PyTorch 无法使用 GPU

检查：

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
```

如果返回 `False`，通常是 CUDA 版本、驱动或 PyTorch 安装包不匹配。优先按照 PyTorch 官网给出的命令安装与你机器匹配的版本。

## 显存不足

常见报错：

```text
CUDA out of memory
```

可尝试：

- 降低 `batch_size`
- 降低 `image_size` 或生成分辨率
- 使用 `torch_dtype=torch.float16`
- 开启 `pipe.enable_attention_slicing()`
- 使用更少 inference steps
- 关闭不必要的输出图像网格和日志记录
- DreamBooth 场景优先考虑 LoRA 等低显存微调方法

Stable Diffusion 推理通常建议 8 GB 以上显存；DreamBooth 全量微调建议 14 GB 以上，24 GB 会更稳。

## Hugging Face 登录

如果遇到模型下载权限问题，先登录：

```bash
huggingface-cli login
```

或在 notebook 中运行：

```python
from huggingface_hub import notebook_login
notebook_login()
```

注意：

- 不要把 token 写进 notebook 或 `.py` 文件。
- 不要提交 `.env`。
- 如果 notebook UI 提示 token 可能以明文保存，请登录后清理 notebook 输出与 widget metadata。

## 模型下载慢或失败

可尝试：

- 确认网络能访问 Hugging Face Hub。
- 提前在命令行登录。
- 重试下载，Hub 偶发失败很常见。
- 在云端 notebook 中尽量避免反复重启运行时。
- 如果使用镜像或代理，请确认环境变量只保存在本地，不要提交。

## 图片缺失

如果 README 或 notebook 中的图片显示失败，请确认 `images/` 目录存在：

```text
images/unit1/
images/unit2/
images/unit3/
images/unit4/
images/hackathon/
```

所需图片清单见 [assets.md](assets.md)。其中 `unit3/image.png` 和 `unit3/mask.png` 不只是展示图片，部分 inpainting 示例会直接读取它们。

## Notebook 运行顺序

Jupyter notebook 通常依赖前面 cell 创建的变量。若出现变量未定义：

- 从头按顺序运行。
- 重启 kernel 后重新运行全部 cell。
- 不要跳过安装、导入、模型加载和 scheduler 初始化的 cell。

## `xformers` 相关问题

`xformers` 可降低显存，但安装与 CUDA 版本强相关。如果安装失败，不必阻塞学习，可先使用：

```python
pipe.enable_attention_slicing()
```

或降低分辨率和 batch size。

## W&B 登录问题

部分训练脚本使用 Weights & Biases 记录实验。如果你不想登录，可在脚本中关闭或改为离线模式：

```bash
WANDB_MODE=offline python unit2/finetune_model.py ...
```

或者在 notebook 中跳过日志相关步骤。

## 生成结果不好

常见原因：

- inference steps 太少
- CFG scale 不合适
- prompt 不够具体或负向提示词过强
- scheduler 不适合当前模型
- seed 固定导致探索不足
- 微调数据太少、太杂或 caption 不一致

建议记录每次实验的 prompt、negative prompt、seed、scheduler、steps、CFG scale、分辨率和模型版本。

## 发布仓库前的 notebook 清理

如果你维护仓库，请在提交前清理 notebook：

```bash
python - <<'PY'
import json
from pathlib import Path

for path in Path('.').rglob('*.ipynb'):
    nb = json.loads(path.read_text())
    nb['metadata'].pop('widgets', None)
    for cell in nb.get('cells', []):
        cell['execution_count'] = None
        cell['outputs'] = []
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\\n')
PY
```

清理后再运行密钥扫描，确认没有 token 或个人信息残留。
