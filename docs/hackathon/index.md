# DreamBooth 黑客松 🏆

📣 **黑客松已结束，获奖名单已在 Discord 公布。当前目录主要作为 DreamBooth 练习项目保留：你仍可训练模型并上传到 Hub，但不再发放奖品或证书。**

欢迎参加 DreamBooth 黑客松！这是一场社区活动：你将**用少量自己的图像微调 Stable Diffusion，从而个性化模型。** 技术名为 [_DreamBooth_](https://arxiv.org/abs/2208.12242)，可将主体（如宠物或最爱的一道菜）「植入」模型输出域，在提示词中使用**唯一标识符**即可合成该主体。

竞赛包含 5 个**主题**，每个主题对应以下类别之一：

* **动物 🐨：** 让你的宠物或最喜欢的动物出现在雅典卫城、游泳或飞向太空等场景。  
* **科学 🔬：** 生成星系、蛋白质等自然与医学科学领域的合成图像。  
* **美食 🍔：** 在你最爱的一道菜或菜系上微调 Stable Diffusion。  
* **风景 🏔：** 生成你最喜欢的山、湖或花园等风景。  
* **Wildcard 🔥：** 自由选择类别，尽情发挥！

历史活动中，每个主题点赞数最高的前 3 名模型会获得奖品。今天学习时，更建议把这些主题当作练习任务：准备一组自己的图片、训练一个可复现模型、记录参数与失败样例。

## 如何参与

1. 加入 [Hugging Face Discord](https://huggingface.co/join/discord)，关注 `#dreambooth-hackathon` 频道获取动态。  
2. 打开并运行 [DreamBooth 笔记本](dreambooth.md)（本仓库中文版）。在 Colab / Kaggle 等平台请选择 **GPU** 运行时。

| 笔记本 | 本地文件 |
|:-------|:---------|
| DreamBooth 训练 | [dreambooth.md](dreambooth.md) |

也可通过 [官方仓库](https://github.com/huggingface/diffusion-models-class/blob/main/hackathon/dreambooth.md) 在 Colab / Kaggle / Gradient / Studio Lab 打开英文环境。

**说明 👋：** 官方笔记本默认微调 [`CompVis/stable-diffusion-v1-4`](https://huggingface.co/CompVis/stable-diffusion-v1-4)。你可改用其他 SD 检查点，只需调整加载组件与安全检查器的代码。例如：

* [`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)  
* [`prompthero/openjourney`](https://huggingface.co/prompthero/openjourney)  
* [`stabilityai/stable-diffusion-2`](https://huggingface.co/stabilityai/stable-diffusion-2)  
* [`hakurei/waifu-diffusion`](https://huggingface.co/hakurei/waifu-diffusion)  
* [`stabilityai/stable-diffusion-2-1`](https://huggingface.co/stabilityai/stable-diffusion-2-1)  
* [`nitrosocke/elden-ring-diffusion`](https://huggingface.co/nitrosocke/elden-ring-diffusion)  

## 评选与排行榜

要参与评奖（已结束），需将 DreamBooth 模型推送到 Hub，并在模型卡片中加入 `dreambooth-hackathon` 标签（[示例](https://huggingface.co/lewtun/ccorgi-dog/blob/main/README.md#L9)）。[DreamBooth 笔记本](dreambooth.md) 会自动创建该标签；若使用自有脚本需自行添加。

模型按**点赞数**排名，可在排行榜查看：

* [DreamBooth Leaderboard](https://huggingface.co/spaces/dreambooth-hackathon/leaderboard)

## 时间线（历史记录）

* **2022 年 12 月 21 日** — 开始  
* **2022 年 12 月 31 日** — Colab Pro 注册截止  
* **2023 年 1 月 22 日** — 最终提交截止（排行榜关闭）  
* **2023 年 1 月 23–27 日** — 各主题获奖公布  

除非另有说明，截止时间为对应日期的 23:59 UTC。

## 奖品（历史记录）

每个主题按排行榜**点赞最多**的前 3 名：

**第 1 名**

* 1 年 [Hugging Face Pro](https://huggingface.co/pricing)，或 [HF 周边商店](https://store.huggingface.co/) 100 美元代金券  

**第 2 名**

* [_NLP with Transformers_](https://transformersbook.com/) 图书，或 50 美元周边代金券  

**第 3 名**

* 1 个月 Hugging Face Pro，或 15 美元周边代金券  

当时向黑客松提交至少 1 个 DreamBooth 模型的参与者还可获得**完成证书** 🔥。

## 算力支持（历史记录）

Google Colab 曾为活动提供 Colab Pro 额度（随机 100 名，2023 年 1 月发放，2022 年 12 月 31 日前注册）。注册表单：[链接](https://docs.google.com/forms/d/e/1FAIpQLSeE_js5bxq_a_nFTglbZbQqjd6KNDD9r4YRg42kDFGSb5aoYQ/viewform)。

## 常见问题

### 微调可以使用什么数据？

可使用属于你本人、或宽松许可证允许使用的图像。若提交人脸模型（如 Wildcard），建议使用自己的肖像。尽量使用个人数据（宠物、本地风景等），往往更受欢迎 😁。

### 是否允许 Textual Inversion 等其他微调方法？

可以！本黑客松聚焦 DreamBooth，但欢迎尝试其他微调技术与框架，只要能为社区带来有趣模型 🥰。

## 作为练习项目时的建议

- 数据集控制在 10-30 张高质量图片，主体清晰、背景多样。
- 使用唯一标识符时避免常见词，降低与已有概念冲突的概率。
- 训练前先记录基础模型、学习率、步数、resolution、prompt 模板。
- 训练后不要只展示成功样例，也保留失败样例，用来判断过拟合、主体漂移或提示词不服从。
- 如果显存不足或模型文件太大，后续优先尝试 LoRA 版本的 DreamBooth。

DreamBooth 是理解个性化微调的经典入口；真实项目中常会结合 LoRA、ControlNet、IP-Adapter 或 SDXL 工作流来获得更低成本和更强控制。
