import wandb
import numpy as np
import torch
import torchvision
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm
from fastcore.script import call_parse
from torchvision import transforms
from diffusers import DDPMPipeline, DDIMScheduler
from datasets import load_dataset


@call_parse
def train(
    image_size=256,
    batch_size=16,
    grad_accumulation_steps=2,
    num_epochs=1,
    start_model="google/ddpm-bedroom-256",
    dataset_name="huggan/wikiart",
    device="cuda",
    model_save_name="wikiart_1e",
    wandb_project="dm_finetune",
    log_samples_every=250,
    save_model_every=2500,
):
    """Fine-tune a pretrained DDPM pipeline on a Hugging Face dataset.

    This script is adapted from the upstream Hugging Face Diffusion Models
    Course and kept locally so the Chinese course can be run standalone.
    """

    # Initialize W&B for experiment logging.
    wandb.init(project=wandb_project, config=locals())

    # Prepare pretrained model.
    image_pipe = DDPMPipeline.from_pretrained(start_model)
    image_pipe.to(device)

    # Get a scheduler for sampling validation images.
    sampling_scheduler = DDIMScheduler.from_config(start_model)
    sampling_scheduler.set_timesteps(num_inference_steps=50)

    # Prepare dataset.
    dataset = load_dataset(dataset_name, split="train")
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            clean_images = batch["images"].to(device)

            # Forward diffusion: add randomly sampled noise at random timesteps.
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Train the UNet to predict the added noise.
            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]
            loss = F.mse_loss(noise_pred, noise)
            wandb.log({"loss": loss.item()})

            loss.backward()

            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % log_samples_every == 0:
                x = torch.randn(8, 3, image_size, image_size).to(device)
                for _, t in tqdm(enumerate(sampling_scheduler.timesteps)):
                    model_input = sampling_scheduler.scale_model_input(x, t)
                    with torch.no_grad():
                        noise_pred = image_pipe.unet(model_input, t)["sample"]
                    x = sampling_scheduler.step(noise_pred, t, x).prev_sample

                grid = torchvision.utils.make_grid(x, nrow=4)
                im = grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5
                im = Image.fromarray(np.array(im * 255).astype(np.uint8))
                wandb.log({"Sample generations": wandb.Image(im)})

            if (step + 1) % save_model_every == 0:
                image_pipe.save_pretrained(f"{model_save_name}step_{step + 1}")

        scheduler.step()

    image_pipe.save_pretrained(model_save_name)
    wandb.finish()
