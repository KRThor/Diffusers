from dataclasses import dataclass
from datasets import load_dataset
from torchvision import transforms
import torch
from diffusers import UNet2DModel
from PIL import Image
from diffusers import DDPMScheduler
import torch.nn.functional as F
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMPipeline
import math
import os
from accelerate import notebook_launcher
from accelerate import Accelerator
from huggingface_hub import whoami, HfFolder, Repository
from pathlib import Path
from tqdm.auto import tqdm
import json
from torch.utils.data import Dataset


@dataclass
class TrainingConfig:
    image_size = 128  # 생성되는 이미지 해상도 8x39
    train_batch_size = 16
    eval_batch_size = 16  # 평가 동안에 샘플링할 이미지 수
    num_epochs = 500
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = "fp16"  # `no`는 float32, 자동 혼합 정밀도를 위한 `fp16`
    output_dir = "sd-bearing-model"  # 로컬 및 HF Hub에 저장되는 모델명
    seed = 0

config = TrainingConfig()

preprocess = transforms.Compose(
    [
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

class BearingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # metadata.jsonl 파일 읽기
        self.metadata = []
        with open(os.path.join(data_dir, "metadata.jsonl"), "r") as f:
            for line in f:
                self.metadata.append(json.loads(line))
                
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        item = self.metadata[idx]
        image_path = os.path.join(self.data_dir, item["file_name"])
        image = Image.open(image_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return {"images": image}

# 데이터셋과 DataLoader 설정
dataset = BearingDataset(data_dir="./data_dir", transform=preprocess)
train_dataloader = torch.utils.data.DataLoader(
    dataset, 
    batch_size=config.train_batch_size, 
    shuffle=True
)

model = UNet2DModel(
    sample_size=config.image_size,  # 타겟 이미지 해상도
    in_channels=3,  # 입력 채널 수, RGB 이미지에서 3
    out_channels=3,  # 출력 채널 수
    layers_per_block=2,  # UNet 블럭당 몇 개의 ResNet 레이어가 사용되는지
    block_out_channels=(128, 128, 256, 256, 512, 512),  # 각 UNet 블럭을 위한 출력 채널 수
    down_block_types=(
        "DownBlock2D",  # 일반적인 ResNet 다운샘플링 블럭
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # spatial self-attention이 포함된 일반적인 ResNet 다운샘플링 블럭
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # 일반적인 ResNet 업샘플링 블럭
        "AttnUpBlock2D",  # spatial self-attention이 포함된 일반적인 ResNet 업샘플링 블럭
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
)

sample_image = dataset[0]["images"].unsqueeze(0)
print("Input shape:", sample_image.shape)

print("Output shape:", model(sample_image, timestep=0).sample.shape)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid

def evaluate(config, epoch, pipeline):
    # 랜덤한 노이즈로 부터 이미지를 추출합니다.(이는 역전파 diffusion 과정입니다.)
    # 기본 파이프라인 출력 형태는 `List[PIL.Image]` 입니다.
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images

    # 이미지들을 그리드로 만들어줍니다.
    image_grid = make_grid(images, rows=4, cols=4)

    # 이미지들을 저장합니다.
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # accelerator 초기화 부분 수정
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs")  # logging_dir 대신 project_dir 사용
    )
    
    if accelerator.is_main_process:
        if config.push_to_hub:
            repo_name = get_full_repo_name(Path(config.output_dir).name)
            repo = Repository(config.output_dir, clone_from=repo_name)
        elif config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")

    # 모든 것이 준비되었습니다.
    # 기억해야 할 특정한 순서는 없으며 준비한 방법에 제공한 것과 동일한 순서로 객체의 압축을 풀면 됩니다.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # 이제 모델을 학습합니다.
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # 이미지에 더할 노이즈를 샘플링합니다.
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # 각 이미지를 위한 랜덤한 타임스텝(timestep)을 샘플링합니다.
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device
            ).long()

            # 각 타임스텝의 노이즈 크기에 따라 깨끗한 이미지에 노이즈를 추가합니다.
            # (이는 foward diffusion 과정입니다.)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # 노이즈를 반복적으로 예측합니다.
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # 각 에포크가 끝난 후 evaluate()와 몇 가지 데모 이미지를 선택적으로 샘플링하고 모델을 저장합니다.
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=True)
                else:
                    pipeline.save_pretrained(config.output_dir)

# notebook_launcher 부분을 제거하고 직접 train_loop 호출
if __name__ == "__main__":
    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델을 GPU로 이동
    model.to(device)

    args = (config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    train_loop(*args)  