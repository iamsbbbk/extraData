import torch
from diffusers import StableDiffusionPipeline

repo_id = "sd-legacy/stable-diffusion-v1-5"  # 或 "runwayml/stable-diffusion-v1-5"（可能需要同意协议）
save_dir = "./sd15"

pipe = StableDiffusionPipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.float32,   # Windows/CPU 稳妥；显存紧张可改 torch.float16
)

pipe.save_pretrained(save_dir)
print("saved to:", save_dir)
