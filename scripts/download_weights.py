import torch
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "digiplay/Photon_v1",
    torch_dtype=torch.float16,
    variant="fp16",
)

pipe.save_pretrained("/src/photon-cache")