"""Verify ML imports and a tiny tensor computation without downloading models."""

import importlib

import torch


for name in (
    "torchvision", "transformers", "diffusers", "accelerate", "safetensors",
    "datasets", "huggingface_hub", "cv2", "pandas", "matplotlib", "skimage",
    "scipy", "einops", "yaml", "websockets",
):
    importlib.import_module(name)
    print(f"import {name}: OK")

print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
devices = [f"cuda:{index}" for index in range(torch.cuda.device_count())] or ["cpu"]
for device in devices:
    tensor = torch.ones((4, 4), device=device)
    assert (tensor @ tensor).sum().item() == 64.0
    print(f"{device}: matrix multiply OK")
    del tensor
    if device != "cpu":
        torch.cuda.empty_cache()
