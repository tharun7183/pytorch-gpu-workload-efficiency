
import torch
from contextlib import nullcontext

def apply_channels_last(model):
    return model.to(memory_format=torch.channels_last)

def maybe_compile(model, enabled: bool):
    # torch.compile is available in PyTorch 2.x. If it fails on a given GPU, fall back gracefully.
    if enabled and hasattr(torch, "compile"):
        try:
            return torch.compile(model)
        except Exception:
            return model
    return model

def autocast_ctx(enabled: bool):
    if torch.cuda.is_available():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return nullcontext()
