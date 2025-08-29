import torch
from typing import Dict

def get_device(config: Dict) -> torch.device:
    """Determine the device to use for computation."""
    if config["device"]["use_cuda"] and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def move_to_device(model, device: torch.device, use_half: bool = False):
    """Move the model to the specified device and optionally cast to float16.

    Args:
        model: The torch.nn.Module or diffusers Pipeline-like object supporting .to.
        device: Target device.
        use_half: If True and device is CUDA, cast weights to float16.
    """
    model = model.to(device)
    if device.type == "cuda" and use_half:
        try:
            model = model.half()
        except Exception:
            pass
    return model
