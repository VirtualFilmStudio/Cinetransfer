import torch

def move_to(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: move_to(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [move_to(x, device) for x in obj]
    return obj

def get_device(i=0):
    device = f"cuda:{i}" if torch.cuda.is_available() else "cpu"
    return torch.device(device)