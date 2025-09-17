import torch

try:
    import torch_musa

    torch.cuda = torch.musa
    _has_musa = True
except ImportError:
    _has_musa = False
    pass


def get_device(index=None):
    def device_str(prefix):
        return f"{prefix}:{index}" if index is not None else prefix

    if _has_musa:
        return device_str("musa")

    return device_str("cuda")


def get_communication_backend():
    if _has_musa:
        return "mccl"

    return "nccl"


def is_sm10x():
    return torch.cuda.get_device_capability() >= (10, 0)


def is_hopper():
    return torch.cuda.get_device_capability() == (9, 0)
