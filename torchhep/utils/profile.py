import torch.nn as nn

def count_params(module: nn.Module) -> int:
    return sum(each.numel() for each in module.parameters())
