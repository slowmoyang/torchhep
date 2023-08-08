import torch.nn as nn
from torch import Tensor

class AdditiveResidual(nn.Module):

    def __init__(self, residual_mapping: nn.Module):
        super().__init__()
        self.residual_mapping = residual_mapping

    def forward(self, input: Tensor) -> Tensor:
        return input + self.residual_mapping(input)
