import math
import torch
import torch.nn as nn
from torch import Tensor

@torch.jit.script # type: ignore
def bert_gelu(x):
    """
    ref. https://github.com/google-research/bert/blob/master/modeling.py#L264-L277
    """

    cdf = 0.5 * (1.0 + torch.tanh(
        (math.sqrt(2 / torch.pi) * (x + 0.044715 * torch.pow(x, 3)))))
    return x * cdf


class BertGELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return bert_gelu(input)
