from typing import Any
import torch.nn as nn
from torchhep.modules.activation import BertGELU
from torchhep.modules.residual import AdditiveResidual

class MLP(nn.Sequential):

    def __init__(self,
                 input_dim: int,
                 output_dim: int | None = None,
                 widening_factor: int = 4,
                 dropout_prob: float = 0.0
    ) -> None:
        output_dim = output_dim or input_dim
        hidden_dim = widening_factor * input_dim
        super().__init__(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            BertGELU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_prob)
        )


class MLPBlock(nn.Sequential):
    def __init__(self,
                 input_dim: int,
                 num_layers: int,
                 widening_factor: int = 4,
                 dropout_prob: float = 0.1,
    ):
        super().__init__()
        mlp_kwargs: dict[str, Any] = dict(
            input_dim=input_dim,
            widening_factor=widening_factor,
            dropout_prob=dropout_prob
        )
        for _ in range(num_layers):
            self.append(AdditiveResidual(MLP(**mlp_kwargs)))
