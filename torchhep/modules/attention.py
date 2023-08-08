import torch
from torch import Tensor
from torch import nn


class CrossAttention(nn.Module):

    def __init__(self,
                 target_dim: int,
                 source_dim: int,
                 num_heads: int = 8,
                 dropout_prob: float = 0.0
    ) -> None:
        super().__init__()

        self.query_proj = nn.Linear(target_dim, target_dim)
        self.key_proj = nn.Linear(source_dim, target_dim)
        self.value_proj = nn.Linear(source_dim, target_dim)
        self.output_proj = nn.Linear(target_dim, target_dim)
        self.attention_dropout = nn.Dropout(dropout_prob)
        self.output_dropout = nn.Dropout(dropout_prob)

        self.num_heads = num_heads

        self.register_buffer(
            name='neg_inf',
            tensor=torch.tensor(float('-inf'))
        )

    def forward(self,
                target: Tensor,
                source: Tensor,
                attention_mask: Tensor | None = None,
                target_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            target: a tensor with the shape of (N, T, E)
            source: a tensor with the shape of (N, S, E)
            attention_mask: a tensor with the shape of (N, T, S)
        Returns:
            output: a tensor with the shape of (N, T, E)

        where N is the batch size, T is the target array length, E is the
        embedding dimension, S is the source array length
        """
        N, T, E = target.size()
        S = source.size(1)
        H = self.num_heads
        D = E // H

        q = self.query_proj(target)
        k = self.key_proj(source)
        v = self.value_proj(source)

        # (N, L, E) -view-> (N, L, H, D) -transpose-> (N, H, L, D)
        # where L is T or S.
        q = q.view(N, T, H, D).transpose(1, 2)
        k = k.view(N, S, H, D).transpose(1, 2)
        v = v.view(N, S, H, D).transpose(1, 2)

        # attention weight matrix
        ## (N, H, T, D) @ (N, H, D, S) -> (N, H, T, S)
        a = q @ k.transpose(2, 3)
        ## scaling
        a = D**-0.5 * a
        if attention_mask is not None:
            a = a.where(attention_mask.unsqueeze(1), self.neg_inf)
        a = a.softmax(dim=-1)
        a = self.attention_dropout(a)

        # output
        ## (N, H, T, S) @ (N, H, S, D) -> (N, H, T, D)
        o = a @ v
        ## (N, H, T, D) -transpose-> (N, T, H, D) -view-> (N, T, D)
        o = o.transpose(1, 2).contiguous().view(N, T, E)
        o = self.output_proj(o)
        o = self.output_dropout(o)
        if target_mask is not None:
            o = o.where(target_mask.unsqueeze(2), 0)
        return o


class SelfAttention(CrossAttention):

    def __init__(self,
                 input_dim: int,
                 num_heads: int = 8,
                 dropout_prob: float = 0.0
    ) -> None:
        super().__init__(
            target_dim=input_dim,
            source_dim=input_dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob
        )

    def forward(self,
                input: Tensor,
                input_mask: Tensor | None = None
    ) -> Tensor:
        return super().forward(
            target=input,
            source=input,
            attention_mask=None,
            target_mask=input_mask
        )

