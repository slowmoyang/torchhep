from torch import Tensor

def reduce_loss(input: Tensor, reduction: str):
    if reduction == 'mean':
        return input.mean()
    elif reduction == 'sum':
        return input.sum()
    elif reduction == 'none':
        return input
    else:
        raise RuntimeError(f'unknown {reduction=}')
