from torch import Tensor
from torch.nn.modules.loss import _Loss
from torch.distributions import Distribution
from torchhep.losses.utils import reduce_loss

def negative_log_likelihood(input: Distribution,
                            target: Tensor,
                            reduction: str = 'mean'
) -> Tensor:
    output = -input.log_prob(target)
    return reduce_loss(output, reduction)


class NegativeLogLikelihood(_Loss):
    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__(reduction=reduction)

    def forward(self, input: Distribution, target: Tensor) -> Tensor:
        return negative_log_likelihood(input, target, reduction=self.reduction)
