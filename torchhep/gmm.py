import torch
from torch import Tensor
import torch.distributions as D


def build_gmm(params: Tensor, num_components: int):
    num = num_components
    dim = (params.size(1) - num) // (2 * num)

    loc_start = num
    loc_stop = num * (1 + dim)

    logits = params[:, : loc_start]
    loc = params[:, loc_start: loc_stop].reshape(-1, num, dim)
    scale = params[:, loc_stop: ].exp().reshape(-1, num, dim)

    mixture = D.Categorical(logits=logits)
    base = D.Normal(loc=loc, scale=scale)
    component = D.Independent(base, reinterpreted_batch_ndims=1)
    return D.MixtureSameFamily(mixture, component)


def get_mode(gmm: D.MixtureSameFamily) -> Tensor:
    # mode index
    idx_batch = gmm.mixture_distribution.probs.argmax(dim=1)
    loc_batch = gmm.component_distribution.base_dist.loc
    mode = [loc[idx] for idx, loc in zip(idx_batch, loc_batch)]
    return torch.stack(tensors=mode, dim=0)
