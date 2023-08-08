import torch
from torch import Tensor
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torchhep.data.utils import compute_data_mask


class ParticleFlowMerger(nn.Module):
    def forward(self,
                jet: Tensor,
                lepton: Tensor,
                met: Tensor,
                jet_data_mask: Tensor | None = None,
                lepton_data_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        particle_flow = zip(
            self._to_list(jet, jet_data_mask),
            self._to_list(lepton, lepton_data_mask),
            met.unsqueeze(1)
        )
        particle_flow = [torch.cat(each, dim=0) for each in particle_flow]
        length = torch.tensor([each.size(0) for each in particle_flow],
                              dtype=torch.long, device=jet.device)
        particle_flow = pad_sequence(particle_flow, batch_first=True)
        data_mask = compute_data_mask(particle_flow, length)
        return particle_flow, data_mask, length


    def _to_list(self, batch: Tensor, data_mask: Tensor | None = None):
        if len(batch.shape) == 3:
            if data_mask is not None:
                # unpad
                output = [o[m] for o, m in zip(batch, data_mask)]
            else:
                output = list(batch)
        elif len(batch.shape) == 2:
            output = list(batch.unsqueeze(1))
        else:
            raise RuntimeError(batch.shape)
        return output
