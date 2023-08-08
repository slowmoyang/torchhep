from dataclasses import dataclass, asdict, astuple, fields
from pathlib import Path
import numpy as np
import torch
from torch import Tensor

def compute_data_mask(data: Tensor, length: Tensor) -> Tensor:
    mask_shape = data.shape[:-1]
    mask = torch.full(size=mask_shape, fill_value=False, dtype=torch.bool,
                      device=data.device)
    for m, l in zip(mask, length):
        m[: l].fill_(True)
    return mask

@dataclass
class TensorCollection:
    def to(self, device):
        def convert(data):
            if torch.is_tensor(data):
                data = data.to(device)
            return data
        batch = [convert(each) for each in astuple(self)]
        return self.__class__(*batch) # type: ignore

    def cpu(self):
        return self.to(torch.device('cpu'))

    def numpy(self) -> dict[str, np.ndarray]:
        return {key: value.detach().numpy()
                for key, value in asdict(self.cpu()).items()}

    def __getitem__(self, key: str) -> Tensor:
        return getattr(self, key)

    def __repr__(self):
        output = [f'{self.__class__.__name__}:']
        for key, value in asdict(self).items():
            if torch.is_tensor(value):
                output.append(f'    {key}: {value.shape}, {value.dtype}, {value.device}')
            elif isinstance(value, np.ndarray):
                output.append(f'    {key}: {value.shape}, {value.dtype}')
            else:
                output.append(f'    {key}: {type(value)}')
        output = '\n'.join(output)
        return output

    @classmethod
    @property
    def field_names(cls) -> list[str]:
        return [each.name for each in fields(cls)]

    def to_npz(self, path: str | Path):
        data = self.numpy()
        return np.savez(path, **data)

    @classmethod
    def from_npz(cls, path: str | Path):
        data = np.load(path)
        data = {key: torch.from_numpy(value) for key, value in data.items()}
        return cls(**data)

def convert_ndarray_to_tensor(arr):
    if arr.dtype == np.object_:
        return [torch.from_numpy(each) for each in arr]
    else:
        return torch.from_numpy(arr)
