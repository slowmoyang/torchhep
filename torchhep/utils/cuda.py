import torch
import json


def is_gpu_idle(device) -> bool:
    output = torch.cuda.list_gpu_processes(device)
    output = output.split('\n')
    assert len(output) >= 2, '\n'.join(output)
    assert output[0].startswith('GPU:')

    if len(output) > 2:
        return False
    else:
        return output[1] == 'no processes are running'


def list_gpus(idle: bool = True,
              as_idx: bool = True,
) -> list[int] | list[torch.device]:
    # TODO allow_busy=False
    device_count = torch.cuda.device_count()
    gpu_list = [torch.device(f'cuda:{idx}') for idx in range(device_count)]
    if idle:
        gpu_list = [each for each in gpu_list if is_gpu_idle(each)]
    if as_idx:
        return [each.index for each in gpu_list]
    else:
        return gpu_list

def select_idle_gpu(as_idx: bool = True):
    gpus = list_gpus(as_idx=as_idx)
    #assert len(gpus) > 0
    idle_gpu = gpus[0] if len(gpus) > 0 else -1
    # TODO warning
    return idle_gpu


def save_cuda_memory_stats(device, path):
    if device.type != 'cuda':
        raise ValueError(f'got the wrong type of devicee: {device}')
    data = torch.cuda.memory_stats(device)
    with open(path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
