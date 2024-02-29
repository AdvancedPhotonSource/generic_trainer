import os
import time
from functools import wraps
import random

import torch
import torch.distributed as dist
import numpy as np


def timeit(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        ret = func(*args, **kwargs)
        t1 = time.time()
        print('Delta-t for {}: {} s'.format(func.__name__, t1 - t0))
        return ret
    return wrapper


def class_timeit(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        t0 = time.time()
        ret = func(self, *args, **kwargs)
        t1 = time.time()
        print('Delta-t for {}: {} s'.format(func.__name__, t1 - t0))
        return ret
    return wrapper


def set_all_random_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_gpu_memory(show=False):
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    if show:
        print('Device 0 memory info:')
        print('    {} MB total'.format(t / 1024 ** 2))
        print('    {} MB reserved'.format(r / 1024 ** 2))
        print('    {} MB allocated'.format(a / 1024 ** 2))
    return t, r, a


def get_cuda_visible_devices_from_environ():
    envvar = os.environ['CUDA_VISIBLE_DEVICES']
    if len(envvar) == 0:
        return []
    else:
        return [int(x) for x in envvar.split(',')]


def setup_multiprocessing(rank, world_size, backend='nccl', allow_switching_backend=True):
    torch_ddp_backends = ['nccl', 'mpi', 'gloo']

    def initialize_process_group(backend, rank, world_size):
        try:
            if rank is not None:
                dist.init_process_group(backend, rank=rank, world_size=world_size)
            else:
                dist.init_process_group(backend)
        except (RuntimeError, ValueError):
            ind = torch_ddp_backends.index(backend)
            del torch_ddp_backends[ind]
            if len(torch_ddp_backends) == 0:
                raise RuntimeError('None of the backends works.')
            new_backend = torch_ddp_backends[0]
            print('Backend {} is not supported, thus switching backend to {}...'.format(backend, new_backend))
            initialize_process_group(new_backend, rank, world_size)

    if allow_switching_backend:
        initialize_process_group(backend, rank, world_size)
    else:
        if rank is not None:
            dist.init_process_group(backend, rank=rank, world_size=world_size)
        else:
            dist.init_process_group(backend)


def cleanup_multiprocessing():
    dist.destroy_process_group()
