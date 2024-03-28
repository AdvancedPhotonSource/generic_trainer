"""
This script demonstrates the use of the trainer on an HPC (Polaris) with multi-node training enabled
through `ParallelizationConfig`.
"""

import dataclasses
from typing import Any
import os

# This should come before import torch.
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['PMI_LOCAL_RANK']

import socket

import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
from mpi4py import MPI

from generic_trainer.trainer import Trainer
from generic_trainer.configs import *
import generic_trainer.message_logger

from ..dataset_handle import DummyClassificationDataset


class ClassificationModel(nn.Module):
    def __init__(self, dim_input=128, dim_hidden=256, num_classes=(5, 7), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feature_extractor = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU()
        )
        self.head1 = nn.Sequential(
            nn.Linear(dim_hidden, num_classes[0]),
            nn.Softmax(dim=1)
        )
        self.head2 = nn.Sequential(
            nn.Linear(dim_hidden, num_classes[1]),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        y1 = self.head1(x)
        y2 = self.head2(x)
        return y1, y2

def print_multiproc(s):
    rank = os.environ["RANK"]
    local_rank = os.environ['PMI_LOCAL_RANK']
    print('[Rank {} Local rank {}] {}'.format(rank, local_rank, s))


@dataclasses.dataclass
class ClassificationModelParameters(ModelParameters):
    dim_input: int = 128,
    dim_hidden: int = 512,
    num_classes: Any = (5, 7)


if __name__ == '__main__':
    #################################
    # Preparation
    #################################
    # Set envvars
    local_rank = os.environ['PMI_LOCAL_RANK']
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(size)

    print_multiproc('Pytorch library: '.format(torch.__file__))
    print_multiproc('CUDA_VISIBLE_DEVICES: {}'.format(os.environ['CUDA_VISIBLE_DEVICES']))
    print_multiproc('cuda.is_available: {}; device_count: {}.'.format(torch.cuda.is_available(), torch.cuda.device_count()))
    print_multiproc('Host: {}'.format(os.environ['HOSTNAME']))

    # It will want the master address too, which we'll broadcast:
    if rank == 0:
        master_addr = socket.gethostname()
    else:
        master_addr = None

    master_addr = MPI.COMM_WORLD.bcast(master_addr, root=0)
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(2345)
    print_multiproc('MASTER_ADDR = {}.'.format(os.environ['MASTER_ADDR']))

    print_multiproc('Initializing process groups...')
    # setup_multiprocessing(backend='nccl', allow_switching_backend=False)
    dist.init_process_group('nccl', init_method='env://')

    print_multiproc('Rank index from dist.get_rank(): {}'.format(dist.get_rank()))

    #################################
    # Training and model parameters
    #################################
    dataset = DummyClassificationDataset(assumed_array_shape=(40, 128), label_dims=(5, 7), add_channel_dim=False)

    configs = TrainingConfig(
        parallelization_params=ParallelizationConfig(
            parallelization_type='multi_node',  # This is set to enable DDP.
            find_unused_parameters=False
        ),
        model_class=ClassificationModel,
        model_params=ClassificationModelParameters(
            dim_input=128,
            dim_hidden=256,
            num_classes=(5, 7)
        ),
        pred_names=('pred1', 'pred2'),
        dataset=dataset,
        batch_size_per_process=2,
        learning_rate_per_process=1e-2,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=torch.optim.AdamW,
        optimizer_params={'weight_decay': 0.01},
        num_epochs=5,
        model_save_dir='temp',
        task_type='classification'
    )

    trainer = Trainer(configs)
    trainer.build()
    trainer.run_training()

    trainer.cleanup()
