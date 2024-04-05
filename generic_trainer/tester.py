import os

import torch
from torch.utils.data import Dataset, DataLoader

import generic_trainer.trainer as trainer
from generic_trainer.configs import *


class Tester(trainer.Trainer):

    def __init__(self, configs: InferenceConfig):
        super().__init__(configs, skip_init=True)
        self.rank = 0
        self.num_processes = 1
        self.gatekeeper = None
        self.dataset = None
        self.sampler = None
        self.dataloader = None
        self.parallelization_type = self.configs.parallelization_params.parallelization_type

    def build(self):
        self.build_ranks()
        self.build_scalable_parameters()
        self.build_device()
        self.build_dataset()
        self.build_dataloaders()
        self.build_model()
        self.build_dir()

    def build_scalable_parameters(self):
        self.all_proc_batch_size = self.configs.batch_size_per_process * self.num_processes

    def build_dataset(self):
        self.dataset = self.configs.dataset

    def build_dataloaders(self):
        if self.parallelization_type == 'multi_node':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,
                                                                           num_replicas=self.num_processes,
                                                                           rank=self.rank,
                                                                           drop_last=False,
                                                                           shuffle=False)
            self.dataloader = DataLoader(self.dataset,
                                         batch_size=self.configs.batch_size_per_process,
                                         sampler=self.sampler,
                                         collate_fn=lambda x: x)
        else:
            self.dataloader = DataLoader(self.dataset, shuffle=False,
                                         batch_size=self.all_proc_batch_size,
                                         collate_fn=lambda x: x, worker_init_fn=self.get_worker_seed_func(),
                                         generator=self.get_dataloader_generator(), num_workers=0,
                                         drop_last=False)

    def build_dir(self):
        if self.gatekeeper.should_proceed(gate_kept=True):
            if not os.path.exists(self.configs.prediction_output_path):
                os.makedirs(self.configs.prediction_output_path)
        self.barrier()

    def run(self):
        self.model.eval()
        for j, data_and_labels in enumerate(self.dataloader):
            data, _ = self.process_data_loader_yield(data_and_labels)
            preds = self.model(*data)
            self.save_predictions(preds)

    def save_predictions(self, preds):
        pass
