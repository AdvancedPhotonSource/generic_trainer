import logging
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

import generic_trainer.trainer as trainer
from generic_trainer.configs import *
from generic_trainer.inference_util import *


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
        self.mode = 'state_dict'

        # Attributes below are used for ONNX
        self.onnx_mdl = None

        self.trt_hin = None
        self.trt_din = None
        self.trt_hout = None
        self.trt_dout = None

        self.trt_engine = None
        self.trt_stream = None
        self.trt_context = None
        self.context = None

    def build(self):
        self.build_ranks()
        self.build_scalable_parameters()
        self.build_device()
        self.build_dataset()
        self.build_dataloaders()
        self.build_model()
        self.build_dir()

    def build_model(self):
        if self.configs.pretrained_model_path.endswith('onnx'):
            logging.info('An ONNX model is given. This model will be loaded and run with TensorRT.')
            self.build_onnx_model()
            self.mode = 'onnx'
        else:
            super().build_model()

    def build_onnx_model(self):
        import pycuda.autoinit
        self.context = pycuda.autoinit.context
        self.onnx_mdl = self.configs.pretrained_model_path
        self.trt_engine = engine_build_from_onnx(self.onnx_mdl)
        self.trt_hin, self.trt_hout, self.trt_din, self.trt_dout, self.trt_stream = mem_allocation(self.trt_engine)
        self.trt_context = self.trt_engine.create_execution_context()

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
        if self.mode == 'state_dict':
            self.model.eval()
        for j, data_and_labels in enumerate(tqdm.tqdm(self.dataloader)):
            data, labels = self.process_data_loader_yield(data_and_labels)
            if self.mode == 'state_dict':
                preds = self.model(*data)
            else:
                preds = self.run_onnx_inference(*data)
            self.update_result_holders(preds, labels)

    def run_onnx_inference(self, data):
        data = data.cpu().numpy()
        orig_shape = data.shape
        np.copyto(self.trt_hin, data.astype(np.float32).ravel())
        pred = np.array(inference(self.trt_context, self.trt_hin, self.trt_hout,
                                  self.trt_din, self.trt_dout, self.trt_stream))

        pred = pred.reshape(orig_shape)
        return pred

    def update_result_holders(self, preds, *args, **kwargs):
        pass
