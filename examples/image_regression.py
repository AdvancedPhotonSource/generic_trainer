import dataclasses
from typing import Any

import torch
import torch.nn as nn

from generic_trainer.trainer import Trainer
from generic_trainer.configs import *
from generic_trainer.metrics import *

from dataset_handle import DummyImageDataset


class ImageRegressionModel(nn.Module):
    def __init__(self, num_channels_list=(4, 8, 16), kernel_size_list=(3, 3, 3), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels_list = num_channels_list
        self.kernel_size_list = kernel_size_list
        self.encoder = self.get_encoder()
        self.decoder1 = self.get_decoder()
        self.decoder2 = self.get_decoder()

    def get_encoder(self):
        net = []
        last_nc = 1
        for i, n_c in enumerate(self.num_channels_list):
            net.append(nn.Conv2d(last_nc, n_c, kernel_size=self.kernel_size_list[i],
                                 padding=self.kernel_size_list[i] // 2))
            net.append(nn.ReLU())
            net.append(nn.MaxPool2d(2, 2))
            last_nc = n_c
        return nn.Sequential(*net)

    def get_decoder(self):
        net = []
        last_nc = self.num_channels_list[-1]
        for i, n_c in enumerate(list(self.num_channels_list[::-1][1:]) + [1]):
            net.append(nn.Conv2d(last_nc, n_c, kernel_size=self.kernel_size_list[::-1][i],
                                 padding=self.kernel_size_list[::-1][i] // 2))
            net.append(nn.ReLU())
            net.append(nn.Upsample(scale_factor=2))
            last_nc = n_c
        return nn.Sequential(*net)

    def forward(self, x):
        x = self.encoder(x)
        y1 = self.decoder1(x)
        y2 = self.decoder2(x)
        return y1, y2


@dataclasses.dataclass
class ImageRegressionModelParameters(ModelParameters):
    num_channels_list: Any = (4, 8, 16)
    kernel_size_list: Any = (3, 3, 3)


if __name__ == '__main__':
    dataset = DummyImageDataset(assumed_array_shape=(40, 1, 64, 64), label_shapes=((1, 64, 64), (1, 64, 64)))

    configs = TrainingConfig(
        model_class=ImageRegressionModel,
        model_params=ImageRegressionModelParameters(
            num_channels_list=(4, 8, 16),
            kernel_size_list=(5, 3, 3),
        ),
        parallelization_params=ParallelizationConfig(
            parallelization_type='single_node'
        ),
        pred_names=('mag', 'phase'),
        dataset=dataset,
        batch_size_per_process=2,
        learning_rate_per_process=1e-2,
        loss_function=(nn.L1Loss(), nn.L1Loss(), TotalVariationLoss(weight=1000)),
        optimizer=torch.optim.AdamW,
        optimizer_params={'weight_decay': 0.01},
        num_epochs=5,
        model_save_dir='temp',
        task_type='regression'
    )

    trainer = Trainer(configs)
    trainer.build()
    trainer.run_training()

    trainer.cleanup()
