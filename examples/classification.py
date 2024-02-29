import dataclasses
from typing import Any

import torch
import torch.nn as nn

from generic_trainer.trainer import Trainer
from generic_trainer.configs import *

from dataset_handle import DummyClassificationDataset


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


@dataclasses.dataclass
class ClassificationModelParameters(ModelParameters):
    dim_input: int = 128,
    dim_hidden: int = 512,
    num_classes: Any = (5, 7)


if __name__ == '__main__':
    dataset = DummyClassificationDataset(assumed_array_shape=(40, 128), label_dims=(5, 7), add_channel_dim=False)

    configs = TrainingConfig(
        model_class=ClassificationModel,
        model_params=ClassificationModelParameters(
            dim_input=128,
            dim_hidden=256,
            num_classes=(5, 7)
        ),
        parallelization_params=ParallelizationConfig(
            parallelization_type='single_node'
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
