import dataclasses
from typing import Any

import torch
import torch.nn as nn

from generic_trainer.trainer import *
from generic_trainer.configs import *

from dataset_handle import DummyClassificationDataset
from classification import ClassificationModel, ClassificationModelParameters


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

    trainer = HuggingFaceAccelerateTrainer(configs)
    trainer.build()
    trainer.run_training()

    trainer.cleanup()
