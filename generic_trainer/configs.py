import collections
import dataclasses
from typing import Any, Callable, Optional, Union
import json
import os

import torch
from torch.utils.data import Dataset

from generic_trainer.metrics import *

# =============================
# Base class for all
# =============================

@dataclasses.dataclass
class OptionContainer:
    def __str__(self):
        s = ''
        for key in self.__dict__.keys():
            s += '{}: {}\n'.format(key, self.__dict__[key])
        return s

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def is_jsonable(x):
        try:
            json.dumps(x)
            return True
        except (TypeError, OverflowError):
            return False

    def get_serializable_dict(self):
        d = {}
        for key in self.__dict__.keys():
            v = self.__dict__[key]
            if not self.__class__.is_jsonable(v):
                if isinstance(v, (tuple, list)):
                    v = '_'.join([str(x) for x in v])
                else:
                    v = str(v)
            d[key] = v
        return d

    def dump_to_json(self, filename):
        try:
            f = open(filename, 'w')
            d = self.get_serializable_dict()
            json.dump(d, f)
            f.close()
        except:
            print('Failed to dump json.')

    def load_from_json(self, filename):
        """
        This function only overwrites entries contained in the JSON file. Unspecified entries are unaffected.
        """
        f = open(filename, 'r')
        d = json.load(f)
        for key in d.keys():
            self.__dict__[key] = d[key]
        f.close()


# =============================
# Model parameter classes
# =============================

@dataclasses.dataclass
class ModelParameters(OptionContainer):
    pass


# =============================
# Other parameter classes
# =============================

@dataclasses.dataclass
class ParallelizationConfig(OptionContainer):
    find_unused_parameters: bool = False
    """
    Whether to enable finding unused parameters for DistributedDataParallel. Enabling this can surpass some
    errors, but also increases the overhead. 
    """

    parallelization_type: str = 'single_node'
    """
    Type of parallel computing environment. Can be the following:
    - `'single_node'`: a single computing node with one, many, or no GPUs. This assumes only 1 process with one
                       or multiple workers for data parallelism. In this mode, model is wrapped in 
                       torch.nn.DataParallel. 
    - `'multi_node'`: multiple computing nodes with more than 1 processes. Each process should see only 1 GPU.
                      In this mode, model is wrapped in torch.distributed.DistributedDataParallel. 
                      When using this mode, it is important to limit CUDA_VISIBLE_DEVICES to 1 GPU for each local rank.
                      See
                      https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/ for more details.
    """


@dataclasses.dataclass
class LossTrackerParameters(OptionContainer):
    require_cs_labels: bool = False
    """
    When True, the loss tracker will expect CS labels as the last element in the
    label list when calculating accuracies. This is usually
    used for calculating the accuracies of CS deduced from SG predictions, without
    an actual CS classification head.
    """


# =============================
# Config classes
# =============================

@dataclasses.dataclass
class Config(OptionContainer):
    model_class: Any = None
    """The model class, which is supposed to be a subclass of torch.nn.Module."""

    model_params: ModelParameters = dataclasses.field(default_factory=ModelParameters)
    """Arguments of the model class."""

    dataset: Optional[Dataset] = None
    """The dataset object."""

    pred_names: Any = ('cs', 'eg', 'sg')
    """Names of the quantities predicted by the model."""

    debug: bool = False

    cpu_only: bool = False

    random_seed: Any = None

    parallelization_params: ParallelizationConfig = dataclasses.field(default_factory=ParallelizationConfig)

    task_type: str = 'classification'
    """
    Task type. Can be 'classification', 'regression'. Currently this only affects the logging of the loss tracker.
    """


@dataclasses.dataclass
class InferenceConfig(Config):
    # ===== PtychoNN configs =====
    batch_size: int = 1

    pretrained_model_path: str = None
    """Path to a trained model."""

    prediction_output_path: str = None
    """Path to save prediction results."""

    load_pretrained_encoder_only: bool = False
    """Keep this False for testing."""

    batch_size_per_process: int = 64
    """The batch size per process."""

    prediction_postprocessor: Optional[Callable] = None
    """
    Postprocessing function to run after prediction. Should take a tuple of predicted tensors as input and return 
    processed variables.
    """


@dataclasses.dataclass
class TrainingConfig(Config):

    training_dataset: Optional[Dataset] = None
    """
    The training dataset. If this is None, then the whole dataset (including training and validation) must be
    provided through the `dataset` parameter, and in that case the dataset will be split into training and
    validation set according to `validation_ratio` in the trainer. 
    """

    validation_dataset: Optional[Dataset] = None
    """The validation dataset. See the docstring of `training_dataset` for more details."""

    test_dataset: Optional[Dataset] = None
    """
    The test dataset. It has no influence on training, just providing a way to check test performance after each epoch.
    """

    batch_size_per_process: int = 64
    """
    The batch size per process. With this value denoted by `n_bspp`, the trainer behaves as the following:

    In single-node mode (with DataParallel), the data loader yields `n_bspp * n_gpus` at a time. Due to the 
    distributed directed by DataParallel under the hood, each GPU runs `n_bspp` samples through the model at a time.

    In multi-node mode (with DistributedDataParallel), the data loader yields `n_bspp * n_processes` samples at a time.
    Each rank takes `n_bspp` samples from it to calculate the gradient. Note that in multi-node mode, the data loaders
    are set with drop_last == True, meaning if the dataset size is not divisible by the number of processes, the
    last batch is dropped. 
    """

    num_epochs: int = 60
    """
    The number of epochs. When loading a checkpoint using `TrainingConfig.checkpoint_dir`, the epoch counter continues
    from the checkpoint, and this parameters sets the final number of epochs: for example, if the checkpoint is at 
    epoch 200 and `num_epochs` is 300, then only 100 more epochs will be run in the current job. However, if
    `TrainingConfigl.pretrained_model_path` is used instead, then the epoch counter and all the other states start from 
    scratch. 
    """

    learning_rate_per_process: float = 1e-3
    """
    The learning rate for each process. The actual learning rate is this value multiplied by the number of GPUs
    or processes.
    """

    optimizer: Any = torch.optim.Adam
    """String of optimizer name or the handle of a subclass of torch.optim.Optimizer"""

    optimizer_params: dict = dataclasses.field(default_factory=dict)
    """Optimizer parameters."""

    model_save_dir: str = '.'
    """Directory to save trained models."""

    checkpoint_dir: Any = None
    """
    The checkpoint directory. If not None, the trainer will load the checkpoint that contains the model,
    optimizer, and all the other state dictionaries from it. The given directory should be the one that contains
    "checkpoint_model.pth"; if using HuggingFaceAccelerateTrianer, this should be the directory that contains
    "checkpoint_model".
    """

    pretrained_model_path: Any = None
    """
    Path to the pretrained model. If provided, the model will be loaded for fine-tuning. This differs from
    checkpoint_dir in that it requires the path to a pth file, and that it only loads the model but not
    the states of the optimizer, etc.
    """

    load_pretrained_encoder_only: bool = False
    """If True, only the pretrained encoder (backbone) will be loaded if `pretrained_model_path` is not None."""

    validation_ratio: float = 0.1
    """Ratio of data to be used as validation set."""

    post_training_epoch_hook: Any = None
    """A Callable that can be called after each training epoch."""

    post_validation_epoch_hook: Any = None
    """A Callable that can be called after each validation epoch."""

    loss_function: Union[Callable, list[Callable, ...]] = torch.nn.CrossEntropyLoss()
    """
    The loss function. This could be either a Callable (like torch.nn.L1Loss) or a list of Callables.
    When it is a list, its length should be at least `len(pred_names)` and the Callables are respectively applied
    to predictions following the order of `pred_names`. If there are more Callables than `len(pred_names)`,
    the rest of them are treated as regularizers and are called like `func(**pred_dict)`, where `pred_dict`
    is a dictionary whose keys are elements in `pred_names` and values are the corresponding predictions.  
    """

    loss_tracker_params: LossTrackerParameters = dataclasses.field(default_factory=LossTrackerParameters)
    """Arguments of the loss tracker."""

    automatic_mixed_precision: bool = False
    """Automatic mixed precision and gradient scaling are enabled if True."""

    save_onnx: bool = False
    """If True, ONNX models are saved along with state dicts."""


@dataclasses.dataclass
class PretrainingConfig(TrainingConfig):
    loss_function: Callable = dataclasses.field(default_factory=SymmetricNegativeCosineSimilarity)
