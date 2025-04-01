import copy
import collections
import dataclasses
from typing import *
import json
import os
import re
import importlib

import torch
from torch.utils.data import Dataset

from generic_trainer.metrics import *


def remove(*exclusions):
    """
    Remove fields from a dataclass. Note: if you use this on an inherited dataclass, 
    isinstance(this_object, BaseClass) will return False. To avoid this, use a
    class factory:
    ```
    def factory(base, name, exclusions):
        new_fields = [(i.name, i.type, i) for i in fields(base) if i.name not in exclusions]
        return make_dataclass(name, new_fields)

    B = factory(base=A, name='B', exclusions=('b', 'c'))
    ```
    See https://stackoverflow.com/questions/69289547/how-to-remove-dynamically-fields-from-a-dataclass.
    """
    def wrapper(cls):
        new_fields = [(i.name, i.type, i) for i in dataclasses.fields(cls) if i.name not in exclusions]
        return dataclasses.make_dataclass(cls.__name__, new_fields)
    return wrapper

# =============================
# Base class for all
# =============================

@dataclasses.dataclass
class OptionContainer:

    class SkipKey:
        pass

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.globals = {}

    def __str__(self):
        s = ''
        for key in self.__dict__.keys():
            s += '{}: {}\n'.format(key, self.__dict__[key])
        return s

    def __repr__(self):
        return self.__str__()

    @staticmethod
    def remove(*fields):
        def _(cls):
            fields_copy = copy.copy(cls.__dataclass_fields__)
            annotations_copy = copy.deepcopy(cls.__annotations__)
            for field in fields:
                del fields_copy[field]
                del annotations_copy[field]
            d_cls = dataclasses.make_dataclass(cls.__name__, annotations_copy)
            d_cls.__dataclass_fields__ = fields_copy
            return d_cls
        return _

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
            v = self.object_to_string(key, v)
            d[key] = v
        return d

    def deserizalize_dict(self, d):
        for key in d.keys():
            v = self.string_to_object(key, d[key])
            if not isinstance(v, self.SkipKey):
                self.__dict__[key] = v

    def dump_to_json(self, filename):
        try:
            f = open(filename, 'w')
            d = self.get_serializable_dict()
            json.dump(d, f, indent=4, separators=(',', ': '))
            f.close()
        except:
            print('Failed to dump json.')

    def load_from_json(self, filename, namespace=None):
        """
        This function only overwrites entries contained in the JSON file. Unspecified entries are unaffected.
        """
        if namespace is not None:
            for key in namespace.keys():
                globals()[key] = namespace[key]
        f = open(filename, 'r')
        d = json.load(f)
        self.deserizalize_dict(d)
        f.close()

    def string_to_object(self, key, value):
        """
        Create an object based on the string value of a config key.
        
        :param key: str.
        :param value: str.
        :return: object.
        """
        # Value is a class handle
        if isinstance(value, dict):
            # Only convert the dict to an OptionContainer object if they are supposed to. Otherwise, leave it as a dict.
            if 'model_params' in key or key in ['loss_tracker_params', 'parallelization_params']:
                assert 'config_class' in value.keys(), ('The value of {} is supposed to be an object of a subclass '
                                                        'of OptionContainer, but I cannot find the '
                                                        'class name.'.format(key))
                try:
                    config = globals()[value['config_class']]()
                    config.deserizalize_dict(value)
                    value = config
                except KeyError as e:
                    raise ModuleNotFoundError(
                        "When loading {} from JSON, the following error occurred when attempting to create the "
                        "OptionContainer object for it:'\n{}\n"
                        "To create an OptionContainer object, its class name must be in the global namespace. You can "
                        "import the proper classes in your driver script using from ... import ..., and pass "
                        "globals() to load_from_json:\n"
                        "    configs.load_from_json(filename, namespace=globals())\n".format(key, e)
                    )
        elif key == 'config_class':
            return self.SkipKey()
        elif isinstance(value, (list, tuple)):
            value = [self.string_to_object(key, v) for v in value]
        elif isinstance(value, str) and (res := re.match(r"<class '(.+)'>", value)):
            class_import_path = res.groups()[0].split('.')
            value = getattr(importlib.import_module('.'.join(class_import_path[:-1])), class_import_path[-1])
        elif value in ['True', 'False']:
            value = True if value == 'True' else False
        elif isinstance(value, (int, float, bool, dict)):
            value = value
        else:
            for caster in (int, float):
                try:
                    value = caster(value)
                    break
                except (ValueError, TypeError):
                    pass
        return value

    def object_to_string(self, key, value):
        """
        Convert an object in a config key to string.
        :param key: str.
        :param value: object.
        :return: str.
        """
        if isinstance(value, OptionContainer):
            config_class_name = value.__class__.__name__
            value = value.get_serializable_dict()
            value['config_class'] = config_class_name
        elif isinstance(value, (dict, int, float, bool)):
            value = value
        elif isinstance(value, (tuple, list)):
            value = [self.object_to_string(key, x) for x in value]
        elif value is None:
            value = None
        else:
            value = str(value)
        return value


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

    pred_names_and_types: Tuple[Tuple[str, str], ...] = (('cs', 'cls'), ('eg', 'cls'), ('sg', 'cls'))
    """
    Names and types of the quantities predicted by the model. It should be a tuple of 2-tuples.
    The first element of each sub-tuple is the name of the prediction, and the second element is
    its type, which can be one of:
    - 'cls': a classification prediction.
    - 'regr': a regression prediction. 
    """
    
    dataset_creator_name: Optional[str] = None
    """
    Name of the data creator function. This parameter is not used by the trainer itself, but the training
    script can use this parameter to locate a custom data creator function that creates the training,
    validation, and test sets. 
    """
    
    data_label_separation_index: Optional[int] = 1
    """
    The index of label in the returned tuple of the dataset object. All elements at and after this index
    are assumed to be labels, while those before this index are assumed to be data.
    """

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

    def string_to_object(self, key, value):
        value = super().string_to_object(key, value)
        if key == 'model_save_dir':
            self.pretrained_model_path = os.path.join(value, 'best_model.pth')
        if key == 'pretrained_model_path' and value is None:
            value = self.pretrained_model_path
        return value


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

    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam
    """The optimizer class. Should be given as the handle of a subclass of torch.optim.Optimizer."""

    optimizer_params: dict = dataclasses.field(default_factory=dict)
    """Optimizer parameters."""

    multi_optimizer_param_dicts: Optional[Sequence[Dict]] = None
    """
    The optimizer uses different learning rates for different parameters if this is provided.
    It should be a list of dictionaries as described in 
    https://pytorch.org/docs/stable/optim.html#per-parameter-options.
    However, the code to get trainable parameters in the "params" keys should be given
    as a string, where the model object should be referenced as `self.get_model_object()`.
    """
    
    scheduler: Type[torch.optim.lr_scheduler._LRScheduler] = torch.optim.lr_scheduler.CyclicLR
    """
    The scheduler class. Should be given as the handle of a subclass of torch.optim.lr_scheduler._LRScheduler.
    """

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

    post_test_epoch_hook: Any = None
    """A Callable that can be called after each test epoch."""

    loss_function: Union[Callable, list[Callable, ...]] = torch.nn.CrossEntropyLoss()
    """
    The loss function. This could be either a Callable (like torch.nn.L1Loss) or a list of Callables.
    When it is a list, its length should be at least `len(pred_names_and_types)` and the Callables are respectively applied
    to predictions following the order of `pred_names_and_types`. If there are more Callables than `len(pred_names)`,
    the rest of them are treated as regularizers and are called like `func(**pred_dict)`, where `pred_dict`
    is a dictionary whose keys are elements in `pred_names_and_types` and values are the corresponding predictions.  
    """

    loss_tracker_params: LossTrackerParameters = dataclasses.field(default_factory=LossTrackerParameters)
    """Arguments of the loss tracker."""
    
    num_workers: int = 0
    """Number of workers for dataloader."""
    
    pin_memory_for_dataloader: bool = True
    """If True, dataloader will put fetched data tensor in pinned memory, which accelerates training."""

    automatic_mixed_precision: bool = False
    """Automatic mixed precision and gradient scaling are enabled if True."""

    save_onnx: bool = False
    """If True, ONNX models are saved along with state dicts."""

    curriculum_learning_rate: float = 0.
    """A value between 0 and 1. If nonzero, the rate (in fraction of batch per epoch) at which the predicted labels
    should be used over the true ones to determine the regression head to use for branched regression. All true labels
    will always be used in the first epoch, and subsequently, this fraction will be reduced by the curriculum learning
    rate at each epoch until only predicted labels are used.
    """

    def string_to_object(self, key, value):
        value = super().string_to_object(key, value)
        if key == 'loss_function' and not isinstance(value, (list, tuple)):
            try:
                value = eval(value)
            except Exception as e:
                raise ModuleNotFoundError(
                    "When loading loss_function from JSON, the following error occurred:'\n{}\n"
                    "To create a loss function object, its class name must be in the global namespace. You can "
                    "import the proper classes in your driver script using from ... import ..., and pass "
                    "globals() to load_from_json:\n"
                    "    configs.load_from_json(filename, namespace=globals())\n".format(e)
                )
        return value

@dataclasses.dataclass
class PretrainingConfig(TrainingConfig):
    loss_function: Callable = dataclasses.field(default_factory=SymmetricNegativeCosineSimilarity)
