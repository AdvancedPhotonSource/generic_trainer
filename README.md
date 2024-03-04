# A generic PyTorch trainer with multi-node support

This repository contains a generic PyTorch trainer that can be used for various
projects. It is designed to be model-agnostic and (maximally) task-agnostic.
Users can customize the following through the configuration interface:
- model class and parameters
- model checkpoint
- dataset object
- expected predictions
- loss functions (allows different loss functions for different predictions and Tikonov regularizers)
- parallelization (single node multiple GPUs, multinode)
- optimizer class and parameters
- Training parameters (learning rate, batch size per process, ...)

If more freedom is needed, one can also conveniently create a subclass of the trainer
and override certain methods.

## Installation

To install `generic_trainer` as a Python package, clone the GitHub
repository, then
```
pip install -e .
```

This command should automatically install the dependencies, which are
specified in `pyproject.toml`. If you prefer not to install the
dependencies, do
```
pip install --no-deps -e .
```
The `-e` flag makes the installation editable, *i.e.*, any
modifications made in the source code will be reflected when you
`import generic_trainer` in Python without reinstalling the package. 

## Usage 

In general, model training using `generic_trainer` involves the following steps:

1. Create a model that is a subclass of `torch.nn.Module`.
2. Create a model configuration class that is a subclass of `ModelParameters`, and contains the arguments of the constructor of the model class.
3. Create a dataset object that is a subclass of `torch.utils.data.Dataset` and has essential methods like `__len__`, `__getitem__`.
4. Instantiate a `TrainingConfig` object, and plug in the class handles or objects created above, along with other configurations and parameters.
5. Run the following:
```
trainer.build()
trainer.run_training()
```
Example scripts are available in `examples` and users are highly recommended to refer to them.

### Config objects

Most configurations and parameters are passed to the trainer through the `OptionContainer`
objects defined in `configs.py`. `TrainingConfig` is the main config object for training. 
Some of its fields accept other config objects. For example, model parameters are provided by
passing an object of a subclass of `ModelParameters` to `TrainingConfig.model_params`; 
also, parallelization options are provided by passing a object of `ParallelizationConfig` to
`TrainingConfig.parallelization_params`.

### Model

The model definition should be given to the trainer through `TrainingConfig.model_class`
and `TrainingConfig.model_params`. The former should be the handle of a subclass of
`torch.nn.Module`, and the latter should be an object of a subclass of `ModelParameters`.
The fields of the model parameter config object must match the arguments in the `__init__`
method of the model class.

### Dataset

Data are passed to the trainer through `TrainingConfig.dataset`. This field expect an
object of a subclass of `torch.utils.data.Dataset`. Please refer to
[PyTorch's official documentation](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
for a guide on creating custom dataset classes for your own data.

The provided dataset is assumed to contain both training and validation data. Inside the
trainer, it will be randomly split into a training dataset and validation dataset. To
control the ratio of train-validation separation, use `TrainingConfig.validation_ratio`.

### Optimizer

One can specify an optimizer by passing the class handle of the desired optimizer to
`TrainingConfig.optimizer`, and its arguments (if any) to `TrainingConfig.optimizer_params`
as a `dict`. The learning rate should NOT be included in the optimizer parameter object, as
it is set elsewhere. 

For example, the following code instructs the trainer to use `AdamW` with `weight_decay=0.01`:
```
configs = TrainingConfig(
    ...
    optimizer=torch.optim.AdamW,
    optimizer_params={'weight_decay': 0.01},
    ...
)
```

### Prediction names

The trainer needs to know the names and orders of the model's predictions. These are set
through `TrainingConfig.pred_names` as a list or tuple of strings, with each string being
the name of a prediction. The names could be anything as long as they suggest the nature
of the prediction. The length of list or tuple is more important, as it tells the trainer
how many model predictions to expect. 

### Loss functions

Loss function can be customized through `TrainingConfig.loss_function`. This field takes
either a single, or a list/tuple of Callables that have a signature of 
`loss_func(preds, labels)`. When using loss functions from Pytorch,
they should be instantiated objects instead of the class handles (*e.g.*, `nn.CrossEntropyLoss()`
instead of `nn.CrossEntropyLoss`, because the Callable that has the required signature is
the `forward` method of the object). 

Currently, additioanl arguments to the loss function
is not allowed, but one can create a loss function class subclassing `nn.Module`, and
set extra arguments through its constructor. For example, to set the weight to a particular
loss function, one can create the loss as
```
class MyLoss(nn.Module):
    def __init__(weight=0.01):
        self.weight = weight
        
    def forward(preds, labels):
        return self.weight * torch.mean((preds - labels) ** 2)
```

When a list or tuple of Callables is passed to `loss_function`, it uses the loss functions
respectively for each prediction defined in `TrainingConfig.pred_names`. If there
are more loss functions than the number of `pred_names`, the rest are treated as regularizers
and they should have a signature of `loss_func(pred1, pred2, ...)`. When encountering these
loss functions, the trainer would first try calling them with keyword arguments 
`loss_func(pred_name_1=pred_1, pred_name_2=pred_2, ...)`. This is done in case the function's arguments
come in a different order from the predictions. If the argument names do not match, it would then
pass the predictions as positional arguments like `loss_func(pred_1, pred_2, ...)`.

### Parallelization

The trainer should work with either single-node (default) or multi-node parallelization.
To run multi-node training, one should create a `ParallelizationConfig` object, set
`parallelization_type` to `'multi_node''`, then pass the config object to 
`TrainingConfig.parallelization_params`.

When `parallelization_type` is `single_node`, the trainer wraps the model 
object with `torch.nn.DataParallel`,
allowing it to use all GPUs available on a single machine. 

If `parallelization_type` is set to `multi_node`, the trainer instead wraps the model
object with `torch.nn.parallel.DistributedDataParallel` (DDP), which should allow it to
work with multiple processes that are potentially distributed over multiple nodes on an HPC.

In order to run multi-node training on an HPC like ALCF's Polaris, one should
launch multiple processes when submitting the job to the HPC's job scheduler. 
PyTorch DDP's offcial documentation says jobs should be launched using `torchrun` in this case,
but it was found that in some cases, 
jobs with `torchrun` would run into GPU visibility-related exception
when using the NCCL backend. Instead, we figured out that the job may also be launched with
`aprun`, the standard multi-processing run command with Cobalt (Theta) or PBS (Polaris) scheduler.
Some environment variables need to be set in the Python script, as shown in the 
[ALCF training material repository](https://github.com/argonne-lcf/ai-science-training-series/tree/13bd951ca01dd432f4939c309834252de2a493e9/06_distributedTraining/DDP). 
An example of multi-node training on Polaris
is available in `examples/hpc/ddp_training_with_dummy_data.py`. 