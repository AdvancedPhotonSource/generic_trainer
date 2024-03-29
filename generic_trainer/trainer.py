import logging
import os
from typing import Optional, Any
import itertools
import copy
import re
import warnings

try:
    from mpi4py import MPI
except:
    MPI = None
    warnings.warn('Since mpi4py is not installed, some multi-node features might be not available.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

from .configs import *
from .util import *
from .compat import *


class MultirankGateKeeper:
    """
    A gatekeeper class that determines if a routine should be executed with the current rank.
    """
    def __init__(self, rank, num_ranks):
        self.rank = rank
        self.num_ranks = num_ranks

    def should_proceed(self, gate_kept=True):
        if not gate_kept:
            return True
        else:
            if self.rank == 0:
                return True
            else:
                return False


class LossTracker(dict):

    def __init__(self, pred_names=('cs', 'eg', 'sg'), *args, **kwargs):
        """
        A dictionary-like object that stores the values of losses.

        Below are the major keys and their structures:
        - `loss`: 1D list. The total training loss.
        - `loss_<pred_name>`: 1D list. The sub-loss corresponding to prediction item `<pred_name>`.
        - `val_loss`: 1D list. The total validation loss.
        - `val_loss_<pred_name>`: 1D list. The sub-validation loss corresponding to prediction item `<pred_name>`.
        - `best_val_loss`, `best_val_loss_<pred_name>`: float. The best total and sub-validation loss so far.
        - `epochs`: 1D list. Epoch indices.
        - `lrs`: 1D list. The history of learning rate. *The LRs may be updated every minibatch in contrast to losses
                 which are updated every opech.*

        :param pred_names: tuple(str). Names of predicted quantities.
        """
        super().__init__(*args, **kwargs)
        self.pred_names = pred_names
        self.n_preds = len(pred_names)
        self['epochs'] = []
        self['loss'] = []
        self['val_loss'] = []
        self['best_val_loss'] = np.inf
        self['lrs'] = []
        self['epoch_best_val_loss'] = 0
        self.current_epoch = 0

        for pred_name in self.pred_names:
            self['loss_{}'.format(pred_name)] = []
            self['val_loss_{}'.format(pred_name)] = []
            self['best_val_loss_{}'.format(pred_name)] = np.inf
            self['train_acc_{}'.format(pred_name)] = []
            self['val_acc_{}'.format(pred_name)] = []
            self['classification_preds_{}'.format(pred_name)] = []
            self['classification_labels_{}'.format(pred_name)] = []

    def update_losses(self, losses, type='loss', epoch=None, lr=None):
        """
        Update losses.

        :param losses: list. Loss values. The first value must be the total loss, which is followed by sub-losses.
                       Sub-losses should follow the order given in `pred_names`.
        :param type: str. "loss" or "val_loss".
        :return: bool. If type is "val_loss" and the given validation loss is lower than the current best validation
                       loss, return True.
        """
        self[type].append(losses[0])

        if epoch is not None:
            self['epochs'].append(epoch)
        else:
            self['epochs'].append(len(self[type]) - 1)
        self.current_epoch = self['epochs'][-1]

        if lr is not None:
            self['lrs'].append(lr)

        for i, pred_name in enumerate(self.pred_names):
            self['{}_{}'.format(type, pred_name)].append(losses[i + 1])
        if type == 'val_loss' and losses[0] < self['best_val_loss']:
            self['best_val_loss'] = losses[0]
            for i, pred_name in enumerate(self.pred_names):
                self['best_val_loss_{}'.format(pred_name)] = losses[i + 1]
            if epoch is not None:
                self['epoch_best_val_loss'] = epoch
            return True
        else:
            return False

    def get_all_losses(self, type='loss'):
        """
        Get all losses in a 2D array.

        :param type: str. "loss" or "val_loss".
        :return: np.ndarray of shape `(len(pred_names), n_epochs)`.
        """
        loss_list = [self[type]]
        for pred_name in self.pred_names:
            loss_list.append(self['{}_{}'.format(type, pred_name)])
        return np.stack(loss_list)

    def get_metric_names_for_hyperparam_tuning(self):
        """
        Get a list of keys taht can be used as the objective for hyperparameter tuning.

        :return: list[str].
        """
        names = ['best_val_loss']
        for pred in self.pred_names:
            names.append('best_val_loss_{}'.format(pred))
        names.append('epoch_best_val_loss')
        return names

    def print_losses(self):
        logging.info('Epoch: %d | All | Train Loss: %.5f | Val Loss: %.5f' % (
            self.current_epoch, self['loss'][-1], self['val_loss'][-1]))
        for i_pred, pred_name in enumerate(self.pred_names):
            logging.info('Epoch: %d | %s  | Train Loss: %.4f | Val Loss: %.4f' % (
                self.current_epoch, pred_name.upper(),
                self['loss_{}'.format(pred_name)][-1],
                self['val_loss_{}'.format(pred_name)][-1]))
        if len(self['lrs']) > 0:
            logging.info('Epoch: %d | Ending LR: %.6f ' % (self.current_epoch, self['lrs'][-1]))

    def plot(self, quantities=('loss', 'val_loss'), save_path=None):
        plt.figure()
        for q in quantities:
            plt.plot(self[q], label=q)
        plt.legend()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def clear_classification_results_and_labels(self):
        for pred_name in self.pred_names:
            self['classification_preds_{}'.format(pred_name)] = []
            self['classification_labels_{}'.format(pred_name)] = []

    def update_classification_results_and_labels(self, preds, labels):
        """
        Update the classification results recorded for the current epoch with the predictions and labels of
        the current iteration.

        :param preds: list[torch.tensor]. Each tensor should be of shape [n_batch, n_classes].
        :param labels: list[torch.tensor]. Each tensor should be of shape [n_batch, n_classes].
        :return:
        """
        pred_dict = {}
        label_dict = {}
        for i, pred_name in enumerate(self.pred_names):
            inds_pred = torch.argmax(preds[i], dim=1)
            inds_label = torch.argmax(labels[i], dim=1)
            pred_dict[pred_name] = inds_pred
            label_dict[pred_name] = inds_label
            self['classification_preds_{}'.format(pred_name)] += inds_pred.tolist()
            self['classification_labels_{}'.format(pred_name)] += inds_label.tolist()

    def calculate_classification_accuracy(self):
        """
        Calculate classification accuracies at the end of an epoch using the recorded predictions and labels.
        """
        acc_dict = {}
        for i, pred_name in enumerate(self.pred_names):
            inds_pred = self['classification_preds_{}'.format(pred_name)]
            inds_label = self['classification_labels_{}'.format(pred_name)]
            acc = np.mean((np.array(inds_pred) == np.array(inds_label)))
            acc_dict[pred_name] = acc
        return acc_dict

    def update_accuracy_history(self, acc_dict, type='train'):
        """
        Update accuracy history.

        :param acc_dict: dict. A dictionary where each key is in pred_names, and the corresponding value is
                         the accuracy of that catefory for all samples in the current epoch.
        :param type: str. Can be 'train' or 'val'.
        """
        for i, pred_name in enumerate(self.pred_names):
            self['{}_acc_{}'.format(type, pred_name)].append(acc_dict[pred_name])

    def sync_classification_preds_and_labels_across_ranks(self):
        if MPI is None:
            return
        comm = MPI.COMM_WORLD
        n_ranks = comm.Get_size()
        if n_ranks == 1:
            return
        for i, pred_name in enumerate(self.pred_names):
            assert isinstance(self['classification_preds_{}'.format(pred_name)], list)
            self['classification_preds_{}'.format(pred_name)] = (
                comm.allreduce(self['classification_preds_{}'.format(pred_name)], op=MPI.SUM))
            assert isinstance(self['classification_labels_{}'.format(pred_name)], list)
            self['classification_labels_{}'.format(pred_name)] = (
                comm.allreduce(self['classification_labels_{}'.format(pred_name)], op=MPI.SUM))



class Trainer:

    def __init__(self, configs: TrainingConfig, rank=None, num_processes=None, *args, **kwargs):
        """
        Trainer constructor.

        :param configs: TrainingConfig.
        :param rank: int. The current index of rank. This argument should be kept None unless multi_node
                     parallelization is intended and training is run using torch.multiprocessing.spawn
                     (instead of torchrun, where the rank can be automatically figured out and does not need
                     to be passed to the trainer explicitly).
        :param num_processes: int. The total number of processes. Similar to `rank`, this argument should be kept
                              None unless multi_node is intended and training is run using torch.multiprocessing.spawn.
        """
        self.configs = configs
        self.parallelization_type = self.configs.parallelization_params.parallelization_type
        self.dataset = self.configs.dataset
        assert isinstance(self.dataset, Dataset)
        self.training_dataset = None
        self.validation_dataset = None
        self.validation_ratio = self.configs.validation_ratio
        self.model = None
        self.model_params = None
        self.model_class_handle = None
        self.training_dataloader = None
        self.validation_dataloader = None
        self.num_local_devices = self.get_num_local_devices()
        self.num_processes = num_processes
        self.rank = rank
        self.device = self.get_device()
        self.all_proc_batch_size = self.configs.batch_size_per_process
        self.learning_rate = self.configs.learning_rate_per_process
        self.num_epochs = self.configs.num_epochs
        self.optimizer = None
        self.scheduler = None
        self.loss_tracker = None
        self.loss_criterion = self.configs.loss_function
        self.iterations_per_epoch = 0
        self.current_epoch = 0
        self.gatekeeper = MultirankGateKeeper(0, 1)

        self.debug = self.configs.debug
        self.verbose = True

    def get_num_processes(self):
        """
        The number of processes refers to the number of workers in data parallelism. Gradients are averaged
        among these workers. The base learning rate and batch size are multiplied by this number.
        """
        if self.parallelization_type == 'single_node':
            if self.configs.cpu_only:
                return 1
            if not torch.cuda.is_available():
                return 1
            else:
                return torch.cuda.device_count()
        elif self.parallelization_type == 'multi_node':
            # If self.rank is not None, we assume the trainer is run using torch.multiprocessing.spawn, where
            # the rank and the number of processes are passed to the trainer explicitly.
            if self.num_processes is not None:
                return self.num_processes
            # Otherwise, we assume the trainer is run using torchrun or mpirun, where the rank and the number of
            # processes should be figured out from the context.
            else:
                try:
                    return dist.get_world_size()
                except RuntimeError as e:
                    raise RuntimeError('The following error occurred: {}\nDid you call dist.init_process_group '
                                       'or util.setup_multiprocessing in the launch script? Either function must be '
                                       'called prior to instantiating the trainer.'.format(e))
        else:
            raise ValueError('{} is not a valid parallelization type.'.format(self.parallelization_type))

    def get_rank(self):
        if self.parallelization_type == 'single_node':
            return 0
        elif self.parallelization_type == 'multi_node':
            # If self.rank is not None, we assume the trainer is run using torch.multiprocessing.spawn, where
            # the rank and the number of processes are passed to the trainer explicitly.
            if self.rank is not None:
                return self.rank
            # Otherwise, we assume the trainer is run using torchrun or mpirun, where the rank and the number of
            # processes should be figured out from the context.
            else:
                try:
                    return dist.get_rank()
                except RuntimeError as e:
                    raise RuntimeError('The following error occurred: {}\nDid you call dist.init_process_group '
                                       'or util.setup_multiprocessing in the launch script? Either function must be '
                                       'called prior to instantiating the trainer.'.format(e))
        else:
            raise ValueError('{} is not a valid parallelization type.'.format(self.parallelization_type))

    def get_num_local_devices(self):
        if self.parallelization_type == 'single_node':
            if self.configs.cpu_only:
                return 1
            if not torch.cuda.is_available():
                return 1
            else:
                return torch.cuda.device_count()
        elif self.parallelization_type == 'multi_node':
            return torch.cuda.device_count()

    def get_device(self):
        if self.configs.cpu_only:
            return torch.device('cpu')
        # Don't specify GPU index here (i.e., no "cuda:X").
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_worker_seed_func(self):
        if self.configs.random_seed is None:
            return None
        else:
            def seed_worker(worker_id):
                worker_seed = torch.initial_seed() % 2 ** 32
                np.random.seed(worker_seed)
                random.seed(worker_seed)
            return seed_worker

    def get_dataloader_generator(self):
        if self.configs.random_seed is None:
            return None
        g = torch.Generator()
        g.manual_seed(self.configs.random_seed)
        return g

    def build(self):
        if self.configs.random_seed is not None:
            set_all_random_seeds(self.configs.random_seed)

        # When default device is set to `cuda`, DataLoader with `shuffle=True` would crash when yielding due to an
        # internal bug of PyTorch. Therefore, we set default device to `cpu` here and manually assign device to objects.
        set_default_device('cpu')
        self.check_configs()

        self.build_loss_tracker()

        self.build_ranks()
        self.build_scalable_parameters()
        self.build_device()

        self.build_split_datasets()
        self.build_dataloaders()

        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.load_state_checkpoint()

        self.build_dir()

    def check_configs(self):
        if ('pred_names_and_num_classes' in self.configs.model_params.__dict__.keys() and
                'pred_names' in self.configs.__dict__.keys()):
            pred_names_model_params = [x[0] for x in self.configs.model_params.pred_names_and_num_classes]
            pred_names_configs = list(self.configs.pred_names)
            if pred_names_model_params != pred_names_configs:
                warnings.warn('pred_names in model_params and configs are not the same: it is {} and {}.'.format(
                    pred_names_model_params, pred_names_configs
                ))

    def build_loss_tracker(self):
        self.loss_tracker = LossTracker(pred_names=self.configs.pred_names,
                                        **self.configs.loss_tracker_params.__dict__)

    def build_dir(self):
        if self.gatekeeper.should_proceed(gate_kept=True):
            if not os.path.exists(self.configs.model_save_dir):
                os.makedirs(self.configs.model_save_dir)
        self.barrier()

    def build_device(self):
        self.device = self.get_device()
        self.num_local_devices = self.get_num_local_devices()

    def build_ranks(self):
        self.rank = self.get_rank()
        self.num_processes = self.get_num_processes()
        self.gatekeeper = MultirankGateKeeper(self.rank, self.num_processes)

    def build_scalable_parameters(self):
        self.all_proc_batch_size = self.configs.batch_size_per_process * self.num_processes
        self.learning_rate = self.configs.learning_rate_per_process * self.num_processes

    def build_dataloaders(self):
        if self.parallelization_type == 'multi_node':
            training_sampler = torch.utils.data.distributed.DistributedSampler(self.training_dataset,
                                                                               num_replicas=self.num_processes,
                                                                               rank=self.rank,
                                                                               drop_last=True)
            self.training_dataloader = DataLoader(self.training_dataset,
                                                  batch_size=self.configs.batch_size_per_process,
                                                  sampler=training_sampler,
                                                  collate_fn=lambda x: x)
            validation_sampler = torch.utils.data.distributed.DistributedSampler(self.validation_dataset,
                                                                                 num_replicas=self.num_processes,
                                                                                 rank=self.rank,
                                                                                 drop_last=True)
            self.validation_dataloader = DataLoader(self.validation_dataset,
                                                    batch_size=self.all_proc_batch_size,
                                                    sampler=validation_sampler,
                                                    collate_fn=lambda x: x)
        else:
            # ALCF documentation mentions that there is a bug in Pytorch's multithreaded data loaders with
            # distributed training across multiple nodes. Therefore, `num_workers` is set to 0. See also:
            # https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/.
            self.training_dataloader = DataLoader(self.training_dataset, shuffle=True,
                                                  batch_size=self.all_proc_batch_size,
                                                  collate_fn=lambda x: x, worker_init_fn=self.get_worker_seed_func(),
                                                  generator=self.get_dataloader_generator(), num_workers=0,
                                                  drop_last=False)
            self.validation_dataloader = DataLoader(self.validation_dataset, shuffle=True,
                                                    batch_size=self.all_proc_batch_size,
                                                    collate_fn=lambda x: x, worker_init_fn=self.get_worker_seed_func(),
                                                    generator=self.get_dataloader_generator(), num_workers=0,
                                                    drop_last=False)

    def run_training(self):
        for self.current_epoch in range(self.current_epoch, self.num_epochs):
            # Set model to train mode and run training
            self.model.train()
            self.run_training_epoch()

            # Switch model to eval mode and run validation
            self.model.eval()
            self.run_validation()

            if self.verbose and self.rank == 0:
                self.loss_tracker.print_losses()
        self.update_saved_model(filename='final_model.pth')

    def compute_losses(self, loss_records, preds, labels):
        """
        Run the model with the data of the current iteration and get losses.

        :param loss_records: list[float]. A list that keep tracks of the accumulated losses in the current epoch.
                             These values are just for record keeping and are not tensors.
        :param preds: list[torch.Tensor]. The list of predictions.
        :param labels: list[torch.Tensor]. The list of labels.
        :return: list, torch.Tensor. Updated loss records and total loss tensor.
        """
        # Compute losses
        total_loss_tensor = 0.0
        for i_pred in range(len(preds)):
            if isinstance(self.loss_criterion, Callable):
                this_loss_func = self.loss_criterion
            else:
                this_loss_func = self.loss_criterion[i_pred]
            # Try casting labels to be the same type as preds. If it doesn't work, use the original type.
            try:
                this_loss_tensor = this_loss_func(preds[i_pred], labels[i_pred].type(preds[i_pred].dtype))
            except:
                this_loss_tensor = this_loss_func(preds[i_pred], labels[i_pred])
            total_loss_tensor = total_loss_tensor + this_loss_tensor
            loss_records[i_pred + 1] += this_loss_tensor.detach().item()

        if hasattr(self.loss_criterion, '__len__') and len(self.loss_criterion) > len(preds):
            pred_dict = self.get_pred_dict(preds)
            for i in range(len(preds), len(self.loss_criterion)):
                this_loss_func = self.loss_criterion[i]
                try:
                    this_loss_tensor = this_loss_func(**pred_dict)
                except TypeError:
                    this_loss_tensor = this_loss_func(*preds)
                total_loss_tensor = total_loss_tensor + this_loss_tensor
                loss_records[i + 1] += this_loss_tensor.detach().item()
        loss_records[0] += total_loss_tensor.detach().item()
        return loss_records, total_loss_tensor

    def get_pred_dict(self, preds):
        """
        Convert a list of predictions to a dictionary keyed according to `pred_names`.
        :param preds: list[torch.tensor].
        :return: dict.
        """
        d = {}
        for i, name in enumerate(self.configs.pred_names):
            d[name] = preds[i]
        return d

    def process_data_loader_yield_sample_first(self, data_and_labels, data_label_separation_index):
        n_items = len(data_and_labels[0])
        n_samples = len(data_and_labels)
        if data_label_separation_index is None:
            data_label_separation_index = n_items
        data_list = []
        for i_item in range(data_label_separation_index):
            data_all_samples_this_item = [data_and_labels[i_sample][i_item] for i_sample in range(n_samples)]
            data_all_samples_this_item = self.move_to_device(torch.concat(data_all_samples_this_item, dim=0))
            data_list.append(data_all_samples_this_item)
        label_list = []
        for i_item in range(data_label_separation_index, n_items):
            label_all_samples_this_item = [data_and_labels[i_sample][i_item] for i_sample in range(n_samples)]
            label_all_samples_this_item = self.move_to_device(torch.concat(label_all_samples_this_item, dim=0))
            label_list.append(label_all_samples_this_item)
        return data_list, label_list

    def process_data_loader_yield_item_first(self, data_and_labels, data_label_separation_index):
        n_items = len(data_and_labels)
        if data_label_separation_index is None:
            data_label_separation_index = n_items
        data_list = []
        for i_item in range(data_label_separation_index):
            data_all_samples_this_item = data_and_labels[i_item]
            data_all_samples_this_item = self.move_to_device(data_all_samples_this_item)
            data_list.append(data_all_samples_this_item)
        label_list = []
        for i_item in range(data_label_separation_index, n_items):
            label_all_samples_this_item = data_and_labels[i_item]
            label_all_samples_this_item = self.move_to_device(label_all_samples_this_item)
            label_list.append(label_all_samples_this_item)
        return data_list, label_list

    def process_data_loader_yield(self, data_and_labels: Any, data_label_separation_index: Optional[int] = 1):
        """
        Disentangles the yields from the dataloader, returning a tuple of (data, label1, label2, ...) with
        each element being a tensor of [batch_size_per_process, ...].

        With the collate_fn defined, the yields of dataloader are different between PyTorch 1.x and 2.x. This
        function automatically detects the format and treat the data accordingly.

        :param data_and_labels: Any. The yield of the dataloader. It could be either a tuple of tensors, in which
                                case of each tensor is the stacked data/label for all samples in the batch, or a
                                tuple of `batch_size` tuples, where each sub-tuple contains several tensors that are
                                the data/label of a sample.
        :param data_label_separation_index: int | None. The separation index of data and label. If this is an integer,
                                            then the first `data_label_separation_index` items are considered to be
                                            data, and the rest are labels. If this is None, all items are considered
                                            to be data.
        :returns tuple[torch.tensor], tuple[torch.tensor]. 2 tuples for data and labels.
        """
        if isinstance(data_and_labels[0], tuple):
            # In this case, data_and_labels is in sample-then-item order.
            data_list, label_list = self.process_data_loader_yield_sample_first(data_and_labels,
                                                                                data_label_separation_index)
        else:
            # In this case, data_and_labels is in item-then-sample order.
            data_list, label_list = self.process_data_loader_yield_item_first(data_and_labels,
                                                                              data_label_separation_index)
        return data_list, label_list

    def get_epoch_loss_buffer(self):
        if hasattr(self.loss_criterion, '__len__'):
            n = len(self.loss_criterion) + 1
        else:
            n = self.loss_tracker.n_preds + 1
        return [0.0] * n

    def load_data_and_get_loss(self, data_and_labels, loss_buffer, *args, **kwargs):
        """
        Load data, run prediction, calculate loss, and return the tracked loss values,
        the gradient-tracked loss tensor, and predictions and labels. Override this method
        in subclasses to adapt to different dataloader/model/loss signatures and returned
        structures.

        :param data_and_labels: Any. The data structure yielded by the dataloader.
        :param loss_buffer: list[float]. A list that stores the total loss and all sub-losses.
                            The elements should be constant values, not differentiable tensors.
        :return: loss_buffer, total_loss_tensor, preds, labels
        """
        data, labels = self.process_data_loader_yield(data_and_labels)
        preds = self.model(*data)
        losses, total_loss_tensor = self.compute_losses(loss_buffer, preds, labels)
        return loss_buffer, total_loss_tensor, preds, labels

    def run_training_epoch(self):
        losses = self.get_epoch_loss_buffer()
        n_batches = 0
        if self.configs.task_type == 'classification':
            self.loss_tracker.clear_classification_results_and_labels()
        for i, data_and_labels in enumerate(tqdm(self.training_dataloader, disable=(not self.verbose))):
            losses, total_loss_tensor, preds, labels = self.load_data_and_get_loss(data_and_labels, losses)
            if self.configs.task_type == 'classification':
                self.loss_tracker.update_classification_results_and_labels(preds, labels)

            # Zero current grads and do backprop
            self.run_model_update_step(total_loss_tensor)

            # Update the LR according to the schedule -- CyclicLR updates each batch
            self.scheduler.step()
            self.loss_tracker['lrs'].append(self.scheduler.get_last_lr()[0])

            n_batches += 1
        # Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        losses = [self.communicate_value_across_ranks(l / n_batches, mode='average') for l in losses]
        self.loss_tracker.update_losses(losses, type='loss', epoch=self.current_epoch)

        if self.configs.task_type == 'classification':
            self.loss_tracker.sync_classification_preds_and_labels_across_ranks()
            acc_dict = self.loss_tracker.calculate_classification_accuracy()
            self.loss_tracker.update_accuracy_history(acc_dict, 'train')

        self.save_model_and_states_checkpoint()

        if self.configs.post_training_epoch_hook is not None:
            self.configs.post_training_epoch_hook()

    def run_validation(self):
        losses = self.get_epoch_loss_buffer()
        n_batches = 0
        if self.configs.task_type == 'classification':
            self.loss_tracker.clear_classification_results_and_labels()
        for j, data_and_labels in enumerate(self.validation_dataloader):
            losses, _, preds, labels = self.load_data_and_get_loss(data_and_labels, losses)
            if self.configs.task_type == 'classification':
                self.loss_tracker.update_classification_results_and_labels(preds, labels)
            n_batches += 1
        if n_batches == 0:
            logging.warning('Validation set might be too small that at least 1 rank did not get any validation data.')
        n_batches = np.max([n_batches, 1])
        last_best_val_loss = self.loss_tracker['best_val_loss']

        losses = [self.communicate_value_across_ranks(l / n_batches, mode='average') for l in losses]
        is_best = self.loss_tracker.update_losses(losses, epoch=self.current_epoch, type='val_loss')
        self.write_training_info()

        # Update saved model if val loss is lower
        if is_best:
            logging.info("Saving improved model after Val Loss improved from %.5f to %.5f" % (
                last_best_val_loss, self.loss_tracker['best_val_loss']))
            self.update_saved_model(filename='best_model.pth')

        if self.configs.task_type == 'classification':
            self.loss_tracker.sync_classification_preds_and_labels_across_ranks()
            acc_dict = self.loss_tracker.calculate_classification_accuracy()
            self.loss_tracker.update_accuracy_history(acc_dict, 'val')

        if self.configs.post_validation_epoch_hook is not None:
            self.configs.post_validation_epoch_hook()

    def run_model_update_step(self, loss_node):
        self.optimizer.zero_grad()
        loss_node.backward()
        self.optimizer.step()

    def build_split_datasets(self):
        train_idx, val_idx = train_test_split(list(range(len(self.dataset))), test_size=self.validation_ratio)
        self.training_dataset = Subset(self.dataset, train_idx)
        self.validation_dataset = Subset(self.dataset, val_idx)

    def build_optimizer(self):
        if isinstance(self.configs.optimizer, str):
            if self.configs.optimizer == 'adam':
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            self.optimizer = self.configs.optimizer(self.model.parameters(), lr=self.learning_rate,
                                                    **self.configs.optimizer_params)

    def build_scheduler(self):
        self.iterations_per_epoch = (len(self.training_dataset) - len(self.validation_dataset))
        self.iterations_per_epoch = self.iterations_per_epoch / self.all_proc_batch_size
        self.iterations_per_epoch = np.floor(self.iterations_per_epoch) + 1
        step_size = 6 * self.iterations_per_epoch
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.learning_rate / 10,
                                                           max_lr=self.learning_rate, step_size_up=step_size,
                                                           cycle_momentum=False, mode='triangular2')

    def build_model(self):
        self.model_class_handle = self.configs.model_class
        self.model = self.configs.model_class(**self.configs.model_params.__dict__)

        # For single_node parallelization using DataParallel model, checkpoint should be loaded before
        # DataParallel(model).
        if self.parallelization_type == 'single_node':
            if self.configs.pretrained_model_path is not None:
                if self.configs.load_pretrained_encoder_only:
                    logging.info('Loading pretrained encoder from {}.'.format(self.configs.pretrained_model_path))
                    self.load_model(self.configs.pretrained_model_path, subcomponent='backbone_model')
                else:
                    logging.info('Loading pretrained model from {}.'.format(self.configs.pretrained_model_path))
                    self.load_model(self.configs.pretrained_model_path)
            elif self.configs.checkpoint_dir is not None:
                logging.info('Loading checkpointed model from {}.'.format(self.configs.checkpoint_dir))
                self.load_model(os.path.join(self.configs.checkpoint_dir, 'checkpoint_model.pth'))

        self.build_parallelism()

        # For multi_node parallelization using DistributedDataParallel (DDP), checkpoint should be loaded after
        # DistributedDataParallel(model).
        if self.parallelization_type == 'multi_node':
            if self.configs.pretrained_model_path is not None:
                if self.configs.load_pretrained_encoder_only:
                    logging.info('Loading pretrained encoder from {}.'.format(self.configs.pretrained_model_path))
                    self.load_model(self.configs.pretrained_model_path, subcomponent='backbone_model')
                else:
                    logging.info('Loading pretrained model from {}.'.format(self.configs.pretrained_model_path))
                    self.load_model(self.configs.pretrained_model_path)
            elif self.configs.checkpoint_dir is not None:
                logging.info('Loading checkpointed model from {}.'.format(self.configs.checkpoint_dir))
                self.load_model(os.path.join(self.configs.checkpoint_dir, 'checkpoint_model.pth'))

    def build_parallelism(self):
        if self.parallelization_type == 'single_node':
            if self.device.type == 'cuda' and self.num_local_devices > 1:
                self.model = nn.DataParallel(self.model)
                logging.info('Using DataParallel with {} devices.'.format(self.num_local_devices))
            self.model.to(self.device)

        elif self.parallelization_type == 'multi_node':
            if self.device.type == 'cuda' and len(get_cuda_visible_devices_from_environ()) != 1:
                logging.warning('Parallelization type is multi_node, but CUDA_VISIBLE_DEVICES is {}. This variable '
                               'should be set to exactly 1 GPU for each process before MPI initialization.'.
                               format(os.environ['CUDA_VISIBLE_DEVICES']))
            logging.info('Sending model on rank {} to device {}.'.format(self.rank, self.rank))
            if not self.configs.cpu_only:
                self.model.to(self.device)
            try:
                self.model = torch.nn.parallel.DistributedDataParallel(
                    self.model,
                    find_unused_parameters=self.configs.parallelization_params.find_unused_parameters)
            except RuntimeError as e:
                raise RuntimeError('The following error occurred: {}\nDid you call dist.init_process_group '
                                   'or util.setup_multiprocessing in the launch script? Either function must be '
                                   'called prior to instantiating the trainer.'.format(e))

        else:
            raise ValueError('{} is not a valid parallelization type.'.format(self.parallelization_type))

    def load_model(self, path=None, state_dict=None, subcomponent=None):
        if path is not None:
            if self.parallelization_type == 'single_node':
                state_dict = torch.load(path)
            else:
                # In multi-node mode each rank should only get 1 GPU, so we do not use map_location.
                state_dict = torch.load(path)
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            m = self.model.module
        else:
            m = self.model
        if subcomponent is None:
            m.load_state_dict(state_dict)
        else:
            getattr(m, subcomponent).load_state_dict(state_dict)

    def save_model(self, path, subcomponent=None):
        if isinstance(self.model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
            m = self.model.module
        else:
            m = self.model
        if subcomponent is not None:
            m = getattr(m, subcomponent)
        torch.save(m.state_dict(), path)

    def update_saved_model(self, filename='best_model.pth', save_configs=True, subcomponent=None,
                           run_with_only_rank_0=True):
        """
        Updates saved model if validation loss is minimum.
        """
        if not self.gatekeeper.should_proceed(gate_kept=run_with_only_rank_0):
            return
        path = self.configs.model_save_dir
        dest_path = os.path.join(path, filename)
        try:
            if not os.path.isdir(path):
                os.mkdir(path)
            if os.path.exists(dest_path):
                os.remove(dest_path)
        except Exception as e:
            logging.warning('The following exception occurred when creating directories/removing old files in '
                            'updated_saved_model: \n{}\nThis could happen if multiple processes are competing '
                            'for I/O operations. It normally should not happen, but could occur if you are '
                            'using the updated_saved_model method defined in the base class in a subclass using '
                            'high-level wrappers like HuggingFace Accelerate. For now, you can ignore this warning.'.
                            format(e))
        self.save_model(dest_path, subcomponent=subcomponent)
        if save_configs:
            self.configs.dump_to_json(os.path.join(path, 'configs.json'))

    def generate_state_dict(self):
        """
        Get a dictionart of the state_dicts of all states but not the model.
        """
        state = {
            'current_epoch': self.current_epoch + 1,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'loss_tracker': self.loss_tracker
        }
        return state

    def save_model_and_states_checkpoint(self):
        if not self.gatekeeper.should_proceed(gate_kept=True):
            return
        state_dict = self.generate_state_dict()
        torch.save(state_dict, os.path.join(self.configs.model_save_dir, 'checkpoint.state'))
        self.update_saved_model('checkpoint_model.pth')

    def load_state_checkpoint(self):
        if self.configs.checkpoint_dir is None:
            return
        checkpoint_fname = os.path.join(self.configs.checkpoint_dir, 'checkpoint.state')
        if not os.path.exists(checkpoint_fname):
            logging.warning('Checkpoint not found in {}.'.format(checkpoint_fname))
        state_dict = torch.load(checkpoint_fname)
        self.current_epoch = state_dict['current_epoch']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if state_dict['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        self.loss_tracker = state_dict['loss_tracker']

    def write_training_info(self):
        if not self.gatekeeper.should_proceed(gate_kept=True):
            return
        f = open(os.path.join(self.configs.model_save_dir, 'training_info.txt'), 'w')
        for key in self.loss_tracker:
            f.write('{} = {}\n'.format(key, self.loss_tracker[key]))

    def plot_training_history(self):
        self.plot_lr_history()
        self.plot_loss_history()

    def plot_lr_history(self):
        batches = np.linspace(0, len(self.loss_tracker['lrs']), len(self.loss_tracker['lrs']) + 1)
        epoch_list = batches / self.iterations_per_epoch
        fig, ax = plt.subplots(1, 1)
        ax.plot(epoch_list[1:], self.loss_tracker['lrs'], 'C3-')
        plt.grid()
        ax.set_ylabel("Learning rate")
        ax.set_xlabel("Epoch")

    def plot_loss_history(self):
        losses_arr = self.loss_tracker.get_all_losses('loss')
        if (len(losses_arr) == 0):
            print('Unable to plot: loss array is empty.')
            return
        val_losses_arr = self.loss_tracker.get_all_losses('val_loss')
        fig, ax = plt.subplots(losses_arr.shape[0], sharex=True, figsize=(15, 3 * losses_arr.shape[0]))
        for i in range(losses_arr.shape[0]):
            name = 'total' if i == 0 else self.loss_tracker.pred_names[i - 1]
            ax[i].plot(losses_arr[:, 0], 'C3o-', label="{} train loss".format(name))
            ax[i].plot(val_losses_arr[:, 0], 'C0o-', label="{} val loss".format(name))
            ax[i].set(ylabel='Loss')
            ax[i].grid()
            ax[i].legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
        plt.tight_layout()
        plt.xlabel("Epochs")
        plt.savefig(os.path.join(self.configs.model_save_dir, 'loss_history.png'))

    def plot_accuracy_history(self):
        for name in self.configs.pred_names:
            self.loss_tracker.plot(
                quantities=('train_acc_{}'.format(name), 'val_acc_{}'.format(name)),
                save_path=os.path.join(self.configs.model_save_dir,
                                       'acc_history_{}.png'.format(name))
            )

    def plot_images(self, image_list):
        fig, ax = plt.subplots(1, len(image_list))
        for i, img in enumerate(image_list):
            try:
                ax[i].imshow(img)
            except TypeError:
                try:
                    img = img.cpu().numpy()
                    ax[i].imshow(img)
                except:
                    img = img.detach().cpu().numpy()
                    ax[i].imshow(img)
        plt.show()

    def run_testing(self, ind_list, dataset='train'):
        self.model.eval()
        dset = self.training_dataset if dataset == 'train' else self.validation_dataset
        dp_list, true_amp, true_ph = dset.__getitems__(ind_list)
        dp_list.to(self.device)
        preds = self.model(dp_list)
        return preds

    def barrier(self):
        if self.parallelization_type == 'single_node':
            return
        elif self.parallelization_type == 'multi_node':
            dist.barrier()

    def communicate_value_across_ranks(self, var, mode='average'):
        """
        Communicate values across MPI ranks. This can be used for allreduce (averaging the values of a variable
        across all ranks), or gathering (allowing each rank to receive the values from other ranks and concatenate
        them).

        :param var: Any. The variable to be communicated.
        :param mode: str. Can be the following:
                     - 'average': averages the value across all ranks.
                     - 'gather': gather the value from all ranks, and concatenate them as a list.
        :return: value after communication.
        """
        if MPI is None or self.num_processes == 1:
            return var
        comm = MPI.COMM_WORLD
        if mode == 'average':
            var = comm.allreduce(var, op=MPI.SUM) / self.num_processes
        elif mode == 'gather':
            var = comm.allgather(var)
        return var

    def move_to_device(self, var):
        return var.to(self.device)

    def cleanup_memory(self):
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        self.model = None
        if self.configs.parallelization_params.parallelization_type == 'multi_node':
            cleanup_multiprocessing()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Pretrainer(Trainer):
    def __init__(self, configs: TrainingConfig, rank=None, num_processes=None, *args, **kwargs):
        """
        Trainer constructor.

        :param configs: TrainingConfig.
        :param rank: int. The current index of rank. This argument should be kept None unless multi_node
                     parallelization is intended and training is run using torch.multiprocessing.spawn
                     (instead of torchrun, where the rank can be automatically figured out and does not need
                     to be passed to the trainer explicitly).
        :param num_processes: int. The total number of processes. Similar to `rank`, this argument should be kept
                              None unless multi_node is intended and training is run using torch.multiprocessing.spawn.
        """
        super().__init__(configs, rank, num_processes, *args, **kwargs)
        self.configs.task_type = None

    def process_data_loader_yield(self, data, **kwargs):
        # All that the dataloader yield are supposed to be data. No label.
        super().process_data_loader_yield(data, data_label_separation_index=None)
        return data

    def compute_losses(self, loss_records, preds, *args, **kwargs):
        """
        Run the model with the data of the current iteration and get losses.

        :param loss_records: list[float]. A list that keep tracks of the accumulated losses in the current epoch.
                             These values are just for record keeping and are not tensors.
        :param preds: list[torch.Tensor]. The list of predictions.
        :return: list, torch.Tensor. Updated loss records and total loss tensor.
        """
        # Compute losses
        if isinstance(self.loss_criterion, Callable):
            this_loss_func = self.loss_criterion
        else:
            this_loss_func = self.loss_criterion[0]
        total_loss_tensor = this_loss_func(*preds)
        loss_records[1] += total_loss_tensor.detach().item()

        if hasattr(self.loss_criterion, '__len__') and len(self.loss_criterion) > 1:
            for i in range(1, len(self.loss_criterion)):
                this_loss_func = self.loss_criterion[i]
                this_loss_tensor = this_loss_func(*preds)
                total_loss_tensor = total_loss_tensor + this_loss_tensor
                loss_records[i + 1] += this_loss_tensor.detach().item()
        loss_records[0] += total_loss_tensor.detach().item()
        return loss_records, total_loss_tensor

    def load_data_and_get_loss(self, data, loss_buffer):
        # elements of data are supposed to be 2 different augmentations.
        data = self.process_data_loader_yield(data)
        preds = self.model(*data)
        losses, total_loss_tensor = self.compute_losses(loss_buffer, preds)
        return loss_buffer, total_loss_tensor, preds, None

    def update_saved_model(self, filename='best_model.pth', **kwargs):
        """
        Updates saved model if validation loss is minimum.
        """
        if not self.gatekeeper.should_proceed(gate_kept=True):
            return
        super().update_saved_model(filename, save_configs=True, subcomponent=None)
        encoder_filename = os.path.splitext(filename)[0] + '_encoder.pth'
        super().update_saved_model(encoder_filename, save_configs=False, subcomponent='encoder')


class HuggingFaceAccelerateTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accelerator = None

    def build(self):
        if self.configs.random_seed is not None:
            set_all_random_seeds(self.configs.random_seed)

        # When default device is set to `cuda`, DataLoader with `shuffle=True` would crash when yielding due to an
        # internal bug of PyTorch. Therefore, we set default device to `cpu` here and manually assign device to objects.
        set_default_device('cpu')
        self.check_configs()

        self.build_loss_tracker()

        self.build_ranks()
        self.build_scalable_parameters()
        self.build_device()

        self.build_split_datasets()
        self.build_dataloaders()

        self.build_model()
        self.build_optimizer()
        self.build_scheduler()
        self.build_accelerate()

        self.build_dir()

    def build_model(self):
        self.model = self.configs.model_class(**self.configs.model_params.__dict__)

    def build_accelerate(self):
        from accelerate import Accelerator
        self.accelerator = Accelerator()

        self.model, self.optimizer, self.training_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.training_dataloader, self.scheduler
        )

        if self.configs.pretrained_model_path is not None:
            if self.configs.load_pretrained_encoder_only:
                self.load_model(self.configs.pretrained_model_path, subcomponent='encoder')
            else:
                self.load_model(self.configs.pretrained_model_path)
        elif self.configs.checkpoint_dir is not None:
            self.load_state_checkpoint()

    def run_model_update_step(self, loss_node):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss_node)
        self.optimizer.step()

    def move_to_device(self, var):
        # HuggingFace Accelerate should not need manual data offloading.
        return var

    def update_saved_model(self, filename='best_model', save_configs=True, subcomponent=None, **kwargs):
        """self.configs.model_class
        Save model checkpoint.
        HuggingFace Accelerate takes a directory to save the model. This directory will be named as
        basename(splitext(filename)[0]).

        :param filename: str. Name of the checkpoint directory. If it comes with an extension, the extension will
                         be removed.
        :param save_configs: bool. If True, trainer configs will also be saved as a JSON.
        :param subcomponent: str. If not None, only the subcomponent of the model with this name will be saved.
        """
        path = os.path.join(self.configs.model_save_dir, os.path.splitext(filename)[0])
        self.accelerator.save_state(path)
        if save_configs and self.gatekeeper.should_proceed(gate_kept=True):
            self.configs.dump_to_json(os.path.join(self.configs.model_save_dir, 'configs.json'))

    def save_model_and_states_checkpoint(self):
        # Save epoch counter and loss tracker.
        state_dict = self.generate_state_dict()
        torch.save(state_dict, os.path.join(self.configs.model_save_dir, 'checkpoint.state'))

        # Save model, optimizer, scheduler, and dataloader states.
        self.update_saved_model(filename='checkpoint_model')

    def load_model(self, path=None, state_dict=None, subcomponent=None):
        if len(os.path.splitext(path)[1]) == '':
            logging.warning('The provided mode path {} does not have an extension so I am assuming it is the '
                            'HuggingFace checkpoint format. The states of all other components like optimizer, '
                            'scheduler etc. will also be loaded.'.format(path))
            self.accelerator.load_state(path)
        else:
            logging.warning('The provided mode path {} is assumed to be a native PyTorch checkpoint. Loading it '
                            'with the native load_model method.'.format(path))
            Pretrainer.load_model(self, path=path, state_dict=state_dict, subcomponent=subcomponent)

    def load_state_checkpoint(self):
        if self.configs.checkpoint_dir is None:
            return
        # Accelerator loads model, optimizer, scheduler, and dataloader states.
        self.accelerator.load_state(os.path.join(self.configs.checkpoint_dir, 'checkpoint_model'))

        # Also load epoch counter and loss tracker.
        checkpoint_fname = os.path.join(self.configs.checkpoint_dir, 'checkpoint.state')
        if not os.path.exists(checkpoint_fname):
            logging.warning('Checkpoint not found in {}.'.format(checkpoint_fname))
        state_dict = torch.load(checkpoint_fname)
        self.current_epoch = state_dict['current_epoch']
        self.loss_tracker = state_dict['loss_tracker']


class HuggingFaceAcceleratePretrainer(HuggingFaceAccelerateTrainer, Pretrainer):
    def __init__(self, configs: TrainingConfig, rank=None, num_processes=None, *args, **kwargs):
        """
        Trainer constructor.

        :param configs: TrainingConfig.
        :param rank: int. The current index of rank. This argument should be kept None unless multi_node
                     parallelization is intended and training is run using torch.multiprocessing.spawn
                     (instead of torchrun, where the rank can be automatically figured out and does not need
                     to be passed to the trainer explicitly).
        :param num_processes: int. The total number of processes. Similar to `rank`, this argument should be kept
                              None unless multi_node is intended and training is run using torch.multiprocessing.spawn.
        """
        HuggingFaceAccelerateTrainer.__init__(self, configs, rank, num_processes, *args, **kwargs)
        self.configs.task_type = None

    def update_saved_model(self, filename='best_model.pth', **kwargs):
        """
        HuggingFace Accelerate's save_state won't save the encoder parameters of a Siamese network at all
        because they are shared tensors. Even with safe_serialization set to False, it still complains about
        missing parameters in the encoder when loading state_dict. Therefore, we save and load models and
        states using the native PyTorch method. It will work because model objects in HuggingFace Accelerate
        are just PyTorch's DistributedDataParallelModels.

        This is really a makeshift and is causing inconsistency between HuggingFaceAccelerateTrainer and
        HuggingFaceAcceleratePretrainer. It should be fixed at some point.
        """
        if not filename.endswith('.pth'):
            filename = filename + '.pth'
        Pretrainer.update_saved_model(self, filename, **kwargs)

    def load_state_checkpoint(self):
        Pretrainer.load_state_checkpoint(self)

    def load_model(self, **kwargs):
        Pretrainer.load_model(self, **kwargs)


class PyTorchLightningTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        import lightning
        self.llib = lightning

        super().__init__(*args, **kwargs)
        self.lightning_model = None

    def build(self):
        self.build_split_datasets()
        self.build_dataloaders()
        self.build_model()
        self.build_dir()
        self.build_lightning_model()

    def add_lightning_methods_to_model_class(self):
        assert self.model_class_handle is not None
        setattr(self.model_class_handle, 'training_step', self.load_data_and_get_loss)
        setattr(self.model_class_handle, 'configure_optimizers', self.build_optimizer)

    def build_model(self):
        self.model_class_handle = self.configs.model_class

    def build_lightning_model(self):
        llib = self.llib
        model_class_handle = self.model_class_handle

        class LightningModel(llib.LightningModule, model_class_handle):
            def __init__(self, *args, **kwargs):
                llib.LightningModule.__init__(self)
                model_class_handle.__init__(self, *args, **kwargs)
                self.gtrainer = None

            def training_step(self, batch, batch_idx):
                return PyTorchLightningTrainer.load_data_and_get_loss(self, batch, batch_idx)

            def configure_optimizers(self):
                return PyTorchLightningTrainer.build_optimizer(self)

        self.lightning_model = LightningModel(**self.configs.model_params.__dict__)
        self.lightning_model.gtrainer = self

    def run_training(self):
        if self.configs.checkpoint_dir is None:
            checkpoint_path = None
        else:
            checkpoint_path = os.path.join(self.configs.checkpoint_dir, 'checkpoint_model.pth')

        lightning_trainer = self.llib.Trainer(
            max_epochs=self.configs.num_epochs
        )
        lightning_trainer.fit(self.lightning_model,
                              train_dataloaders=self.training_dataloader,
                              val_dataloaders=self.validation_dataloader,
                              ckpt_path=checkpoint_path
        )

    @staticmethod
    def load_data_and_get_loss(lightning_model, batch, batch_idx, *args, **kwargs):
        """
        Load data, run prediction, calculate loss, and return the loss.

        This defines the training_step method for PyTorch Lightning.
        """
        assert isinstance(lightning_model.gtrainer, Trainer)
        if hasattr(lightning_model.gtrainer.loss_criterion, '__len__'):
            n_losses = len(lightning_model.gtrainer.loss_criterion) + 1
        else:
            n_losses = len(lightning_model.gtrainer.configs.pred_names) + 1
        data, labels = lightning_model.gtrainer.process_data_loader_yield(batch)
        preds = lightning_model(*data)
        _, total_loss_tensor = lightning_model.gtrainer.compute_losses([0.0] * n_losses, preds, labels)
        return total_loss_tensor

    @staticmethod
    def build_optimizer(lightning_model):
        assert isinstance(lightning_model.gtrainer, Trainer)
        if isinstance(lightning_model.gtrainer.configs.optimizer, str):
            if lightning_model.gtrainer.configs.optimizer == 'adam':
                lightning_model.gtrainer.optimizer = torch.optim.Adam(lightning_model.parameters(),
                                                                      lr=lightning_model.gtrainer.learning_rate)
        else:
            lightning_model.gtrainer.optimizer = lightning_model.gtrainer.configs.optimizer(
                lightning_model.parameters(),
                lr=lightning_model.gtrainer.learning_rate,
                **lightning_model.gtrainer.configs.optimizer_params
            )
        return lightning_model.gtrainer.optimizer
