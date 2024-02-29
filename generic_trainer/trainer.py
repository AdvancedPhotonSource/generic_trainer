import os
import itertools
import copy
import re
import warnings

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.model_selection import train_test_split

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from .configs import *
from .util import *
from .compat import *
from .message_logger import logger


class LossTracker(dict):

    def __init__(self, pred_names=('cs', 'eg', 'sg'), require_cs_labels=False, *args, **kwargs):
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
        :param require_cs_labels: bool. When True, the loss tracker will expect CS labels as the last element in the
                                  label list when calculating accuracies. This is usually
                                  used for calculating the accuracies of CS deduced from SG predictions, without
                                  an actual CS classification head.
        """
        super().__init__(*args, **kwargs)
        self.pred_names = pred_names
        self.require_cs_labels = require_cs_labels
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
        # Also calculate the CS accuracy deduced from SG predictions.
        if 'sg' in self.pred_names:
            self['classification_preds_cs_from_sg'] = []
            self['train_acc_cs_from_sg'] = []
            self['val_acc_cs_from_sg'] = []
        if self.require_cs_labels and ('cs' not in self.pred_names):
            self['classification_labels_cs'] = []

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
        logger.info('Epoch: %d | All | Train Loss: %.5f | Val Loss: %.5f' % (
            self.current_epoch, self['loss'][-1], self['val_loss'][-1]))
        for i_pred, pred_name in enumerate(self.pred_names):
            logger.info('Epoch: %d | %s  | Train Loss: %.4f | Val Loss: %.4f' % (
                self.current_epoch, pred_name.upper(),
                self['loss_{}'.format(pred_name)][-1],
                self['val_loss_{}'.format(pred_name)][-1]))
        if len(self['lrs']) > 0:
            logger.info('Epoch: %d | Ending LR: %.6f ' % (self.current_epoch, self['lrs'][-1]))

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
        if 'classification_preds_cs_from_sg' in self.keys():
            self['classification_preds_cs_from_sg'] = []
        if self.require_cs_labels:
            self['classification_labels_cs'] = []

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
        if 'classification_preds_cs_from_sg' in self.keys():
            inds_cs_from_sg = self.get_cs_from_sg_predictions(pred_dict['sg'])
            self['classification_preds_cs_from_sg'] += inds_cs_from_sg.tolist()
        if self.require_cs_labels and 'cs' not in self.pred_names:
            assert len(labels) > len(self.pred_names), ('require_cs_labels is True, so I am expecting CS labels even '
                                                        'pred_names does not include it. Make sure dataset returns '
                                                        'CS labels at last. ')
            inds_label = torch.argmax(labels[-1], dim=1)
            self['classification_labels_cs'] += inds_label.tolist()

    def get_cs_from_sg_predictions(self, inds_sg):
        """
        Get the predicted classes for CS from the predictions of SG using their hierarchical relation.

        :param inds_sg: torch.tensor. The predicted DG indices (not one-hot).
        :return: torch.tensor. 1D tensor of predicted CS indices.
        """
        sg_group_starting_inds = torch.tensor(consts.cs_to_sg_index_bracket_array[:, 0], device=inds_sg.device)
        diff_array = inds_sg.view(-1, 1) - sg_group_starting_inds
        diff_array[diff_array < 0] = diff_array.max() + 1
        cs_inds = torch.argmin(diff_array, dim=1)
        return cs_inds

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
        if 'classification_preds_cs_from_sg' in self.keys():
            inds_pred = self['classification_preds_cs_from_sg']
            inds_label = self['classification_labels_cs']
            acc = np.mean((np.array(inds_pred) == np.array(inds_label)))
            acc_dict['cs_from_sg'] = acc
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
        if 'cs_from_sg' in acc_dict.keys():
            self['{}_acc_cs_from_sg'.format(type)].append(acc_dict['cs_from_sg'])


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
        if self.rank == 0:
            if not os.path.exists(self.configs.model_save_dir):
                os.makedirs(self.configs.model_save_dir)
        self.barrier()

    def build_device(self):
        self.device = self.get_device()
        self.num_local_devices = self.get_num_local_devices()

    def build_ranks(self):
        self.rank = self.get_rank()
        self.num_processes = self.get_num_processes()

    def build_scalable_parameters(self):
        self.all_proc_batch_size = self.configs.batch_size_per_process * self.num_processes
        self.learning_rate = self.configs.learning_rate_per_process * self.num_processes

    def build_dataloaders(self):
        shuffle = True
        drop_last = False
        training_sampler = None
        validation_sampler = None
        # Need double check on this.
        if self.parallelization_type == 'multi_node':
            training_sampler = torch.utils.data.distributed.DistributedSampler(
                self.training_dataset, num_replicas=self.num_processes, rank=self.rank, shuffle=True)
            validation_sampler = torch.utils.data.distributed.DistributedSampler(
                self.validation_dataset, num_replicas=self.num_processes, rank=self.rank, shuffle=True)
            shuffle = False
            drop_last = True
        # ALCF documentation mentions that there is a bug in Pytorch's multithreaded data loaders with
        # distributed training across multiple nodes. Therefore, `num_workers` is set to 0. See also:
        # https://docs.alcf.anl.gov/polaris/data-science-workflows/frameworks/pytorch/.
        self.training_dataloader = DataLoader(self.training_dataset, shuffle=shuffle,
                                              batch_size=self.all_proc_batch_size,
                                              collate_fn=lambda x: x, worker_init_fn=self.get_worker_seed_func(),
                                              generator=self.get_dataloader_generator(), num_workers=0,
                                              sampler=training_sampler, drop_last=drop_last)
        self.validation_dataloader = DataLoader(self.validation_dataset, shuffle=shuffle,
                                                batch_size=self.all_proc_batch_size,
                                                collate_fn=lambda x: x, worker_init_fn=self.get_worker_seed_func(),
                                                generator=self.get_dataloader_generator(), num_workers=0,
                                                sampler=validation_sampler, drop_last=drop_last)

    def run_training(self):
        for self.current_epoch in range(self.current_epoch, self.num_epochs):
            # Set model to train mode and run training
            self.model.train()
            self.run_training_epoch()

            # Switch model to eval mode and run validation
            self.model.eval()
            self.run_validation()

            if self.verbose:
                self.loss_tracker.print_losses()
        self.update_saved_model(filename='final_model.pth')

    def compute_losses(self, loss_records, preds, labels):
        """
        Run the model with the data of the current iteration and get losses.

        :param loss_records: list[float]. A list that keep tracks of the accumulated losses in the current epoch.
                             These values are just for record keeping and are not tensors.
        :param preds: list[torch.Tensor]. The list of predictions.
        :return: list, torch.Tensor. Updated loss records and total loss tensor.
        """
        # Compute losses
        total_loss_tensor = 0.0
        for i_pred in range(len(preds)):
            if isinstance(self.loss_criterion, Callable):
                this_loss_func = self.loss_criterion
            else:
                this_loss_func = self.loss_criterion[i_pred]
            this_loss_tensor = this_loss_func(preds[i_pred], labels[i_pred])
            total_loss_tensor = total_loss_tensor + this_loss_tensor
            loss_records[i_pred + 1] += this_loss_tensor.detach().item()
        if hasattr(self.loss_criterion, '__len__') and len(self.loss_criterion) > len(preds):
            pred_dict = self.get_pred_dict(preds)
            for i in range(len(preds), len(self.loss_criterion)):
                this_loss_func = self.loss_criterion[i]
                this_loss_tensor = this_loss_func(**pred_dict)
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

    def process_data_loader_yield(self, data_and_labels):
        """
        Disentangles the yields from the dataloader, returning a tuple of (data, label1, label2, ...) with
        each element being a tensor of [batch_size_per_process, ...].

        With the collate_fn defined, the yields of dataloader are different between PyTorch 1.x and 2.x. This
        function automatically detects the format and treat the data accordingly.
        """
        if self.parallelization_type == 'multi_node':
            bsize_per_rank = self.configs.batch_size_per_process
            if isinstance(data_and_labels[0], tuple):
                # In this case, data_and_labels is organized in a sample-then-item order:
                # it is a tuple of samples. Each element of the tuple is another tuple
                # containing the data and labels of that sample.
                data_and_labels = data_and_labels[self.rank * bsize_per_rank:(self.rank + 1) * bsize_per_rank]
                data = []
                for i in range(len(data_and_labels)):
                    data.append(data_and_labels[i][0])
                data = torch.concat(data, dim=0).to(self.device)
                n_labels = len(data_and_labels[0]) - 1
                label_list = [[] for i in range(n_labels)]
                for i_item in range(n_labels):
                    for i_sample in range(len(data_and_labels)):
                        label_list[i_item].append(data_and_labels[i_sample][i_item + 1])
                labels = [torch.concat(label_list[i]).to(self.device) for i in range(len(label_list))]
            else:
                # In this case, data_and_labels is organized in a item-then-sample order:
                # it is a tuple of items. Each element of the tuple is a tensor of
                # [n_total_batch_size, ...].
                data = data_and_labels[0][self.rank * bsize_per_rank:(self.rank + 1) * bsize_per_rank].to(self.device)
                labels = []
                for i in range(1, len(data_and_labels)):
                    labels.append(
                        data_and_labels[i][self.rank * bsize_per_rank:(self.rank + 1) * bsize_per_rank].to(self.device)
                    )
        else:
            if isinstance(data_and_labels[0], tuple):
                # In this case, data_and_labels is in sample-then-item order.
                data = []
                for i in range(len(data_and_labels)):
                    data.append(data_and_labels[i][0])
                data = torch.concat(data, dim=0).to(self.device)
                n_labels = len(data_and_labels[0]) - 1
                label_list = [[] for i in range(n_labels)]
                for i_item in range(n_labels):
                    for i_sample in range(len(data_and_labels)):
                        label_list[i_item].append(data_and_labels[i_sample][i_item + 1])
                labels = [torch.concat(label_list[i]).to(self.device) for i in range(len(label_list))]
            else:
                # In this case, data_and_labels is in item-then-sample order.
                data = data_and_labels[0].to(self.device)
                labels = [data_and_labels[i].to(self.device) for i in range(1, len(data_and_labels))]
        return data, labels

    def get_epoch_loss_buffer(self):
        if hasattr(self.loss_criterion, '__len__'):
            n = len(self.loss_criterion) + 1
        else:
            n = self.loss_tracker.n_preds + 1
        return [0.0] * n

    def run_training_epoch(self):
        losses = self.get_epoch_loss_buffer()
        n_batches = 0
        if self.configs.task_type == 'classification':
            self.loss_tracker.clear_classification_results_and_labels()
        for i, data_and_labels in enumerate(tqdm(self.training_dataloader, disable=(not self.verbose))):
            data, labels = self.process_data_loader_yield(data_and_labels)
            preds = self.model(data)
            losses, total_loss_tensor = self.compute_losses(losses, preds, labels)
            if self.configs.task_type == 'classification':
                self.loss_tracker.update_classification_results_and_labels(preds, labels)

            # Zero current grads and do backprop
            self.optimizer.zero_grad()
            total_loss_tensor.backward()
            self.optimizer.step()

            # Update the LR according to the schedule -- CyclicLR updates each batch
            self.scheduler.step()
            self.loss_tracker['lrs'].append(self.scheduler.get_last_lr()[0])

            n_batches += 1
        # Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        self.loss_tracker.update_losses([l / n_batches for l in losses], type='loss', epoch=self.current_epoch)

        if self.configs.task_type == 'classification':
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
            data, labels = self.process_data_loader_yield(data_and_labels)
            preds = self.model(data)
            losses, _ = self.compute_losses(losses, preds, labels)
            if self.configs.task_type == 'classification':
                self.loss_tracker.update_classification_results_and_labels(preds, labels)
            n_batches += 1
        if n_batches == 0:
            logger.warning('Validation set might be too small that at least 1 rank did not get any validation data.')
        n_batches = np.max([n_batches, 1])
        last_best_val_loss = self.loss_tracker['best_val_loss']
        is_best = self.loss_tracker.update_losses([l / n_batches for l in losses],
                                                  epoch=self.current_epoch, type='val_loss')
        self.write_training_info()

        # Update saved model if val loss is lower
        if is_best:
            logger.info("Saving improved model after Val Loss improved from %.5f to %.5f" % (
                last_best_val_loss, self.loss_tracker['best_val_loss']))
            self.update_saved_model(filename='best_model.pth')

        if self.configs.task_type == 'classification':
            acc_dict = self.loss_tracker.calculate_classification_accuracy()
            self.loss_tracker.update_accuracy_history(acc_dict, 'val')

        if self.configs.post_validation_epoch_hook is not None:
            self.configs.post_validation_epoch_hook()

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
        self.model = self.configs.model_class(**self.configs.model_params.__dict__)

        # For single_node parallelization using DataParallel model, checkpoint should be loaded before
        # DataParallel(model).
        if self.parallelization_type == 'single_node':
            if self.configs.pretrained_model_path is not None:
                if self.configs.load_pretrained_encoder_only:
                    logger.info('Loading pretrained encoder from {}.'.format(self.configs.pretrained_model_path))
                    self.load_model(self.configs.pretrained_model_path, subcomponent='backbone_model')
                else:
                    logger.info('Loading pretrained model from {}.'.format(self.configs.pretrained_model_path))
                    self.load_model(self.configs.pretrained_model_path)
            elif self.configs.checkpoint_dir is not None:
                logger.info('Loading checkpointed model from {}.'.format(self.configs.checkpoint_dir))
                self.load_model(os.path.join(self.configs.checkpoint_dir, 'checkpoint_model.pth'))

        self.build_parallelism()

        # For multi_node parallelization using DistributedDataParallel (DDP), checkpoint should be loaded after
        # DistributedDataParallel(model).
        if self.parallelization_type == 'multi_node':
            if self.configs.pretrained_model_path is not None:
                logger.info('Loading pretrained model from {}.'.format(self.configs.pretrained_model_path))
                self.load_model(self.configs.pretrained_model_path)
            elif self.configs.checkpoint_dir is not None:
                logger.info('Loading checkpointed model from {}.'.format(self.configs.checkpoint_dir))
                self.load_model(os.path.join(self.configs.checkpoint_dir, 'checkpoint_model.pth'))

    def build_parallelism(self):
        if self.parallelization_type == 'single_node':
            if self.device.type == 'cuda' and self.num_local_devices > 1:
                self.model = nn.DataParallel(self.model)
                logger.info('Using DataParallel with {} devices.'.format(self.num_local_devices))
            self.model.to(self.device)

        elif self.parallelization_type == 'multi_node':
            if self.device.type == 'cuda' and len(get_cuda_visible_devices_from_environ()) != 1:
                logger.warning('Parallelization type is multi_node, but CUDA_VISIBLE_DEVICES is {}. This variable '
                               'should be set to exactly 1 GPU for each process before MPI initialization.'.
                               format(os.environ['CUDA_VISIBLE_DEVICES']))
            logger.info('Sending model on rank {} to device {}.'.format(self.rank, self.rank))
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
                map_location = {'cuda:%d' % 0: 'cuda:%d' % self.rank}
                state_dict = torch.load(path, map_location=map_location)
        if subcomponent is None:
            self.model.load_state_dict(state_dict)
        else:
            getattr(self.model, subcomponent).load_state_dict(state_dict)

    def save_model(self, path, subcomponent=None):
        if subcomponent is None:
            m = self.model
        else:
            try:
                m = getattr(self.model, subcomponent)
            except:
                m = getattr(self.model.module, subcomponent)
        if self.rank == 0:
            try:
                torch.save(m.module.state_dict(), path)
            except AttributeError:
                torch.save(m.state_dict(), path)
        self.barrier()

    def update_saved_model(self, filename='best_model.pth', save_configs=True, subcomponent=None):
        """
        Updates saved model if validation loss is minimum.
        """
        path = self.configs.model_save_dir
        dest_path = os.path.join(path, filename)
        if self.rank == 0:
            if not os.path.isdir(path):
                os.mkdir(path)
            if os.path.exists(dest_path):
                os.remove(dest_path)
        self.barrier()
        self.save_model(dest_path, subcomponent=subcomponent)
        if self.rank == 0 and save_configs:
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
        state_dict = self.generate_state_dict()
        torch.save(state_dict, os.path.join(self.configs.model_save_dir, 'checkpoint.state'))
        self.update_saved_model('checkpoint_model.pth')

    def load_state_checkpoint(self):
        if self.configs.checkpoint_dir is None:
            return
        checkpoint_fname = os.path.join(self.configs.checkpoint_dir, 'checkpoint.state')
        if not os.path.exists(checkpoint_fname):
            logger.warning('Checkpoint not found in {}.'.format(checkpoint_fname))
        state_dict = torch.load(checkpoint_fname)
        self.current_epoch = state_dict['current_epoch']
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        if state_dict['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(state_dict['scheduler_state_dict'])
        self.loss_tracker = state_dict['loss_tracker']

    def write_training_info(self):
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

    def process_data_loader_yield(self, data):
        """
        Disentangles the yields from the dataloader, returning a tuple of (data1, data2) with
        each element being a tensor of [batch_size_per_process, ...].

        With the collate_fn defined, the yields of dataloader are different between PyTorch 1.x and 2.x. This
        function automatically detects the format and treat the data accordingly.
        """
        if self.parallelization_type == 'multi_node':
            bsize_per_rank = self.configs.batch_size_per_process
            if isinstance(data[0], tuple):
                # In this case, data_and_labels is organized in a sample-then-item order:
                # it is a tuple of samples. Each element of the tuple is another tuple
                # containing the data and labels of that sample.
                data_chunk = data[self.rank * bsize_per_rank:(self.rank + 1) * bsize_per_rank]
                data = []
                for i in range(len(data_chunk)):
                    data.append(data_chunk[i][0])
                data = torch.concat(data, dim=0).to(self.device)
                n_data = len(data[0])
                data_list = [[] for i in range(n_data)]
                for i_item in range(n_data):
                    for i_sample in range(len(data)):
                        data_list[i_item].append(data[i_sample][i_item])
                data = [torch.concat(data_list[i]).to(self.device) for i in range(len(data_list))]
            else:
                # In this case, data_and_labels is organized in a item-then-sample order:
                # it is a tuple of items. Each element of the tuple is a tensor of
                # [n_total_batch_size, ...].
                data_list = []
                for i in range(len(data)):
                    data_list.append(
                        data[i][self.rank * bsize_per_rank:(self.rank + 1) * bsize_per_rank].to(self.device)
                    )
                data = data_list
        else:
            if isinstance(data[0], tuple):
                # In this case, data_and_labels is in sample-then-item order.
                n_data = len(data[0])
                data_list = [[] for i in range(n_data)]
                for i_item in range(n_data):
                    for i_sample in range(len(data)):
                        data_list[i_item].append(data[i_sample][i_item])
                data = [torch.concat(data_list[i]).to(self.device) for i in range(len(data_list))]
            else:
                # In this case, data_and_labels is in item-then-sample order.
                data = [data[i].to(self.device) for i in range(len(data))]
        # data needs to be a 3D tensor of [n_batches, 1, spec_len]. The 2nd dimension is necessary because it is
        # treated as the channel dimension.
        for i, d in enumerate(data):
            if len(d.shape) == 2:
                data[i] = d[:, None, :]
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

    def run_training_epoch(self):
        losses = self.get_epoch_loss_buffer()
        n_batches = 0
        for i, data in enumerate(tqdm(self.training_dataloader, disable=(not self.verbose))):
            # elements of data are supposed to be 2 different augmentations.
            data = self.process_data_loader_yield(data)
            preds = self.model(*data)
            losses, total_loss_tensor = self.compute_losses(losses, preds)

            # Zero current grads and do backprop
            self.optimizer.zero_grad()
            total_loss_tensor.backward()
            self.optimizer.step()

            # Update the LR according to the schedule -- CyclicLR updates each batch
            self.scheduler.step()
            self.loss_tracker['lrs'].append(self.scheduler.get_last_lr()[0])

            n_batches += 1
        # Divide cumulative loss by number of batches-- sli inaccurate because last batch is different size
        self.loss_tracker.update_losses([l / n_batches for l in losses], type='loss', epoch=self.current_epoch)

        self.save_model_and_states_checkpoint()

        if self.configs.post_training_epoch_hook is not None:
            self.configs.post_training_epoch_hook()

    def run_validation(self):
        losses = self.get_epoch_loss_buffer()
        n_batches = 0
        for j, data in enumerate(self.validation_dataloader):
            data = self.process_data_loader_yield(data)
            preds = self.model(*data)
            losses, _ = self.compute_losses(losses, preds)
            n_batches += 1
        if n_batches == 0:
            logger.warning('Validation set might be too small that at least 1 rank did not get any validation data.')
        n_batches = np.max([n_batches, 1])
        last_best_val_loss = self.loss_tracker['best_val_loss']
        is_best = self.loss_tracker.update_losses([l / n_batches for l in losses],
                                                  epoch=self.current_epoch, type='val_loss')
        self.write_training_info()

        # Update saved model if val loss is lower
        if is_best:
            logger.info("Saving improved model after Val Loss improved from %.5f to %.5f" % (
                last_best_val_loss, self.loss_tracker['best_val_loss']))
            self.update_saved_model(filename='best_model.pth')

        if self.configs.post_validation_epoch_hook is not None:
            self.configs.post_validation_epoch_hook()


    def update_saved_model(self, filename='best_model.pth'):
        """
        Updates saved model if validation loss is minimum.
        """
        super().update_saved_model(filename)
        encoder_filename = os.path.splitext(filename)[0] + '_encoder.pth'
        super().update_saved_model(encoder_filename, save_configs=False, subcomponent='encoder')


class AlphaDiffractHyperparameterScanner:
    def __init__(self, scan_params_dict: dict, base_config_dict: TrainingConfig, keep_models_in_memory=False):
        """
        Hyperparameter scanner.

        :param scan_params_dict: dict. A dictionary of the parameters to be scanned. The keys of the dictionary
                                 should be from `TrainingConfig`, and the value should be a list of values to test.
        :param base_config_dict: TrainingConficDict. A baseline config dictionary.
        """
        self.scan_params_dict = scan_params_dict
        self.result_table = None
        self.n_params = len(self.scan_params_dict)
        self.param_comb_list = None
        self.base_config_dict = base_config_dict
        # Might need to be a copy. Deepcopy is currently not done because of the H5py object in `dataset`.
        self.config_dict = self.base_config_dict
        self.trainer_list = []
        self.model_save_dir_prefix = self.base_config_dict['model_save_dir']
        self.keep_models_in_memory = keep_models_in_memory
        self.dummy_loss_tracker = LossTracker(self.config_dict['pred_names'])
        self.metric_names = self.dummy_loss_tracker.get_metric_names_for_hyperparam_tuning()
        self.verbose = True

    def build_result_table(self):

        self.param_comb_list = list(itertools.product(*self.scan_params_dict.values()))
        dframe_dict = {}
        for i_param, param in enumerate(self.scan_params_dict.keys()):
            dframe_dict[param] = []
            for i_comb in range(len(self.param_comb_list)):
                v = self.param_comb_list[i_comb][i_param]
                v = self.convert_item_to_be_dataframe_compatible(v)
                dframe_dict[param].append(v)
        for metric in self.metric_names:
            dframe_dict[metric] = [0.0] * len(self.param_comb_list)
        self.result_table = pd.DataFrame(dframe_dict)

    def build(self, seed=123):
        if seed is not None:
            set_all_random_seeds(seed)
        self.build_result_table()

    def convert_item_to_be_dataframe_compatible(self, v):
        if isinstance(v, nn.Module):
            nv = v._get_name()
        elif isinstance(v, (tuple, list)) and issubclass(v[0], nn.Module):
            nv = v[0].__name__
            if len(v[1]) > 0:
                nv += '_' + self.convert_dict_to_string(v[1])
        else:
            nv = v
        return nv

    def modify_condig_dict(self, param_dict):
        for i, param in enumerate(param_dict.keys()):
            # if param == 'model':
            #     # For testing different models, the input in `scan_params_dict['model']` is supposed to be a list of
            #     # 2-tuples, where the first element is the class handle of the model, and the second element is a
            #     # dictionary of keyword arguments in the constructor of that class.
            #     self.config_dict[param] = param_dict[param][0](**param_dict[param][1])
            # else:
            self.config_dict[param] = param_dict[param]
        # Update save path.
        appendix = self.convert_dict_to_string(param_dict)
        self.config_dict['model_save_dir'] = self.model_save_dir_prefix + '_' + appendix

    @staticmethod
    def convert_string_to_camel_case(s):
        s = re.sub(r"(_|-)+", " ", s).title().replace(" ", "")
        return ''.join([s[0].lower(), s[1:]])

    def convert_dict_to_string(self, d):
        s = ''
        for i, (k, v) in enumerate(d.items()):
            s += self.convert_string_to_camel_case(k)
            s += '_'
            s += str(self.convert_item_to_be_dataframe_compatible(v))
            if i < len(d) - 1:
                s += '_'
        return s

    def create_param_dict(self, config_val_list: list):
        d = {}
        for i in range(len(config_val_list)):
            param_name = list(self.scan_params_dict.keys())[i]
            d[param_name] = config_val_list[i]
        return d

    def run(self):
        for i_comb in tqdm(range(len(self.param_comb_list))):
            param_dict = self.create_param_dict(self.param_comb_list[i_comb])
            self.modify_condig_dict(param_dict)
            trainer = AlphaDiffractTrainer(self.config_dict)
            self.run_trainer(trainer)
            self.trainer_list.append(trainer)
            self.update_result_table(i_comb, trainer)
            self.cleanup()

    def run_trainer(self, trainer):
        trainer.verbose = False
        trainer.build()
        trainer.run_training()

    def plot_all_training_history(self):
        for i_comb in range(len(self.param_comb_list)):
            print('Training history for the following config - ')
            print(self.result_table.iloc[i_comb])
            trainer = self.trainer_list[i_comb]
            trainer.plot_training_history()

    def load_model_for_trainer(self, trainer):
        # Reinitialize with a brand new model object
        assert isinstance(trainer.configs['model'], (tuple, list)), \
            '`config_dict["model"]` should be a tuple of class handle and kwargs.'
        trainer.build_model()
        trainer.configs['model_path'] = os.path.join(trainer.configs['model_save_dir'], 'best_model.pth')
        trainer.load_state_checkpoint()

    def run_testing_for_all(self, indices, dataset='train'):
        """
        Run test for all trained models with selected samples and plot results.

        :param indices: tuple.
        :param dataset: str. Can be 'train' or 'validation'.
        """
        for i_comb in range(len(self.param_comb_list)):
            print('Testing results for the following config - ')
            print(self.result_table.iloc[i_comb])
            trainer = self.trainer_list[i_comb]
            if not self.keep_models_in_memory:
                # If the models of trainers were not kept in memory, load them back from hard drive.
                self.load_model_for_trainer(trainer)
            trainer.run_testing(indices, dataset=dataset)

    def update_result_table(self, i_comb, trainer):
        for metric_name in self.metric_names:
            self.result_table.at[i_comb, metric_name] = trainer.loss_tracker[metric_name]

    def cleanup(self):
        if self.verbose:
            get_gpu_memory(show=True)
        if not self.keep_models_in_memory:
            # Destroy model to save memory.
            del self.trainer_list[-1].model
            self.trainer_list[-1].model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
