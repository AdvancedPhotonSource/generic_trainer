import torch


class DistributedDataLoader:
    """
    A dataloader for distributed training on HPCs.

    When using the built-in DataLoader from PyTorch on HPCs (mainly Polaris), we noticed that it took long to finish
    loading a batch (by calling dataloader.__iter__().__next__()), even though the __getitem__ calls for all samples
    in the batch would take just a fraction of that time. ALCF documentation pointed out a bug that requires setting
    num_workers = 0, but that didn't help. Also, we found the built-in dataloader not using the __getitems__ method
    in the dataset even when it is defined. So we created this minimalistic dataloader with only essential operations
    (mainly dataset.__getitems__) to work around that.
    """
    def __init__(self, dataset, batch_size, sampler, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.i_sample = 0
        self.i_batch = 0
        self.sampler_iter = iter(self.sampler)
        self.iter = self.__iter__()

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __iter__(self):
        return DistributedDataLoaderIterator(self.dataset, self.batch_size, self.sampler)

    def __next__(self):
        data = next(self.iter)
        return data


class DistributedDataLoaderIterator:

    def __init__(self, dataset, batch_size, sampler, collate_fn=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.sampler = sampler
        self.i_sample = 0
        self.i_batch = 0
        self.sampler_iter = iter(self.sampler)

    def reset(self):
        self.i_sample = 0
        self.i_batch = 0

    def __len__(self):
        return len(self.sampler) // self.batch_size

    def __next__(self):
        if self.i_batch >= self.__len__():
            self.reset()
            raise StopIteration

        # Get indices for this batch
        inds = []
        for i in range(self.batch_size):
            inds.append(next(self.sampler_iter))
        inds = tuple(inds)

        self.i_sample += self.batch_size
        self.i_batch += 1
        try:
            data = self.dataset.__getitems__(inds)
        except AttributeError:
            raw_data = [self.dataset[i] for i in inds]
            data = []
            for i in range(len(raw_data[0])):
                data.append(torch.cat([raw_data[j][i] for j in range(len(raw_data))]))
        return data
