import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    """
    A dummy dataset that generates random data for debugging.
    """

    def __init__(self, assumed_array_shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array_shape = assumed_array_shape

    def __len__(self):
        return self.array_shape[0]

    def get_labels(self, size, *args, **kwargs):
        return [torch.randint(0, 10, size)]

    def __getitem__(self, idx):
        x = torch.rand([1, *self.array_shape[1:]])
        labels = self.get_labels(1)
        return x, *labels

    def __getitems__(self, idx_list):
        n = len(idx_list)
        x = torch.rand([n, *self.array_shape[1:]])
        labels = self.get_labels(n)
        return x, *labels



class DummyClassificationDataset(DummyDataset):
    """
    A dummy dataset that generates random data for debugging.
    """

    def __init__(self, assumed_array_shape, label_dims=(7, 101, 230), add_channel_dim=False, *args, **kwargs):
        """
        The constructor.

        :param assumed_array_shape: list or tuple. The assumed array size that the dataset contains. For 1D data,
                                    this should be a 2D vector of (n_samples, n_features).
        :param label_dims: list or tuple. The lengths of one-hot encoded label vectors.
        """
        super().__init__(assumed_array_shape, *args, **kwargs)
        self.label_dims = label_dims
        self.add_channel_dim = add_channel_dim

    def get_labels(self, n, *args, **kwargs):
        labels = []
        for d in self.label_dims:
            label = torch.zeros([n, d])
            inds = torch.randint(0, d, (n,))
            label[tuple(range(n)), inds] = 1.0
            labels.append(label)
        return labels

    def __getitem__(self, idx):
        if self.add_channel_dim:
            x = torch.rand([1, 1, self.array_shape[-1]])
        else:
            x = torch.rand([1, self.array_shape[-1]])
        labels = self.get_labels(1)
        return x, *labels

    def __getitems__(self, idx_list):
        n = len(idx_list)
        if self.add_channel_dim:
            x = torch.rand([n, 1, self.array_shape[-1]])
        else:
            x = torch.rand([n, self.array_shape[-1]])
        labels = self.get_labels(n)
        return x, *labels


class DummyImageDataset(DummyDataset):

    def __init__(self, assumed_array_shape, label_shapes=((1, 32, 32),), *args, **kwargs):
        super().__init__(assumed_array_shape, *args, **kwargs)
        self.label_shapes = label_shapes

    def get_labels(self, size, *args, **kwargs):
        labels = []
        for label_shape in self.label_shapes:
            lab = torch.rand(size, *label_shape)
            labels.append(lab)
        return labels
