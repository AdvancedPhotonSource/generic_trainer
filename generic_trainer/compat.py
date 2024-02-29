import torch
import numpy


def set_default_device(dev):
    try:
        torch.set_default_device(dev)
    except:
        pass
