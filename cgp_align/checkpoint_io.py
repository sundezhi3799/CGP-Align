"""Load trusted research checkpoints with platform-independent saved path objects."""
import pathlib
import pickle
import sys

import torch


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('pathlib', 'pathlib._local') and name in ('PosixPath', 'WindowsPath'):
            return getattr(pathlib, 'Pure' + name)
        return super().find_class(module, name)


def load(file, **kwargs):
    return Unpickler(file, **kwargs).load()


def load_checkpoint(path, map_location='cpu'):
    """This uses pickle and must only be used with trusted checkpoint files."""
    return torch.load(path, map_location=map_location, weights_only=False,
                      pickle_module=sys.modules[__name__])
