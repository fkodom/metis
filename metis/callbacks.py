"""
callbacks.py
-------------
Callbacks for use during model training
"""

from typing import AnyStr, Callable

import torch


def checkpoint(module: torch.nn.Module, save_path: AnyStr) -> Callable:
    """Defines a callback for saving PyTorch models during training.  The callback function will always accept `self`
    as the first argument.  We include `*args` positional arguments for added flexibility, so that a network could
    pass multiple arguments (e.g. training/validation loss, epoch number, etc.) without breaking it.

    :param save_path:  Absolute path to the model's save file
    :return:  Callback function for model saving
    """
    # noinspection PyUnusedLocal
    def callback(*args, **kwargs):
        torch.save(module.state_dict(), save_path)

    return callback
