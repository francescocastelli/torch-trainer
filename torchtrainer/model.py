import torch
from typing import Callable, Any
from functools import partial
from collections import defaultdict

# we need a top-module function to create the default dict values,
# since the multiprocessing requires pickable objects
def _get_zero_tensor(device):
    return torch.tensor(0.0, device=device, requires_grad=False)

def _default_stats(device):
    default_callable = partial(_get_zero_tensor, device=device) 
    return defaultdict(default_callable)

class Model(torch.nn.Module):
    ''' 
        Model class that should be inhereted for using the Trainer.
        Children of the torch.nn.Module, so the user should also define the forward pass.
        Train step and Valid step are used to define the training and validation logic.

        In the training and valid step we can store values like loss and other stuff by using 
        the save_epoch_stats
    '''

    def __init__(self, name, load_path=None):
        super().__init__()
        # will be set by the Trainer
        self._device = None
        self.name = name
        self.load_path = load_path
        
    def _train_step_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _valid_step_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _embeddings_forward_unimplemented(self, *input: Any):
        raise NotImplementedError

    def _define_opt_sched_unimplemented(self, *input: Any):
        raise NotImplementedError

    def save_train_stats(self, **kwargs):
        self.train_stats.update({k: self.train_stats[k] + d for k, d in kwargs.items()})

    def save_valid_stats(self, **kwargs):
        self.valid_stats.update({k: self.valid_stats[k] + d for k, d in kwargs.items()})

    def init_stats(self):
        self.train_stats = _default_stats(self._device)
        self.valid_stats = _default_stats(self._device) 

    def reset_stats(self):
        self.train_stats.clear()
        self.valid_stats.clear()
        
    '''
        These methods should be overrided by the user model class, otherwise throw 
        a NotImplemented exception.
    '''
    # should return the value of the loss function
    training_step: Callable[..., Any] = _train_step_unimplemented
    # may return the value of the loss, not important
    validation_step: Callable[..., Any] = _valid_step_unimplemented
    # defines the forward pass for getting in output embeddings
    # should returns embeddings
    embeddings_forward: Callable[..., Any] = _embeddings_forward_unimplemented
    # should return a tuple (optimizer, scheduler)
    define_optimizer_scheduler: Callable[..., Any] = _define_opt_sched_unimplemented
