import torch
from typing import Callable, Any
from collections import defaultdict

class Model(torch.nn.Module):
    ''' 
        Model class that should be inhereted for using the Trainer.
        Children of the torch.nn.Module, so the user should also define the forward pass.
        Train step and Valid step are used to define the training and validation logic.

        In the training and valid step we can store values like loss and other stuff by using 
        the save_epoch_stats
    '''

    def __init__(self, name):
        super().__init__()
        # will be set by the Trainer
        self._device = None
        self.name = name
        
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
        self.train_stats = defaultdict(lambda: torch.tensor(0.0, device=self._device, 
                                       requires_grad=False))
        self.valid_stats = defaultdict(lambda: torch.tensor(0.0, device=self._device, 
                                       requires_grad=False))

    def reset_stats(self):
        self.train_stats.clear()
        self.valid_stats.clear()
        
    # should return the value of the loss function
    training_step: Callable[..., Any] = _train_step_unimplemented
    # may return the value of the loss, not important
    validation_step: Callable[..., Any] = _valid_step_unimplemented
    # defines the forward pass for getting in output embeddings
    # should returns embeddings
    embeddings_forward: Callable[..., Any] = _embeddings_forward_unimplemented
    # should return a tuple (optimizer, scheduler)
    define_optimizer_scheduler: Callable[..., Any] = _define_opt_sched_unimplemented
