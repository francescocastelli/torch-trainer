import torch
from typing import Callable, Any

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
        self.train_stats = dict()
        self.valid_stats = dict()
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
        for key, value in kwargs.items():
            if key in self.train_stats:
                self.train_stats[key] += value
            else: 
                self.train_stats[key] = torch.tensor(0.0, device=self._device, 
                                                       requires_grad=False) 
                self.train_stats[key] += value

    def save_valid_stats(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.valid_stats:
                self.valid_stats[key] += value
            else: 
                self.valid_stats[key] = torch.tensor(0.0, device=self._device, 
                                                       requires_grad=False)
                self.valid_stats[key] += value

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
