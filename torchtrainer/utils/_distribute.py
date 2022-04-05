import os 
from typing import Iterator, List, Optional 
from operator import itemgetter
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset, Sampler, DistributedSampler

'''
world_size is the total number of processes (in general one per gpu)
rank is the number of the current node (goes from 0 to tot_num_of_nodes)
rank is only useful is we train over multiple nodes
'''
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", init_method='env://', 
                            rank=rank, 
                            world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def spawn_processes(train_fn, world_size):
    # spawn will automatically pass the index of the process 
    # as first arg to train_fn
    mp.spawn(train_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

class DDP(DistributedDataParallel):
    def init_stats(self):
        self.module.init_stats()

    def reset_stats(self):
        self.module.reset_stats()

    def train_stats(self):
        return self.module.train_stats()

    def valid_stats(self):
        return self.module.valid_stats()

    def training_step(self, data):
        return self.module.training_step(data)

    def validation_step(self, data):
        return self.module.validation_step(data)

    def embeddings_forward(self, x):
        return self.module.embeddings_forward(x)

    def define_optimizer_scheduler(self):
        return self.module.define_optimizer_scheduler()

 
# https://github.com/catalyst-team/catalyst/
class DatasetFromSampler(Dataset):
    """
    Dataset to create indexes from `Sampler`.
    Args:
        sampler: PyTorch sampler
    """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
            index: index of the element in the dataset
        Returns:
            Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
            int: length of the dataset
        """
        return len(self.sampler)

# https://github.com/catalyst-team/catalyst/
class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
                distributed training
            rank (int, optional): Rank of the current process
                within ``num_replicas``
            shuffle (bool, optional): If true (default),
                sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
            python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))
