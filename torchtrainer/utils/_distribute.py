import os 
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

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

    def training_step(self, data):
        return self.module.training_step(data)

    def validation_step(self, data):
        return self.module.validation_step(data)

    def embeddings_forward(self, x):
        return self.module.embeddings_forward(x)

    def define_optimizer_scheduler(self):
        return self.module.define_optimizer_scheduler()