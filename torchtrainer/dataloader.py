import torch
import random 

class TrainerLoader():
    """ TrainerLoader:

    A wrapper over torch.utils.data.DataLoader 

    """
    
    def __init__(self,  batch_size: int, shuffle: bool = False, 
                 num_workers: int = 0, train_sampler: torch.utils.data.Sampler = None, 
                 valid_sampler: torch.utils.data.Sampler = None, collate_fn=None, worker_init=None):

        self.bs = batch_size 
        self.shuffle = shuffle
        self.workers = num_workers
        self.train_sampler = train_sampler
        self.valid_sampler = valid_sampler 
        self.collate_fn = collate_fn
        self.worker_init = worker_init
        self.g = torch.Generator()
        self.g.manual_seed(0)

        if self.train_sampler is None and self.valid_sampler is not None:
            raise ValueError("Valid sampler should be none since train sampler is None")

        if self.train_sampler is not None and self.valid_sampler is None:
            raise ValueError("Valid sampler should be none since train sampler is None")

        self.default_sampler = self.train_sampler == self.valid_sampler == None

    def _worker_init_fn(self, w_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        
        if self.worker_init is not None: 
            self.worker_init()

    def _get_distributed_loader(self, dataset, world_size, rank, device):
        # set up distributed sampler
        sampler = torch.utils.data.distributed.DistributedSampler(
                             dataset,
                             num_replicas=world_size,
                             rank=rank, 
                             shuffle=self.shuffle)
        
        return self._create_loader(dataset, sampler, device), sampler


    def _get_loader(self, dataset, device, mode):
        assert mode == 'train' or mode == 'valid', "mode should either be train or valid"

        # TODO: here we miss a default sampler if shuffle is false
        if self.default_sampler and self.shuffle: 
            sampler = torch.utils.data.RandomSampler(dataset, 
                                                     replacement=False, 
                                                     generator=self.g)
        else: 
            sampler = self.train_sampler if mode == 'train' else self.valid_sampler

        return self._create_loader(dataset, sampler, device)

    def _create_loader(self, dataset, sampler, device):
        # if the device is a gpu we pin the memory on the dataloader for faster transfer of data
        pin_mem = (device.type == 'cuda')
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.bs, 
                                             sampler=sampler, 
                                             num_workers=self.workers, 
                                             collate_fn=self.collate_fn, 
                                             pin_memory=pin_mem, 
                                             worker_init_fn=self._worker_init_fn, 
                                             generator=self.g)

        return loader
