import os 
import torch
import copy
import random 
import pandas as pd
from torchtrainer.model import Model
from torch.utils.tensorboard import SummaryWriter
from torchtrainer.utils import _trainer_helper as helper
from torch.profiler import profile, record_function, ProfilerActivity, schedule
import torchtrainer.utils._distribute as dist

class Trainer:
    def __init__(self, model: Model, train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset, 
                 epoch_num: int, batch_size: int, shuffle: bool, num_workers: int, summary_args: dict, 
                 device=None, collate_fn=None, worker_init=None, distributed=False, print_stats=True, tb_logs=False, tb_embeddings=False, save_path=None,
                 tb_embeddings_num=None):
        """
        Parameters 
        __________

        summary_args : dict, required
            Is used for printing the summary before training
        tb_logs : bool, default=False
            Specify if you want to save tensorboard logs 
        tb_embeddings : bool, default=False
            Specify if you want to save embeddings for tensorboard projections     

        """
        # model info
        self.model = model
        self.model_name = model.name
        # data info
        self.train_dataset = train_dataset 
        self.valid_dataset = valid_dataset
        self.bs = batch_size 
        self.shuffle = shuffle
        self.workes = num_workers
        self.collate_fn = collate_fn
        self.worker_init = worker_init
        # train info
        self.epoch_num = epoch_num
        self._multi_train = False
        self._conf_num = 0 
        # device info
        self.gpu_num = torch.cuda.device_count()
        self.device = device
        self.model._device = device
        # init stats on the model
        self.model.init_stats()
        # args 
        self.args = summary_args
        self.print_stats = print_stats
        # tensorboard stuffs
        self.tb_embeddings = tb_embeddings
        self.tb_logs = tb_logs
        self.save_path = save_path
        self.tb_embeddings_num = tb_embeddings_num

        if distributed and self.gpu_num < 2: 
            raise ValueError(f"cannot use distributed training, only {self.gpu_num} gpu are present")

        # set true only if distributed is enable and device is not set to any particular one
        # at this point we are also sure to have at least 2 gpus for training
        self._distributed = (distributed and (self.device is None))

        if tb_logs and save_path is None: 
            raise ValueError("save_path should be defined if tb_logs is True")
        if tb_embeddings and tb_embeddings_num is None: 
            raise ValueError("tb_embeddings_num should be defined if tb_embeddings is True")
        if self.model.load_path is not None:
            self.model.load_state_dict(torch.load(os.path.join(os.getcwd(), self.model.load_path),
                                                   map_location=device))

        # set seeds
        torch.manual_seed(0)
        random.seed(0)

    def _check_device(self, device):
        if device is None:
            return torch.device('cuda:'+str(helper.get_free_gpu()) if torch.cuda.is_available() else 'cpu')
        return device  

    def _send_to_device(self, data: dict, device):
        data.update({k: d.to(device) for k, d in data.items()})

    def _print_epoch_stats(self, train_stats, current_epoch, i, train_len, num_train_batches, valid_len, last_lr, end=False, valid_stats=None):
        out_dict = {k: v.item() / train_len for k, v in train_stats.items()}

        if end:
            out_dict.update({k: v.item() / (valid_len) for k, v in valid_stats.items()})

        helper.print_epoch_stats(current_epoch, i, num_train_batches, last_lr, 
                                 end, **out_dict)

    def _setup_tensorboard(self):
        self.tb_logdir = helper.create_logs_dir(self.save_path)    
        self.tb_writer = SummaryWriter(log_dir=self.tb_logdir)

    def _save_epoch_stats(self, epoch, train_len, valid_len, train_stats, valid_stats):
        for key, value in train_stats.items():
            self.tb_writer.add_scalar('Train/{}'.format(key), value / train_len, epoch)

        for key, value in valid_stats.items():
            self.tb_writer.add_scalar('Valid/{}'.format(key), value / valid_len, epoch)

    def _save_results(self, train_len, valid_len, model):
        # save model
        torch.save(model.state_dict(), os.path.join(self.tb_logdir, self.model_name + '.pt'))
        # write hparameters to tensorboard
        # this can be useful to compare different runs with different hparams
        # TODO: check this
        hpar_dict = {'hparam/{}'.format(k): v / train_len for k, v in model.train_stats.items()}
        hpar_dict.update({'hparam/{}'.format(k): v / valid_len for k, v in model.valid_stats.items()})

        self.tb_writer.add_hparams(self.args, hpar_dict)
        self.tb_writer.flush()
        self.tb_writer.close()

    def _seed_workers(self, w_id):
        worker_seed = torch.initial_seed() % 2**32
        random.seed(worker_seed)
        
        if self.worker_init is not None: 
            self.worker_init()

    def _save_embeddings(self, model, device, valid_loader):
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_loader):
                # send to device
                self._send_to_device(data, device)

                # inference 
                embeddings, *metadata = model.embeddings_forward(data)

                if i == 0:
                    final_embeddings = embeddings
                    final_meta = metadata
                else:
                    final_embeddings = torch.vstack((final_embeddings, embeddings))
                    final_meta = [torch.hstack((f_m, m)) for f_m, m in zip(final_meta, metadata)]

                if i > int(self.tb_embeddings_num/embeddings.shape[0]): break

        print("\nSaving {} tensors for projection...".format(i*embeddings.shape[0]))
        # save the embeddings + the metadata
        for i, meta in enumerate(final_meta):
            self.tb_writer.add_embedding(final_embeddings, metadata=meta, global_step=i)

    def train(self):
        if self._distributed:
            helper.print_summary(self.model, None, self.gpu_num, self.args, self._multi_train, self._conf_num) 
            dist.spawn_processes(self._train_loop, self.gpu_num)
        else:
            device = self._check_device(self.device)
            helper.print_summary(self.model, device, self.gpu_num, self.args, self._multi_train, self._conf_num) 
            self._train_loop(device, 0)

    # gpu argument can either be a device or the index of the device
    # in the case of distributed training
    def _train_loop(self, gpu, world_size):
        device = torch.device(gpu)
        
        # we need to use a deep copy of the model on each subprocess in case of distributed training
        if self._distributed: 
            model = copy.deepcopy(self.model)
        else: 
            model = self.model

        # this is done in place
        model.to(device)
        model._device = device

        # distributed=false -> master=True (only one device that is the master)
        # distributed=True  -> (multiple gpus, only cuda:0 is the master)
        master = ((not self._distributed) or device.index == 0)

        optimizer, scheduler = model.define_optimizer_scheduler()
        train_sampler = None
        valid_sampler = None

        if self._distributed:
            # setup of each process
            dist.setup(rank=gpu, world_size=world_size)

            model = dist.DDP(model, device_ids=[gpu])

            # set up distributed sampler
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                                self.train_dataset,
                                num_replicas=world_size,
                                rank=gpu, 
                                shuffle=self.shuffle)

            valid_sampler = torch.utils.data.distributed.DistributedSampler(
                                self.valid_dataset,
                                num_replicas=world_size,
                                rank=gpu, 
                                shuffle=False)
            
        g = torch.Generator()
        g.manual_seed(0)

        # if the device is a gpu we pin the memory on the dataloader for faster transfer of data
        pin_mem = (device.type == 'cuda')

        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs, sampler=train_sampler, 
                                                   num_workers=self.workes, collate_fn=self.collate_fn, pin_memory=pin_mem, 
                                                   worker_init_fn=self._seed_workers, generator=g)

        valid_loader = torch.utils.data.DataLoader(self.valid_dataset, batch_size=self.bs, sampler=valid_sampler, 
                                                   num_workers=self.workes, collate_fn=self.collate_fn, pin_memory=pin_mem, 
                                                   worker_init_fn=self._seed_workers, generator=g)

        # num of train batches
        train_batch_num = len(train_loader) - 1
        # num of samples in the train dataset
        valid_len = len(valid_loader.dataset)

        if self.tb_logs:
            self._setup_tensorboard()

        # training loop
        for epoch in range(self.epoch_num): 
            model.reset_stats()
            model.train()

            # see https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            if self._distributed:
                train_sampler.set_epoch(epoch)

            for i, data in enumerate(train_loader):
                # send to device
                self._send_to_device(data, device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # training step 
                loss = model.training_step(data)

                # backward + optimize
                loss.backward()
                optimizer.step()

                # (print_stats && (!distributed or device == gpu:0))
                if self.print_stats and master:
                    self._print_epoch_stats(model.train_stats, epoch, i, (i+1) * self.bs, train_batch_num, valid_len, scheduler.get_last_lr()[0])

            # compute validation loss and acc at the end of each epoch
            torch.cuda.empty_cache()
            model.eval()
            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    # send to device
                    self._send_to_device(data, device)

                    # valid step 
                    model.validation_step(data)

            if (not self._distributed) or device.index == 0:
                self._print_epoch_stats(model.train_stats, epoch, train_batch_num, len(train_loader.dataset), train_batch_num, valid_len, 
                                        scheduler.get_last_lr()[0], end=True, valid_stats=model.valid_stats)

            #lr decay step
            if scheduler is not None: 
                scheduler.step()
            
            if master and self.tb_logs:
                self._save_epoch_stats(epoch, train_len=len(train_loader.dataset), valid_len=len(valid_loader.dataset),
                                       train_stats=model.train_stats, valid_stats=model.valid_stats)

        # train end
        helper.print_end_train() 
    
        # tensorboard for saving results and embeddings
        if master and self.tb_logs:
            self._save_results(model=model, train_len=len(train_loader.dataset), valid_len=len(valid_loader.dataset))

        if master and self.tb_embeddings:
            self._save_embeddings(model, device, valid_loader)

    def multi_train(self, train_config_path):
        train_config_df = pd.read_csv(train_config_path, sep=' ')
        train_configs = train_config_df.to_dict(orient='records')

        # save initial model parameters 
        self._initial_model_param = copy.deepcopy(self.model.state_dict())
        self._multi_train = True
        
        helper.print_overall_summary(self.model, train_configs) 
        for i, configs in enumerate(train_configs):
            # set model attributes for the current config
            self.args.update(configs)
            for k, d in configs.items(): 
                setattr(self.model, k, d)

            # train on the current config
            self._conf_num = i
            self.train()

            # reset the parameters of the model
            self.model.reset_stats()
            self.model.load_state_dict(self._initial_model_param)
            torch.cuda.empty_cache()
