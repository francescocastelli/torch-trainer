import os 
import torch
import copy
import random 
import json
from tqdm import tqdm
from torchtrainer.model import Model
from torch.utils.tensorboard import SummaryWriter
from torchtrainer.utils import _trainer_helper as helper
from torchtrainer.dataloader import TrainerLoader
import torchtrainer.utils._distribute as dist

# TODO: add profiling
#from torch.profiler import profile, record_function, ProfilerActivity, schedule

class Trainer:
    def __init__(self, model: Model, train_dataset: torch.utils.data.Dataset, 
                 valid_dataset: torch.utils.data.Dataset, loader: TrainerLoader,
                 epoch_num: int, summary_args: dict, seed: int=0, device=None, 
                 distributed: bool=False, verbose: bool=True, results_path: str=None, 
                 tb_logs: bool=False, tb_checkpoint_rate: int=0, checkpoint_path: str=None,
                 tb_embeddings_num: int=0):
        """
        Parameters 
        __________

        model (torchtrainer.model.Model): instance of torchtrainer Model, the actual model 
                                          that contains the training step and so on

        train_dataset (torch.utils.data.Dataset): dataset for training

        valid_dataset (torch.utils.data.Dataset): dataset for validation 

        loader (torchtrainer.dataloader.TrainerLoader): TrainerLoader which contains all the info
                                                        about data loading (both train and validation)

        epoch_num (int): number of epochs for training

        seed (int): seed used for torch and random. default=0

        device: device on which perform training, eg: cpu, cuda:0, cuda:1, ...
                If set to None and gpus are detected on the machine, the one with the lower memory usage
                will be choosen (otherwise cpu is used). default=None

        distributed (bool): allow distributed training on multiple gpus, using the distributed module of pytorch 
                            on a single node. default=False

        verbose (bool): print info on console at every step. If False prints only at the end of each epoch. 
                        default=True
                 
        results_path (str): absolute path to the folder in which tensorboard logs, model, checkpoint, ecc... will
                            be stored. default=None

        tb_logs (bool): If true tensorboard will be used to store statistics at each epoch, and the final model 
                        at the end of the training. The path for the tb logs will be something like this: 
                           - results_path/model_name/dd_mm_aa_hh:mm:ss/

                        At the end of the training you will find inside all the info for tensorboard and the final
                        model state dict (model_name.pt). If tb_checkpoint_rate is greater than 0, also a folder with
                        checkpoints will be created.

                        default=False

        tb_checkpoint_rate (int): The number of epochs at which checkpoint are stored, eg: 2 -> every two epochs, 
                                  3 -> every three epochs, ... . default=0

        checkpoint_path (str): absolute path to the checkpoint file that you want to load before training. This can
                               be used to start the training from a previous computed checkpoint. 
                               We expect a dictionary with these keys:  

                               - 'model_state_dict': dict, required
                               - 'epoch': int, optional 
                               - 'optimizer_state_dict': dict, optional
                               - 'loss': train_loss, optional

                               default=None

        """
        # model info
        self.model = model
        self.model_name = model.name
        # data info
        self.train_dataset = train_dataset 
        self.valid_dataset = valid_dataset
        self.dataloader = loader
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
        self.print_stats = verbose 
        # tensorboard stuffs
        self.tb_logs = tb_logs
        self.save_path = results_path
        self.tb_embeddings_num = tb_embeddings_num
        self.tb_embeddings = self.tb_embeddings_num > 0
        self.load_path = checkpoint_path
        if tb_checkpoint_rate > 0:
            self.save_checkpoint = True
            self.checkpoint_rate = tb_checkpoint_rate
        else: 
            self.save_checkpoint = False

        if distributed and self.gpu_num < 2: 
            raise ValueError(f"cannot use distributed training, only {self.gpu_num} gpu are present")

        if distributed and (self.dataloader.train_sampler is not None or self.dataloader.valid_sampler is not None):
            raise ValueError(f"Cannot use custom sampler with distributed training")

        # set true only if distributed is enable and device is not set to any particular one
        # at this point we are also sure to have at least 2 gpus for training
        self._distributed = (distributed and (self.device is None))

        # tb checks
        if tb_logs and self.save_path is None: 
            raise ValueError("results_path should be defined if tb_logs is True")
        if self.tb_embeddings and not tb_logs: 
            raise ValueError("tb_logs should be True if tb_embeddings_num is greater than 0")

        # checkpoint checks
        if tb_checkpoint_rate > 0 and not tb_logs: 
            raise ValueError("tb_logs should be True if tb_checkpoint_rate is greater than 0")

        # set seeds
        torch.manual_seed(seed)
        random.seed(seed)


    ## -- utils --

    def _check_device(self, device):
        if device is None:
            return torch.device('cuda:'+str(helper.get_free_gpu()) if torch.cuda.is_available() else 'cpu')
        return device  

    def _send_to_device(self, data: dict, device):
        data.update({k: d.to(device) for k, d in data.items()})

    def _save_checkpoint(self, epoch, model, optimizer, scheduler, train_loss, last=False):
        name = self.model_name
        name += f'.pt' if last else f'_ckpt_epoch_{epoch}.pt'
        path = os.path.join(self.tb_logdir, 'checkpoints', name)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': train_loss}, path)

    ##  -- stats --

    def _save_epoch_stats(self, epoch, train_len, valid_len, train_stats, valid_stats, last_lr):
        for key, value in train_stats.items():
            self.tb_writer.add_scalar('Train/{}'.format(key), value / train_len, epoch)

        for key, value in valid_stats.items():
            self.tb_writer.add_scalar('Valid/{}'.format(key), value / valid_len, epoch)

        self.tb_writer.add_scalar(f'Learning rate', last_lr, epoch)

    def _print_epoch_stats(self, pbar, cur_step, cur_lr, train_len, valid_len, train_stats, valid_stats=None):
        out_dict = {k: v.item() / cur_step for k, v in train_stats.items()}
        out_dict['lr'] = cur_lr

        if valid_stats:
            out_dict.update({k: v.item() / valid_len for k, v in valid_stats.items()})

        end = valid_stats is not None
        helper.print_epoch_stats(pbar, end, **out_dict)

    ## -- tensorboard stuffs --

    def _tb_setup_tensorboard(self):
        self.tb_logdir = helper.create_logs_dir(self.save_path, self.model_name)    
        self.tb_writer = SummaryWriter(log_dir=self.tb_logdir)

    def _tb_save_graph(self, model, input_shape):
        self.tb_writer.add_graph(model, torch.rand(input_shape))
        self.tb_writer.flush()

    def _tb_save_results(self, train_len, valid_len, model, epoch, optimizer, scheduler, train_loss):
        self._save_checkpoint(epoch, model, optimizer, scheduler, train_loss, last=True)

        # write hparameters to tensorboard
        # this can be useful to compare different runs with different hparams
        hpar_dict = {'hparam/{}'.format(k): v / train_len for k, v in model.train_stats.items()}
        hpar_dict.update({'hparam/{}'.format(k): v / valid_len for k, v in model.valid_stats.items()})

        self.tb_writer.add_hparams(self.args, hpar_dict)
        self.tb_writer.flush()
        self.tb_writer.close()

    def _tb_save_embeddings(self, model, device, valid_loader):
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


    ## -- train --

    def train(self):
        if self._distributed:
            helper.print_summary(self.model, None, self._distributed, self.args, self._multi_train, self._conf_num) 
            dist.spawn_processes(self._train_loop, self.gpu_num)
        else:
            device = self._check_device(self.device)
            helper.print_summary(self.model, device, self._distributed, self.args, self._multi_train, self._conf_num) 
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

        if self._distributed:
            # setup of each process
            dist.setup(rank=gpu, world_size=world_size)
            model = dist.DDP(model, device_ids=[gpu])

            train_loader, train_sampler = self.dataloader._get_distributed_loader(self.train_dataset, 
                                                                   world_size, gpu, 
                                                                   device)
            valid_loader, valid_sampler = self.dataloader._get_distributed_loader(self.valid_dataset, 
                                                                   world_size, gpu,
                                                                   device)

        else: 
            train_loader = self.dataloader._get_loader(self.train_dataset, device, 'train')
            valid_loader = self.dataloader._get_loader(self.valid_dataset, device, 'valid')

        # num of train batches
        train_len = len(train_loader)
        # num of valid batches 
        valid_len = len(valid_loader)

        if self.tb_logs and master:
            self._tb_setup_tensorboard()
            # TODO: get the input shape and save the model graph
            #input_sample = next(iter(train_loader))
            #self._tb_save_graph(model, input_sample.shape)

        # load checkpoint
        epoch_start = 0
        if self.load_path:
            ckpt_dict = torch.load(self.load_path, map_location=device)

            model.load_state_dict(ckpt_dict['model_state_dict'])
            if 'optimizer_state_dict' in ckpt_dict:
                optimizer.load_state_dict(ckpt_dict['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt_dict:
                scheduler.load_state_dict(ckpt_dict['scheduler_state_dict'])
            if 'epoch' in ckpt_dict:
                epoch_start = ckpt_dict['epoch'] + 1

        # training loop
        for epoch in range(epoch_start, self.epoch_num): 
            model.reset_stats()
            model.train()

            # see https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
            if self._distributed:
                train_sampler.set_epoch(epoch)
                valid_sampler.set_epoch(epoch)

            # if verbose use tqdm to print status bar
            if self.print_stats and master:
                print(f"Epoch {epoch}/{self.epoch_num}")
                loader = tqdm(train_loader, bar_format='  {n_fmt}/{total_fmt} |{bar:20}| {elapsed} {postfix}',
                              unit_scale=True, leave=True)
            else: 
                loader = train_loader

            for i, data in enumerate(loader):
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
                    self._print_epoch_stats(loader, i+1, scheduler.get_last_lr()[0], train_len, valid_len, model.train_stats)

            # compute validation loss and acc at the end of each epoch
            model.eval()
            with torch.no_grad():
                for _, data in enumerate(valid_loader):
                    # send to device
                    self._send_to_device(data, device)

                    # valid step 
                    model.validation_step(data)

            if self.print_stats and master:
                self._print_epoch_stats(loader, i+1, scheduler.get_last_lr()[0], train_len, valid_len, model.train_stats, model.valid_stats)
                loader.close()

            #lr decay step
            if scheduler is not None: 
                scheduler.step()
            
            if master and self.tb_logs:
                self._save_epoch_stats(epoch, train_len=train_len, valid_len=valid_len,
                                       train_stats=model.train_stats, valid_stats=model.valid_stats, 
                                       last_lr=scheduler.get_last_lr()[0])

            # save checkpoint 
            if master and self.save_checkpoint and (epoch > 0) and not (epoch % self.checkpoint_rate):
                self._save_checkpoint(epoch, model, optimizer, scheduler, loss)

        # train end
        helper.print_end_train() 
    
        # tensorboard for saving results and embeddings
        if master and self.tb_logs:
            self._tb_save_results(model=model, train_len=train_len, valid_len=valid_len,
                                  epoch=epoch, optimizer=optimizer, scheduler=scheduler, train_loss=loss)

        if master and self.tb_embeddings:
            self._tb_save_embeddings(model, device, valid_loader)


    ## -- multi train --

    def multi_train(self, train_config_path):
        with open(train_config_path) as f:
            train_configs = json.load(f)

        #train_config_df = pd.read_csv(train_config_path, sep=' ')
        #train_configs = train_config_df.to_dict(orient='records')

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
