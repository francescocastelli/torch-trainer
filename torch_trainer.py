import torch
from typing import Callable, Any
import utility as utils

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

    # should return the value of the loss function
    training_step: Callable[..., Any] = _train_step_unimplemented
    # may return the value of the loss, not important
    validation_step: Callable[..., Any] = _valid_step_unimplemented
    # defines the forward pass for getting in output embeddings
    # should returns embeddings
    embeddings_forward: Callable[..., Any] = _embeddings_forward_unimplemented
    # should return a tuple (optimizer, scheduler)
    define_optimizer_scheduler: Callable[..., Any] = _define_opt_sched_unimplemented

class Trainer:
    def __init__(self, model, train_loader, valid_loader, epoch_num, summary_args, 
                 device=None, save_path=None, tb_embeddings=False):
        # model info
        self.model = model
        self.model_name = model.name
        # dataloders info
        self.train_loader = train_loader
        self.train_bs = train_loader.batch_size
        self.train_len = len(train_loader)
        self.valid_loader = valid_loader
        self.valid_bs = valid_loader.batch_size
        self.valid_len = len(valid_loader)
        # other info
        self.epoch_num = epoch_num
        self.device = self._check_device(device)
        self.model._device = self.device
        self.args = summary_args
        self.save_path = save_path
        self.tb_embeddings = tb_embeddings

    def _check_device(self, device):
        if device is None:
            return torch.device('cuda:'+str(utils.get_free_gpu()) if torch.cuda.is_available() else 'cpu')
        return device  

    def _send_to_device(self, data: dict):
        for k, d in data.items():
            data[k] = d.to(self.device)

        return data

    def _print_epoch_stats(self, current_epoch, current_i, end=False):
        out_dict = dict()
        for key, value in self.model.train_stats.items():
            out_dict[key] = value.item() / (current_i+1)

        for key, value in self.model.valid_stats.items():
            out_dict[key] = value.item() / (current_i+1) 
        
        utils.print_epoch_stats(current_epoch, current_i, self.train_len, 
                                self.scheduler.get_last_lr()[0], end, **out_dict)

    def _setup_tensorboard():
        if self.save_path is not None:
            save_path = utils.create_logs_dir(self.save_path)    
            self.tb_writer = SummaryWriter(log_dir=save_path)
        else: self.tb_writer = None

    def _save_epoch_stats(epoch):
        for key, value in self.model.train_stats.items():
            self.writer.add_scalar('Train/{}'.format(key), value, epoch)

        for key, value in self.model.valid_stats.items():
            self.writer.add_scalar('Valid/{}'.format(key), value, epoch)

    def _save_results():
        # save model
        torch.save(self.model.state_dict(), os.path.join(self.save_path, self.model_name))
        # write hparameters to tensorboard
        # this can be useful to compare different runs with different hparams
        hpar_dict = {}
        for key, value in self.model.train_stats.items():
            hpar_dict['hparam/{}'.format(key)] = value

        for key, value in self.model.valid_stats.items():
            hpar_dict['hparam/{}'.format(key)] = value

        self.writer.add_hparams(vars(args), hpar_dict)
        self.writer.flush()
        self.writer.close()

    def _save_embeddings():
        self.model.train = False
        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                # send to device
                data = self._send_to_device(data)

                # inference 
                embeddings, *metadata = self.model.embeddings_forward(data)

                if i == 0:
                    final_embeddings = embeddings
                    final_meta = metadata
                else:
                    final_embeddings = torch.vstack((final_embeddings, embeddings))
                    final_meta = torch.vstack((final_meta, metadata))

                if i > int(self.embeddings_num/self.valid_bs): break

        print("\nSaving {} tensors for projection...".format(i*self.valid_bs)) 
        # save the embeddings + the metadata
        for i, meta in enumerate(metadata):
            writer.add_embedding(final_embeddings, metadata=meta, global_step=i)

    def train(self):
        # pre training operations
        self.model.to(self.device)
        self.optimizer, self.scheduler = self.model.define_optimizer_scheduler()

        utils.print_summary(self.model, self.device, self.args) 
        self._setup_tensorboard()

        # training loop
        for epoch in range(self.epoch_num): 
            self.model.train_stats.clear()
            self.model.valid_stats.clear()

            self.model.train()
            for i, data in enumerate(self.train_loader):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # send to device
                data = self._send_to_device(data)

                # training step 
                loss = self.model.training_step(data)

                #backward + optimize
                loss.backward()
                self.optimizer.step()

                self._print_epoch_stats(epoch, i)

            #compute validation loss and acc at the end of each epoch
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.valid_loader):
                    # send to device
                    data = self._send_to_device(data)

                    # valid step 
                    loss = self.model.validation_step(data)

            #lr decay step
            if self.scheduler is not None: 
                self.scheduler.step()

            self._save_epoch_stats(epoch)
            self._print_epoch_stats(epoch, i, end=True)

        if self.tb_writer is not None:
            _self.save_results()

        if self.tb_embeddings:
            _self.save_embeddings()

        print('Finished training!')

