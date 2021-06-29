import os 
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_trainer.utils import _trainer_helper as helper

class Trainer:
    def __init__(self, model, train_loader, valid_loader, epoch_num, summary_args, 
                 device=None, print_stats=True, tb_logs=False, tb_embeddings=False, save_path=None,
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
        self.print_stats = print_stats
        # tensorboard stuffs
        self.tb_embeddings = tb_embeddings
        self.tb_logs = tb_logs
        self.save_path = save_path
        self.tb_embeddings_num = tb_embeddings_num
        if tb_logs and save_path is None: 
            raise ValueError("save_path should be defined if tb_logs is True")
        if tb_embeddings and tb_embeddings_num is None: 
            raise ValueError("tb_embeddings_num should be defined if tb_embeddings is True")

    def _check_device(self, device):
        if device is None:
            return torch.device('cuda:'+str(helper.get_free_gpu()) if torch.cuda.is_available() else 'cpu')
        return device  

    def _send_to_device(self, data: dict):
        data.update({k: d.to(self.device) for k, d in data.items()})

    def _print_epoch_stats(self, current_epoch, current_i, end=False):
        out_dict = dict()
        for key, value in self.model.train_stats.items():
            out_dict[key] = value.item() / (current_i+1)

        if end: 
            for key, value in self.model.valid_stats.items():
                out_dict[key] = value.item() / (self.valid_len) 
        
        helper.print_epoch_stats(current_epoch, current_i+1, self.train_len, 
                                self.scheduler.get_last_lr()[0], end, **out_dict)

    def _setup_tensorboard(self):
        self.tb_logdir = helper.create_logs_dir(self.save_path)    
        self.tb_writer = SummaryWriter(log_dir=self.tb_logdir)

    def _save_epoch_stats(self, epoch):
        for key, value in self.model.train_stats.items():
            self.tb_writer.add_scalar('Train/{}'.format(key), value, epoch)

        for key, value in self.model.valid_stats.items():
            self.tb_writer.add_scalar('Valid/{}'.format(key), value, epoch)

    def _save_results(self):
        # save model
        torch.save(self.model.state_dict(), os.path.join(self.tb_logdir, self.model_name + '.pt'))
        # write hparameters to tensorboard
        # this can be useful to compare different runs with different hparams
        hpar_dict = {}
        for key, value in self.model.train_stats.items():
            hpar_dict['hparam/{}'.format(key)] = value

        for key, value in self.model.valid_stats.items():
            hpar_dict['hparam/{}'.format(key)] = value

        self.tb_writer.add_hparams(vars(self.args), hpar_dict)
        self.tb_writer.flush()
        self.tb_writer.close()

    def _save_embeddings(self):
        self.model.train = False
        with torch.no_grad():
            for i, data in enumerate(self.valid_loader):
                # send to device
                self._send_to_device(data)

                # inference 
                embeddings, *metadata = self.model.embeddings_forward(data)

                if i == 0:
                    final_embeddings = embeddings
                    final_meta = metadata
                else:
                    final_embeddings = torch.vstack((final_embeddings, embeddings))
                    final_meta = torch.vstack((final_meta, metadata))

                assert(embeddings.shape[0] == 128)
                if i > int(self.tb_embeddings_num/embeddings.shape[0]): break

        print("\nSaving {} tensors for projection...".format(i*embeddings.shape[0]))
        # save the embeddings + the metadata
        for i, meta in enumerate(metadata):
            self.tb_writer.add_embedding(final_embeddings, metadata=meta, global_step=i)

    def train(self):
        # pre training operations
        self.model.to(self.device)
        self.optimizer, self.scheduler = self.model.define_optimizer_scheduler()

        helper.print_summary(self.model, self.device, self.args) 

        if self.tb_logs:
            self._setup_tensorboard()

        # training loop
        for epoch in range(self.epoch_num): 
            self.model.reset_stats()
            self.model.train()

            for i, data in enumerate(self.train_loader):
                # send to device
                self._send_to_device(data)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # training step 
                loss = self.model.training_step(data)

                #backward + optimize
                loss.backward()
                self.optimizer.step()

                if self.print_stats:
                    self._print_epoch_stats(epoch, i)

            #compute validation loss and acc at the end of each epoch
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.valid_loader):
                    # send to device
                    self._send_to_device(data)

                    # valid step 
                    loss = self.model.validation_step(data)

            self._print_epoch_stats(epoch, self.train_len-1, end=True)

            #lr decay step
            if self.scheduler is not None: 
                self.scheduler.step()

            if self.tb_logs: 
                self._save_epoch_stats(epoch)

        print('Finished training!')

        # tensorboard for saving results and embeddings
        if self.tb_logs:
            self._save_results()

        if self.tb_embeddings:
            self._save_embeddings()

