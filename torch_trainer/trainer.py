import os 
import torch
import copy
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch_trainer.utils import _trainer_helper as helper
from torch.profiler import profile, record_function, ProfilerActivity, schedule

class Trainer:
    def __init__(self, model, train_loader, valid_loader, epoch_num, summary_args: dict, 
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
        self._multi_train = False
        self._conf_num = 0 
        # inizialize stats on the model 
        self.model._device = self.device
        self.model.init_stats()
        # args 
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
        if self.model.load_path is not None:
            self.model.load_shared_dict(torch.load(os.path.join(os.getcwd(), self.model.load_path),
                                                   map_location=device))


    def _check_device(self, device):
        if device is None:
            return torch.device('cuda:'+str(helper.get_free_gpu()) if torch.cuda.is_available() else 'cpu')
        return device  

    def _send_to_device(self, data: dict):
        data.update({k: d.to(self.device) for k, d in data.items()})

    def _print_epoch_stats(self, current_epoch, current_i, end=False):
        out_dict = dict()

        out_dict = {k: v.item() / current_i for k, v in self.model.train_stats.items()}

        #for key, value in self.model.train_stats.items():
        #    out_dict[key] = value.item() / (current_i)

        if end: 
            for key, value in self.model.valid_stats.items():
                out_dict[key] = value.item() / (self.valid_len) 
        
        helper.print_epoch_stats(current_epoch, current_i, self.train_len, 
                                self.scheduler.get_last_lr()[0], end, **out_dict)

    def _setup_tensorboard(self):
        self.tb_logdir = helper.create_logs_dir(self.save_path)    
        self.tb_writer = SummaryWriter(log_dir=self.tb_logdir)

    def _save_epoch_stats(self, epoch):
        for key, value in self.model.train_stats.items():
            self.tb_writer.add_scalar('Train/{}'.format(key), value / self.train_len, epoch)

        for key, value in self.model.valid_stats.items():
            self.tb_writer.add_scalar('Valid/{}'.format(key), value / self.valid_len, epoch)

    def _save_results(self):
        # save model
        torch.save(self.model.state_dict(), os.path.join(self.tb_logdir, self.model_name + '.pt'))
        # write hparameters to tensorboard
        # this can be useful to compare different runs with different hparams
        hpar_dict = {}
        for key, value in self.model.train_stats.items():
            hpar_dict['hparam/{}'.format(key)] = value / self.train_len

        for key, value in self.model.valid_stats.items():
            hpar_dict['hparam/{}'.format(key)] = value / self.valid_len

        self.tb_writer.add_hparams(self.args, hpar_dict)
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
                    final_meta = [torch.hstack((f_m, m)) for f_m, m in zip(final_meta, metadata)]

                if i > int(self.tb_embeddings_num/embeddings.shape[0]): break

        print("\nSaving {} tensors for projection...".format(i*embeddings.shape[0]))
        # save the embeddings + the metadata
        for i, meta in enumerate(final_meta):
            self.tb_writer.add_embedding(final_embeddings, metadata=meta, global_step=i)

    def train(self):
        # pre training operations
        self.model.to(self.device)
        self.optimizer, self.scheduler = self.model.define_optimizer_scheduler()

        helper.print_summary(self.model, self.device, self.args, self._multi_train, self._conf_num) 

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
                    self._print_epoch_stats(epoch, i+1)

            #compute validation loss and acc at the end of each epoch
            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                for i, data in enumerate(self.valid_loader):
                    # send to device
                    self._send_to_device(data)

                    # valid step 
                    self.model.validation_step(data)

            self._print_epoch_stats(epoch, self.train_len, end=True)

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

    def multi_train(self, train_config_path):
        train_config_df = pd.read_csv(train_config_path, sep=' ')
        train_configs = train_config_df.to_dict(orient='records')

        # save initial model parameters 
        self._initial_model_param = copy.deepcopy(self.model.state_dict())
        self._multi_train = True
        self._output_config_path = os.path.splitext(train_config_path)[0] + '_results.csv'
        
        helper.print_overall_summary(self.model, self.device, train_configs) 
        for i, configs in enumerate(train_configs):
            # set model attributes for the current config
            self.args.update(configs)
            for k, d in configs.items(): 
                setattr(self.model, k, d)

            # train on the current config
            self._conf_num = i
            self.train()

            # save results
            configs.update({k: d.item()/self.train_len for k, d in self.model.train_stats.items()})
            configs.update({k: d.item()/self.valid_len for k, d in self.model.valid_stats.items()})

            # reset the parameters of the model
            self.model.reset_stats()
            self.model.load_state_dict(self._initial_model_param)
            torch.cuda.empty_cache()

        final_df = pd.DataFrame.from_dict(train_configs, orient='columns')
        final_df.to_csv(self._output_config_path, sep=' ')
