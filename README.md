# Torch trainer

Python library for training Pytorch models without having to care about devices, training loops and all the boilerplate code required for usually training a pytorch model. 

### Model

A model is defined by defining a class that inherits from torchtrainer.model.Model and implements the methods forward, training_step and validation_step. These last two methods contain the logic of a single training and validation step, the library is then resposible for creating the training loop and also send all the data to the correct device.

```
from torchtrainer.model import Model
import torch

class MyModel(Model):
    def __init__(self, name):
        super().__init__(name=name)
        # my init
        
    def forward(self, x):
        # my forward
        return x

    def training_step(self, x):
        # all the logic of a single training step
        # goes here. This method must return the value
        # of the loss function at the end of the training step
        
        input, target = x
        prediction = self(input)
        loss = torch.nn.functional.cross_entropy(prediction, target)
        return loss

    def validation_step(self, x):
        # all the logic of a validation step, if needed
        pass

    def define_optimizer_scheduler(self):
        # this method is used to define the optimizer
        # and the learning rate scheduler (not mandatory) to be
        # used for training the model
        
        opt = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt, gamma=1.0)
        return (opt, scheduler)
```

### Dataloaders and training

Once the model is created, an instance of torchtrainer.trainer.Trainer and the dataloaders for the training and validation dataset are needed. TrainLoader is a wrapper of torch.utils.data.DataLoader, which can be used to define custom batchers and samplers. The Trainer instead requires the datasets (train and validation), the model and the loader previously defined. Moreover, there are additional options that can be used, for instance, to: save models checkpoints, save tensorboard data (epochs info and hyper-parameters), ... .

```    
from torchtrainer.trainer import Trainer
from torchtrainer.dataloader import TrainerLoader

def main():
  # the model previously defined
  model = MyModel('model')
  # train dataset and valid dataset
  train_dataset = ...
  valid_dataset = ...
  
  loader = TrainerLoader(batch_size=self.bs, num_workers=0, shuffle=False)
  trainer = Trainer(model=model, train_dataset=train_dataset, 
                    valid_dataset=valid_dataset, epoch_num=10, loader=loader)
  
  trainer.train()
```

## Build

Use conda build to build the library as a conda package:

```
git clone https://github.com/francescocastelli/torch-trainer
cd torch-trainer/conda-recipe
conda build . --output-folder /path/to/output_folder
```

and install it in the current conda environment:

```
conda install --use-local /path/to/output_folder/linux-64/torchtrainer-1.1-py38_1.tar.bz2
```

