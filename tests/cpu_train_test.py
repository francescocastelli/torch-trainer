import unittest
import torch
from torchtrainer.model import Model
from torchtrainer.trainer import Trainer
from torchtrainer.dataloader import TrainerLoader
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim

class Net(Model):
    def __init__(self):
        super().__init__(name='test')
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.pool(torch.functional.F.relu(self.conv1(x)))
        x = self.pool(torch.functional.F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = torch.functional.F.relu(self.fc1(x))
        x = torch.functional.F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def training_step(self, data):
        inputs = data['inputs']
        labels = data['labels']

        # forward + backward + optimize
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        self.save_train_stats(train_loss=loss.item()*inputs.shape[0], test=0.0)

        return loss

    def validation_step(self, data):
        inputs = data['inputs']
        labels = data['labels']

        # forward + backward + optimize
        outputs = self(inputs)
        loss = self.criterion(outputs, labels)

        self.save_valid_stats(valid_loss=loss.item()*inputs.shape[0], v_test=0.0)

        return loss

    def define_optimizer_scheduler(self):
        opt = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=opt,
                                                           gamma=1.0, last_epoch=-1)
        return (opt, scheduler)

    def embeddings_forward(self):
        pass

class TrainDataset():
    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform)

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    def __len__(self):
        return len(self.trainset)

    def __getitem__(self, idx):
        inputs, labels = self.trainset[idx]
        return {'inputs': inputs, 'labels': labels}

class ValidDataset():
    def __init__(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

    def __len__(self):
        return len(self.testset)

    def __getitem__(self, idx):
        inputs, labels = self.testset[idx]
        return {'inputs': inputs, 'labels': labels}


class ModelSetUpTestCase(unittest.TestCase):
    def setUp(self):
        self.bs = 4
        self.epochs = 3
        self.model = Net()
        self.train_dataset = TrainDataset()
        self.valid_dataset = ValidDataset()

    def test_train_trainer(self):
        args = {}
        loader = TrainerLoader(batch_size=self.bs, num_workers=0, shuffle=False)

        trainer = Trainer(model=self.model, train_dataset=self.train_dataset, 
                          valid_dataset=self.valid_dataset, summary_args=args,
                          epoch_num=self.epochs, loader=loader, 
                          distributed=False, print_stats=True, tb_logs=False)

        self.assertFalse(trainer._distributed) 
        self.assertIsNone(trainer.device)

        # train first with the trainer and save the loss
        trainer.train()

        trainer_train_loss = self.model.train_stats['train_loss'].item() / len(self.train_dataset)

        # reset the model and train like in https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        self.model = Net()

        trainloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.bs,
                                          shuffle=False, num_workers=0)

        optimizer, _ = self.model.define_optimizer_scheduler()

        for epoch in range(self.epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs = data['inputs']
                labels = data['labels']

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item() 
                print(f'\r {i+1}\\{len(trainloader)} - loss: {running_loss / (i+1)}', end='')

        final_loss = running_loss / (len(trainloader))
        self.assertAlmostEqual(trainer_train_loss, final_loss, delta=0.1)



if __name__ == '__main__':
    unittest.main()

