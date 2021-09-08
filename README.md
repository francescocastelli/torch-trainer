# Torch trainer

Library for training pytorch models without having to care about device, training loops and all the boilerplate code required for usually training a pytorch model. 
It consist of two simple classes: 
- Model
- Trainer

Just create a new children class of torch_trainer.Model and implement the methods training_step and validation_step. These two method contains only the logic of a single training and validation step, 
the library is then resposible of creating the training loop and also send all the data to the correct device. 

When the model is created, pass it to an instance of class torch_trainer.Trainer along with the dataloader for the train and dev datasets. The trainer has two methods: 
- train: single training loop 
- multi_train: read multiple configurations from a csv file and automatically execute the multiple training loop one after the other (each one is independet from the other)

The library is completly integrated with tensorboard. 
