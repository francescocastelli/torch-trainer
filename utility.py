import os 
import argparse
import numpy as np
from datetime import datetime
import parameters as param
import torch

def get_free_gpu():
    res = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    memory_available = [int(x.split()[2]) for x in res]
    return np.argmax(memory_available)

def create_logs_dir(dir_path):
    dir_name = datetime.now().strftime('%d-%m-%y_%H:%M:%S')
    dir_path = os.path.join(dir_path, dir_name)

    #create the directory 
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path 

def parse_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", type=int, help="number of workes for data loading", default=1)
    parser.add_argument("-d", type=str, help="device name, if not specified the one with lower ram usage will be choosen", default=None)
    #parameters
    parser.add_argument("--bs", type=int, help="batch size", default=param.bs)
    parser.add_argument("--lr", type=float, help="learning rate", default=param.lr)
    parser.add_argument("--l2", type=float, help="weight decay", default=param.l2)
    parser.add_argument("--gamma", type=float, help="gamma for exponential lr decay", default=param.gamma)
    parser.add_argument("--lambd", type=float, help="lambda parameter for the loss function", default=param.lambd)
    parser.add_argument("--epochs", type=int, help="total number of epochs", default=param.epoch_num)
    parser.add_argument("--margin", type=float, help="margin for triplet loss", default=param.margin)
    parser.add_argument("--projection", type=int, help="number of points for pca", default=param.projection_point_num)
    #model
    parser.add_argument("--Mpath", type=str, help="path to the model to load (its relative to the cwd)", default=None)
    parser.add_argument("--Mtype", help="model", choices=['base', 'two_heads', 'dropout', 'classification'])
    #loop
    parser.add_argument("-L", help="training loop", choices=['triplet_hard', 'triplet_all', 'triplet_reg', 'classification'])
    #dataset
    parser.add_argument("--Dtrain", help="train dataset type", default=param.dataset_type, choices=['normal', 'balanced', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06'])
    parser.add_argument("--Dvalid", help="valid dataset type", default=param.dataset_type, choices=['normal', 'balanced', 'A01', 'A02', 'A03', 'A04', 'A05', 'A06'])
    #boolean args
    parser.add_argument("--logs", help="wheter to save logs and model", action="store_true")
    parser.add_argument("--binary", help="binary or multiclass problem", action="store_true")
    parser.add_argument("--speakers", help="use speaker info in triplet selection", action="store_true")
    parser.add_argument("--systems", help="use systems info in triplet negative selection", action="store_true")
    parser.add_argument("--squareRoot", help="compute difference btw embeddings using the square root", action="store_true")

    return parser.parse_args()

def parse_args_eval():
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", type=int, help="number of workes for data loading", default=1)
    parser.add_argument("-d", type=str, help="device name, if not specified the one with lower ram usage will be choosen", default=None)
    parser.add_argument("--bs", type=int, help="batch size", default=param.bs)
    parser.add_argument("--Mpath", type=str, help="path to the model to load (its relative to the cwd)", default=None)
    parser.add_argument("--Mtype", help="model", default=param.model_type, choices=['base', 'two_heads', 'dropout', 'classification'])
    parser.add_argument("--logs", help="wheter to save logs and model", action="store_true")

    return parser.parse_args()

def print_summary(model, device, args, eval_loop=False):
    width = os.get_terminal_size().columns

    # model 
    out_m = '\n'
    out_m += '{:#^{}s}\n\n'.format('  Model: {}  '.format(args.Mtype), width)
    print(out_m)
    print(model)
    
    # parameters 
    out = '\n\n'
    out += '{:#^{}s}\n\n'.format('  Device: {}  '.format(device), width)
    for i, (key, value) in enumerate(vars(args).items()):
        out += '  {}={}  '.format(key, value) 
        if not (i+1) % 6:
            out += '\n'
    
    print('\r{}\n'.format(out))
    if not eval_loop:
        print('{:#^{}s}{}'.format('  Training loop  ', width, '\n'*4))

def print_eval_loop(current_i, data_size):
    width = os.get_terminal_size().columns
    print('\r{:#^{}s}'.format('  Evaluating: {}/{}  '.format(current_i, data_size), width), 
          end='')

def print_epoch_stats(epoch, current_i, data_size, lr, end=False, **kwargs): 
    out = '  Epoch {}:\n'.format(epoch)
    out += '{} lr={:3e}\n'.format(10*' ', lr)
    out += '{} {}/{}'.format(10*' ', current_i, data_size)
    line_num = 0

    for i, (key, value) in enumerate(kwargs.items()):
        out += ' - {}={:.4f}'.format(key, value) 
        if not (i+1) % 3: 
            out += '\n{}'.format(18*' ')
            line_num +=1
    
    print('\033[{}F{}'.format(2+line_num, out), end='')
    if end: print('\n'*3)

def print_eval_stats(**kwargs):
    width = os.get_terminal_size().columns
    out = '\n\n'
    for (key, value) in kwargs.items():
        out += '{:^{}s}\n'.format('{}={:.4f}'.format(key, value), width)

    print(out)

def save_epoch_stats(writer, epoch, name, **kwargs):
    for key, value in kwargs.items():
        writer.add_scalar('{}/{}'.format(name, key), value, epoch)

def save_projections(model, loader, writer, device, args):
    model.train = False
    system_dict = {0: 'bonafide', 1:'A01', 2:'A02',  3:'A03',  4:'A04', 5:'A05', 6:'A06'}
    with torch.no_grad():
        for i, data in enumerate(loader):
            inputs = data['data'].to(device)
            labels = data['target'].to(device)
            systems = data['system'].to(device)

            labels = torch.unsqueeze(labels,1)
            systems = torch.unsqueeze(systems,1)

            #forward
            if args.Mtype == 'two_heads':
                embeddings, linear_out = model(inputs)
            else:
                embeddings = model(inputs)

            if i == 0:
                final_embeddings = embeddings
                final_labels = labels
                final_systems = systems
            else:
                final_embeddings = torch.vstack((final_embeddings, embeddings))
                final_labels = torch.vstack((final_labels, labels))
                final_systems = torch.vstack((final_systems, systems))

            if i > int(args.projection/args.bs): break

    print("\nSaving {} tensors for projection...".format(i*args.bs)) 
    final_labels = ['spoof' if x == 1 else 'bonafide' for x in final_labels]
    final_systems = [system_dict[int(x[0].item())] for x in final_systems]
    writer.add_embedding(final_embeddings, metadata=final_labels, global_step=0)
    writer.add_embedding(final_embeddings, metadata=final_systems, global_step=1)
