import os 
import torch
import numpy as np
from datetime import datetime

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

def _print_model(model):
    width = os.get_terminal_size().columns

    out_m = '\n'
    out_m += '{:#^{}s}\n\n'.format('  Model: {}  '.format(model.name), width)
    print(out_m)
    print(model)

def print_summary(model, device, args, multi_train, config_num, eval_loop=False):
    width = os.get_terminal_size().columns

    # model 
    if not multi_train:
        _print_model(model)
    
    # parameters 
    conf = 'Configuration num: {}'.format(config_num)

    out = '\n\n'
    out += '{:#^{}s}\n\n'.format('  Device: {} - {} '.format(device, conf), width)
    for i, (key, value) in enumerate(vars(args).items()):
        out += '  {}={}  '.format(key, value) 
        if not (i+1) % 6:
            out += '\n'
    
    print('\r{}\n'.format(out))
    if not eval_loop:
        print('{:#^{}s}{}'.format('  Training loop  ', width, '\n'*4))

def print_overall_summary(model, device, train_config):
    width = os.get_terminal_size().columns

    # model 
    _print_model(model)

    # training configs
    out = '\n\n'
    out += '{:#^{}s}\n\n'.format('  Train configurations  ',  width)

    for i, conf in enumerate(train_config):
        out += ' Config {}: {}\n'.format(i, conf)

    print('\r{}\n'.format(out))

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

