import os 
import numpy as np
from tqdm import tqdm
from datetime import datetime

def get_free_gpu():
    res = os.popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free')
    memory_available = [int(x.split()[2]) for x in res]
    return np.argmax(memory_available)

def create_logs_dir(dir_path, model_name):
    dir_name = datetime.now().strftime('%d-%m-%y_%H:%M:%S')
    # expand tilde in case is present
    path_to_folder = os.path.expanduser(dir_path)
    dir_path = os.path.join(path_to_folder, model_name, dir_name)

    #create the directory 
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        os.mkdir(os.path.join(dir_path, 'checkpoints'))

    return dir_path 

def _print_model(model):
    width = os.get_terminal_size().columns

    out_m = '\n'
    out_m += '{:#^{}s}\n\n'.format('  Model: {}  '.format(model.name), width)
    print(out_m)
    print(model)

def print_summary(model, device, distributed, args, multi_train, config_num, eval_loop=False):
    width = os.get_terminal_size().columns

    # model 
    if not multi_train:
        _print_model(model)
    
    # parameters 
    conf = 'Configuration num: {}'.format(config_num)

    out = '\n\n'
    if not distributed:
        out += '{:#^{}s}\n\n'.format('  Device: {} - {} '.format(device, conf), width)
    else:
        out += '{:#^{}s}\n\n'.format('  Train on multi GPU - {} '.format(conf), width)

    for i, (key, value) in enumerate(args.items()):
        out += '  {}={}  '.format(key, value) 
        if not (i+1) % 6:
            out += '\n'
    
    print('\r{}\n'.format(out))
    if not eval_loop:
        print('{:#^{}s}{}'.format('  Training loop  ', width, '\n'*4))

def print_overall_summary(model, train_config):
    width = os.get_terminal_size().columns

    # model 
    _print_model(model)

    # training configs
    out = '\n\n'
    out += '{:#^{}s}\n\n'.format('  Train configurations  ',  width)

    for i, conf in enumerate(train_config):
        out += ' {}: {}\n'.format(i, conf)

    print('\r{}\n'.format(out))

def print_epoch_stats(pbar, end, **kwargs): 
    msg = f"lr:{kwargs.pop('lr'):.2e}"
    for key, value in kwargs.items():
        msg += f" - {key}:{value:.4f}"

    pbar.set_postfix_str(msg)

    if end:
       pbar.unpause()
       bar = tqdm.format_meter(**pbar.format_dict)
       pbar.display(bar, pos=-1)
    

def print_end_train():
    width = os.get_terminal_size().columns
    out = '{:#^{}s}\n\n'.format('  Finished training!  ', width)
    print('\33[32m' + out + '\033[0m')
