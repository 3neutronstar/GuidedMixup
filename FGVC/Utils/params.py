import os
import json
import argparse

def load_params(configs: dict, file_path, file_name):
    ''' load parameters from file_name '''
    with open(os.path.join(file_path, 'training_data', file_name, 'configuration.json'), 'r') as fp:
        get_configs = json.load(fp)
    configs.update(get_configs)
    return configs


def save_params(configs,current_path, file_name):
    with open(os.path.join(current_path, 'training_data',file_name, 'configuration.json'), 'w') as fp:
        json.dump(configs, fp, indent=2)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
