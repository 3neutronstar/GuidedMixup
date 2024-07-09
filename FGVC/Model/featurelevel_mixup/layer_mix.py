import random
import torch
from torch.distributions.dirichlet import Dirichlet as Dirichlet

def determine_layer_mix(train_mode: str,layer_mix: list, configs = None):
    if 0 in layer_mix and configs['input_mix_prob']:
        rest_prob=configs['mix_prob']-configs['input_mix_prob']
        if rest_prob<0:
            raise ValueError('input_mix_prob should be smaller than mix_prob')
        hidden_prob=[1./(len(layer_mix)-1) for i in range(len(layer_mix))]
        hidden_prob[0]=rest_prob
        m= Dirichlet(torch.tensor(hidden_prob))
        return_layer_mix= m.sample().argmax().item()
    else:
        if train_mode == 'manifoldmixup':
            layer_mix_index = random.randint(0, len(layer_mix)-1)
            return_layer_mix= layer_mix[layer_mix_index]
        elif train_mode == 'alignmixup':
            layer_mix_index = random.randint(0, len(layer_mix)-1)
            return_layer_mix= layer_mix[layer_mix_index]
        else:
            if len(layer_mix)==1:
                return_layer_mix = layer_mix[0]
            else: # list
                layer_mix_index = random.randint(0, len(layer_mix)-1)
                return_layer_mix= layer_mix[layer_mix_index]
    return return_layer_mix
