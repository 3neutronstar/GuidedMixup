import torch
import argparse
import time
import os
import sys
import torch.nn as nn
from DataLoad.data_load import DatasetLoader

from Model.baseNet import get_model
from Utils.comixup_augmentation.comixup_parser import comixup_parser
from Utils.guidedmixup_augmentation.guidedmixup_parser import guidedmixup_parser
from Utils.puzzlemix_augmentation.puzzlemix_parser import puzzlemix_parser

from Utils.seed import fix_seed
from Utils.params import load_params, save_params, str2bool
import warnings


def parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="python main.py mode")

    parser.add_argument(
        'mode', type=str,
    )
    parser.add_argument(
        '--seed', type=int, default=1, help='fix random seed')
    parser.add_argument(
        '--deterministic', type=bool, default=True, help='CUDA deterministric.')
    parser.add_argument(
        '--model', type=str, default='resnet18', help='choose NeuralNetwork model')
    parser.add_argument(
        '--device', type=str, default='cuda', help='choose NeuralNetwork')
    parser.add_argument(
        '--batch_size', '-bsz', type=int, default=16, help='set mini-batch size')
    parser.add_argument(
        '--num_workers', type=int, default=4, help='number of process you have')
    parser.add_argument(
        '--weight_decay', type=float, default=5e-4, help='set optimizer\'s weight decay')
    parser.add_argument(
        '--lr', type=float, default=0.01, help='set learning rate')
    parser.add_argument(
        '--epochs', type=int, default=300, help='run epochs')
    parser.add_argument(
        '--dataset', type=str, default='cifar10', help='select dataset')
    parser.add_argument(
        '--no_verbose', type=str2bool, default=False, help='Show all the information of evaluation (ECE, NLL, FGSM, PGD, etc.')
    parser.add_argument(
        '--training_verbose', type=str2bool, default=False, help='training phase verbose for accuracy (loss is measured)')
    parser.add_argument('--warmup', type=int, default=0,
                        help='set lr scheduler warmup epochs')
    parser.add_argument('--cooldown', type=int, default=0,
                        help='set lr scheduler cooldown epochs')
    parser.add_argument('--gpu_ids', default='0',
                        type=str, help=' ex) 0,1,2')
    parser.add_argument('--detect_anomaly', default=False,
                        type=bool, help='Detect anomaly in PyTorch')
    parser.add_argument(
        '--dataset_path', help='dataset path (None: /data or .\\data\\dataset\\)', default=None)
    parser.add_argument(
        '--corruption_dataset_path', help='dataset path (None: /data or .\\data\\dataset\\)', default=None)
    #eval localization while training option
    parser.add_argument(
        '--eval_loc', help='evaluate localization while training', type=str2bool, default=False)
    if parser.parse_known_args(args)[0].eval_loc:
        parser.add_argument(
            '--cam_thres', help='cam threshold', type=float, default=0.2)
    
    #compile option
    parser.add_argument('--compile', help='Use torch.compile() in torch 2.0',
                        default=False, type=str2bool)
    parser.add_argument('--apex', help='Use apex',
                        default=False, type=str2bool)
    parser.add_argument('--use_fp16', help='Use mixed precision',
                        default=False, type=str2bool)
    
    # pretrained path
    if 'pretrained_' in parser.parse_known_args(args)[0].model:
        parser.add_argument(
            '--pretrained_path', type=str, default=None, help='Designate a pretrained path')
        
    #ddp option
    parser.add_argument(
        '--ddp', help='use distributed parallel', action='store_true')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    
    # limited labels per class option
    parser.add_argument('--labels_per_class', help='number of data per class in dataset (Now only support cifar10, cifar100) and this option can not use ddp option',
                        default=None, type=int)
    assert parser.parse_known_args(args)[0].labels_per_class is None or not parser.parse_known_args(args)[0].ddp, "labels_per_class option can not use ddp option"

    # wandb option
    parser.add_argument('--wandb', help='Use wandb',
                        default=True, type=str2bool)# wandb option
    if parser.parse_known_args(args)[0].wandb:
        parser.add_argument(
            '--wandb_gradient_tracking', type=str2bool, default=False, help='wandb gradient tracking')
    
    # criterion
    parser.add_argument(
        '--criterion', type=str, default='crossentropy', help='apply bce instead of cross entropy', choices=['crossentropy', 'bce', 'label_smoothing','onehot_ce'])
    if 'label_smoothing' == parser.parse_known_args(args)[0].criterion:
        parser.add_argument('--smoothing', default=0.0,
                            type=float, help='Label smoothing (default: 0.0)')
    
    # optimizer
    parser.add_argument(
        '--optimizer', type=str, default='sgd', help='apply optimizer', choices=['sgd', 'adamw'])
    if 'sgd' == parser.parse_known_args(args)[0].optimizer:
        parser.add_argument(
            '--nesterov', type=str2bool, default=True, help='set learning rate')
        parser.add_argument(
            '--momentum', type=float, default=0.9, help='set momentum')
    
    # lr scheduler
    parser.add_argument(
        '--lr_scheduler', type=str, default='step', choices=['step', 'cosine','cosine_warmup'], help='Optimizer lr scheduler selection(choices: step, cosine')
    if 'step' == parser.parse_known_args(args)[0].lr_scheduler:
        parser.add_argument(
            '--gamma', type=float, default=0.1, help='set lr decaying rate')
        parser.add_argument('--lr_steps', help='lr decaying epoch determination', default=[150, 225],
                            type=lambda s: [int(item) for item in s.split(',')])
    else:  # cosine
        parser.add_argument(
            '--lr_min', help='cosine annealing scheduler min_lr', default=1e-9, type=float)
    
    # mixup type
    parser.add_argument(
        '--mixup_type', type=str, default='plain', help='select mixup type (hidden or plain)', choices=['hidden', 'plain','hook_hidden'])
    
    # input level cutout option
    parser.add_argument(
        '--cutout', type=str2bool, default=False, help='apply cutout')
    if parser.parse_known_args(args)[0].cutout:
        parser.add_argument(
            '--cutout_length', type=int, default=16, help='cutout length')
        parser.add_argument(
            '--cutout_hole', type=int, default=1, help='cutout holes')
    parser.add_argument(
        '--augmix', type=str2bool, default=False, help='apply augmix')
    parser.add_argument(
        '--randaug', type=str2bool, default=False, help='apply randaugment')

    if parser.parse_known_args(args)[0].mode.lower() in ['train','finetune']:
        parser.add_argument(
            '--train_mode', '-t', type=str, default='baseline', choices=['baseline', 'train_verbose', 'custom_criterion',
                                                                         'cutmix', 'mixup', 'saliencymix',  'snapmix','comixup', 'puzzlemix', 'manifoldmixup', 'alignmixup',  'recursivemix', 'dropblock','dropfilter',
                                                                         'guided-sr', 'guided-ap','sage', 'catchupmix'],
            help='Choose Train Mode')
        train_mode = parser.parse_known_args(args)[0].train_mode.lower()
        if train_mode not in ['baseline', 'custom_criterion', 'isda']:
            parser.add_argument(
                '--mix_prob', type=float, default=1.0, help='how much mix the data')
            parser.add_argument(
                '--mix_prob_scheduling', type=bool, default=False, help='mixup probability scheduling (Only available in mixup_type "hidden".')

        if train_mode in ['cutmix', 'saliencymix', 'snapmix', 'puzzlemix', 'mixup', 'alignmixup', 'manifoldmixup','comixup']:
            parser.add_argument(
                '--alpha', type=float, default=1.0, help='determine lambda by alpha, mixup:0.2, cutmix,saliencymix,: 1.0, snapmix: 5.0, puzzlemix: 1.0, ')
            parser.add_argument(
                '--alpha_scheduling', type=str, default=None, help='determine lambda under schedule (starts from 0.5: manual , random: auto, static: None)',choices=['manual','auto','inv_manual'])

        ############# pairing ################
        parser.add_argument(
            '--condition', '-c', type=str, default='random', help='wrong data augmentation training start epoch',
            choices=['greedy', 'random', 'greedy_c'])
        if parser.parse_known_args(args)[0].condition == 'random' :
            m_part = None
        else:
            m_part = parser.parse_known_args(args)[0].batch_size
            parser.add_argument('--m_part', type=int,
                                    default=m_part, help='recommend 1/4 or 1/8 of your mini-batch size')
            parser.add_argument('--distance_metric', type=str,
                                default='l2', help='choose distance metric: jsd, cosine, l1 and l2 norm')
        if parser.parse_known_args(args)[0].condition in ['greedy_c']:
            try:
                import pairing as pairing
            except:
                raise NotImplementedError(
                    'Please install Implementor package using "python3 setup.py build_ext --inplace "')
        ####################################
            
        ## augmentation specific option ##
        # input level #
        if 'comixup' in train_mode:
            parser = comixup_parser(parser, args)
        if 'puzzlemix' in train_mode:
            parser = puzzlemix_parser(parser, args)
        if train_mode in ['guided-sr', 'guided-ap']:
            parser = guidedmixup_parser(parser, args)
        ####### Feature level Mixup ######
        if parser.parse_known_args(args)[0].mixup_type.lower() in ['hidden','hook_hidden']:
            parser.add_argument('--layer_mix', help='choose which hidden layer you want to mix', default=[0],
                                type=lambda s: [int(item) for item in s.split(',')])
            if 0 in parser.parse_known_args(args)[0].layer_mix:
                parser.add_argument('--input_mixup', type=str,
                                    default=None, choices=['cutmix', 'mixup','guided-sr','guided-ap','puzzlemix'])
            # hidden mixup option is on
            

        if parser.parse_known_args(args)[0].train_mode.lower() != 'baseline':
            parser.add_argument('--mixup_epochs', type=int,
                                default=0, help='Augmentation after few epochs')
        
    elif parser.parse_known_args(args)[0].mode.lower() == 'eval':
        parser.add_argument(
            '--file_name', type=str, default=None,
            help='Read file name')
        parser.add_argument('--evaluator', type=str, default='baseline')
        parser.add_argument('--eval_mode', type=str, default='baseline')

    if 'tiny-imagenet' == parser.parse_known_args(args)[0].dataset:
        parser.add_argument('--tiny_resize', action='store_true',
                            default=False, help='choose 224 size or 64')
    if parser.parse_known_args(args)[0].dataset in ['dogs120', 'cub200', 'cars196', 'caltech101', 'caltech256', 'flowers102', 'aircraft100', 'food101', 'caltech101', 'caltech256', 'inat']:
        parser.add_argument('--fgvc_resize', action='store_true',
                            default=False, help='choose 224 size or 448 (default 224)')

    parser.add_argument('--memo', type=str, default='')
    # return parser.parse_known_args(args)[0]
    return parser.parse_args(args)#[0]


def main(args):
    flags = parse_args(args)
    configs = vars(flags)
    ## gpu parallel ##
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = configs['gpu_ids']
    ##################
    use_cuda = torch.cuda.is_available()
    device = torch.device(
        "cuda" if use_cuda and configs['device'] == 'cuda' else "cpu")
    configs['device'] = str(device)
    if not configs['ddp']:
        # when not using ddp
        configs['local_rank'] = 0
    elif int(torch.__version__.split('.')[0])>=2:
        # pytorch 2.0 ddp with torchrun
        if int(os.environ["LOCAL_RANK"])==0:
            print("pytorch 2.x with torchrun, local_rank argument is ignored")
        print("local rank is set to: " + (os.environ["LOCAL_RANK"]))
        configs['local_rank'] = int(os.environ["LOCAL_RANK"])
    ## detect anomaly ##
    if configs['detect_anomaly']:
        torch.autograd.set_detect_anomaly(True)
    ####################
    ## seed ##
    fix_seed(seed=configs['seed'])
    ##########

    ## time data ##
    time_data = time.strftime(
        '%m-%d_%H-%M-%S', time.localtime(time.time()))
    ###############

    ## File name ##
    if configs['mode'] in ['train']:
        file_name = configs['dataset']+'_' + \
            configs['model']+'_'+configs['train_mode']
        if configs['mixup_type'] == 'hidden':
            file_name += '_hidden{}'.format(','.join(str(e)
                                            for e in configs['layer_mix']))
        if configs['condition'] !='random':
            file_name += '_p{}'.format(configs['condition'])
        file_name = file_name+'_'+time_data
    ###############

    ## data save path ##
    current_path = os.path.dirname(os.path.abspath(__file__))
    if configs['mode'] in ['train']:
        save_path = os.path.join(current_path, 'training_data')
        if not os.path.exists(save_path) and configs['local_rank'] == 0:
            os.mkdir(save_path)
        if not os.path.exists(os.path.join(save_path, file_name)) and configs['local_rank'] == 0:
            os.mkdir(os.path.join(save_path, file_name))
    else:
        save_path = os.path.join(current_path,'training_data')
    ####################

    ## save configuration ##
    parallel = configs['gpu_ids']
    if configs['mode'] in ['train']:
        configs.pop('gpu_ids')
        dataset_path = configs.pop('dataset_path')
        corruption_dataset_path = configs.pop('corruption_dataset_path')
        if configs['local_rank'] == 0:
            save_params(configs, current_path, file_name)
        configs['dataset_path'] = dataset_path
        configs['corruption_dataset_path'] = corruption_dataset_path
    elif configs['mode'] in ['finetune', 'eval']:
        if configs['file_name'] is None and 'pretrained' not in configs['model']:
            warnings.warn('Please provide file name')
        elif 'pretrained' not in configs['model']:
            file_name = configs['file_name']
            seed = configs['seed']
            configs.update(load_params(
                configs, current_path, configs['file_name']))
            configs['mode'] = 'eval'
            configs['file_name'] = file_name
            configs['device'] = device
            configs['seed'] = seed
            time_data = file_name
            configs['ddp'] = False
        else:
            file_name = configs['file_name']
        dataset_path = configs['dataset_path']

    configs['gpu_ids'] = parallel
    if configs['local_rank'] == 0:
        print("="*30)
        print(configs)
        print("="*30)
    ########################

    ## Num Classes ##
    if configs['dataset'] in ['cifar10', 'fashionmnist', 'mnist', 'stl10']:
        configs['num_classes'] = 10
    elif configs['dataset'] == 'speech-commands':
        configs['num_classes'] = 12
    elif configs['dataset'] in ['cifar100', 'aircraft100']:
        configs['num_classes'] = 100
    elif configs['dataset'] in ['flowers102']:
        configs['num_classes'] = 102
    elif configs['dataset'] in ['food101', 'caltech101']:
        configs['num_classes'] = 101
    elif configs['dataset'] == 'caltech256':
        configs['num_classes'] = 257
    elif configs['dataset'] == 'cars196':
        configs['num_classes'] = 196
    elif configs['dataset'] == 'dogs120':
        configs['num_classes'] = 120
    elif configs['dataset'] in ['tiny-imagenet', 'cub200', 'imagenet200']:
        configs['num_classes'] = 200
    elif configs['dataset'] == 'inat':
        configs['num_classes'] = 5089
    else:  # imagenet
        configs['num_classes'] = 1000
    if configs['local_rank'] == 0:
        print("Number of classes : {}".format(configs['num_classes']))
    #################

    ## Model ##
    model = get_model(configs)
    ###########

    if configs['mode'] in ['train','finetune']:
        ## wandb ##
        if configs['wandb'] and configs['local_rank'] == 0:
            wandb_configs=configs.copy()
            wandb_configs.pop('dataset_path')
            wandb_configs.pop('corruption_dataset_path')
            import wandb
            wandb.init(project="Resize_GuidedMixup", entity='3neutronstar',
                       group=configs['dataset']+configs['model'], name=configs['train_mode']+'_'+time_data, config=wandb_configs)
        ###########
        if configs['ddp']:
            configs['nprocs'] = torch.cuda.device_count()
            import torch.distributed as dist
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(configs['local_rank'])
            configs['batch_size'] = int(
                configs['batch_size'] / configs['nprocs'])
            configs['num_workers'] = int(
                (configs['num_workers'] + configs['nprocs'] - 1) / configs['nprocs'])

        ## dataloader ##
        dataset_loader = DatasetLoader(dataset_path, configs=configs)
        train_loader, valid_loader = dataset_loader.get_dataloader()
        ################
        mixup_hidden_list = [
            'manifoldmixup',
            'alignmixup',
            'train_verbose',
        ]
        if configs['train_mode'] in mixup_hidden_list or configs['train_mode'].startswith('feature'):
            from Implementor.mixuphidden import MixupHiddenLearner
            learner = MixupHiddenLearner(
                model, file_name, save_path, device, configs)
        else:
            from Implementor.guidedmixup import GuidedMixupLearner
            from Implementor.comixup import CoMixupLearner
            from Implementor.cutmix import CutMixLearner
            from Implementor.baseline import Baseline
            from Implementor.mixup import MixupLearner
            from Implementor.puzzlemix import PuzzleMixLearner
            from Implementor.saliencymix import SaliencyMixLearner
            from Implementor.snapmix import SnapMixLearner
            LEARNER = {
                'baseline': Baseline,
                'cutmix': CutMixLearner,
                'mixup': MixupLearner,
                'guided-sr': GuidedMixupLearner,
                'guided-ap': GuidedMixupLearner,
                'saliencymix': SaliencyMixLearner,
                'snapmix': SnapMixLearner,
                'comixup': CoMixupLearner,
                'puzzlemix': PuzzleMixLearner,
            }
            learner = LEARNER[configs['train_mode']](
                model, file_name, save_path, device, configs)
        if configs['mode'] in ['train']:
            learner.run(train_loader, valid_loader)
        else:
            raise NotImplementedError        
        if configs['wandb'] and configs['local_rank'] == 0:
            wandb.finish()
        exit()

if __name__ == '__main__':
    main(sys.argv[1:])
