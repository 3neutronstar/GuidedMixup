import argparse
from Utils.params import str2bool

def guidedmixup_parser(parser, args):
    parser.add_argument('--blur', type=int, default=7,
                        help='choose gaussian blur kernel size, if select 0, then no gaussian blur')
    parser.add_argument('--blur_sigma', type=float,
                        default=3, help='choose gaussian blur kernel size')
    parser.add_argument('--saliency_normalization', type=str,
                        default='sumto1', help='choose saliency normalization method (sumto1, minmax, zscore)')
    
    if parser.parse_known_args(args)[0].train_mode.lower() in ['guided-sr','guided-ap']:
        parser.add_argument('--improve', type=str, choices=['learnable'],
                            default=None, help='choose kernel size')
    
    if parser.parse_known_args(args)[0].mixup_type.lower()!='hidden':
        parser.add_argument('--blur_random', type=bool, default=False,
                            help='blur randomness or not')
        parser.add_argument('--blur_sigma_random', type=bool, default=False,
                            help='blur sigma randomness or not')
        parser.add_argument('--m_block_size',  default=[],
                                type=lambda s: [int(item) for item in s.split(',')],
                                help='block size (default: 0 and 0 for pixelwise, others is calculated as 2^n)')
        if len(parser.parse_known_args(args)[0].m_block_size)>=0:
            parser.add_argument('--interpolate_mode', type=str,
                                default='nearest', help='choose interpolation mode (nearest, bicubic, etc.)')
        
        for block_size in parser.parse_known_args(args)[0].m_block_size:
            if parser.parse_known_args(args)[0].train_mode in ['guided-sr','guided-ap'] and parser.parse_known_args(args)[0].improve=='learnable':
                pass
            else: 
                if block_size < 0: 
                    raise ValueError('block size must be greater than 0. 1 means pixel')
        parser.add_argument('--label_mixing', type=str,
                            default='saliency_label', help='choose saliency-based or half-based label assign')
    if parser.parse_known_args(args)[0].train_mode.lower() in ['guided-ap']:
        parser.add_argument('--clean_lam', type=float,
                        default=0, help='0 or 1 for guided-ap')
    if parser.parse_known_args(args)[0].train_mode.lower() in ['guided-sr']:
        parser.add_argument('--saliency_blur', type=int, default=7,
                            help='choose gaussian blur kernel size, if select 0, then no gaussian blur')
        parser.add_argument('--saliency_blur_sigma', type=float,
                            default=3, help='choose gaussian blur kernel size')
    
    return parser
