import argparse

from Utils.params import str2bool



def puzzlemix_parser(parser, args):

    parser.add_argument('--in_batch', type=str2bool, default=False,
                        help='whether to use different lambdas in batch')

    parser.add_argument('--neigh_size', type=int, default=4,
                        help='neighbor size for computing distance beteeen image regions')
    parser.add_argument('--n_labels', type=int, default=3,
                        help='label space size')

    parser.add_argument('--puzzle_beta', type=float,
                        default=1.2, help='label smoothness')
    parser.add_argument('--puzzle_gamma', type=float,
                        default=0.5, help='data local smoothness')
    parser.add_argument('--puzzle_eta', type=float, default=0.2, help='prior term')

    parser.add_argument('--transport', type=str2bool,
                        default=True, help='whether to use transport')
    parser.add_argument('--t_eps', type=float, default=0.8,
                        help='transport cost coefficient')
    parser.add_argument('--t_size', type=int, default=-1,
                        help='transport resolution. -1 for using the same resolution with graphcut')

    parser.add_argument('--p_eta', type=float, default=0.2, help='prior term')
    parser.add_argument('--clean_lam', type=float,
                        default=1.0, help='clean input regularization')
    parser.add_argument('--mp', type=int, default=1,
                        help='multi-process for graphcut (CPU)')
    
    if parser.parse_known_args(args)[0].train_mode.lower() in ['puzzlemix_pixel']:
        parser.add_argument('--block_num', type=int, default=None,
                            help='2**block_num will be used for block size \times block size')
    if parser.parse_known_args(args)[0].train_mode.lower() in ['feature_puzzlemix']:
        parser.add_argument('--improve', type=str, default=None,
                            help='improve mode')
    
    return parser
