import argparse

from Utils.params import str2bool

def comixup_parser(parser,args):

    # Co-Mixup
    parser.add_argument('--comix',
                        type=str2bool,
                        default=True,
                        help='true for Co-Mixup')
    parser.add_argument('--m_block_num',
                        type=int,
                        default=4,
                        help='resolution of labeling, -1 for random')
    parser.add_argument('--m_part', type=int, default=20, help='partition size')
    parser.add_argument('--m_beta',
                        type=float,
                        default=0.32,
                        help='label smoothness coef, 0.16~1.0')
    parser.add_argument('--m_gamma',
                        type=float,
                        default=1.0,
                        help='supermodular diversity coef')
    parser.add_argument('--m_thres',
                        type=float,
                        default=0.83,
                        help='threshold for over-penalization, tau, 0.81~0.86')
    parser.add_argument('--m_thres_type',
                        type=str,
                        default='hard',
                        choices=['soft', 'hard'],
                        help='thresholding type')
    parser.add_argument('--m_eta', type=float, default=0.05, help='prior coef')
    # parser.add_argument('--mixup_alpha',
    #                     type=float,
    #                     default=2.0,
    #                     help='alpha parameter for dirichlet prior')
    parser.add_argument('--m_omega',
                        type=float,
                        default=0.001,
                        help='input compatibility coef, \omega')
    parser.add_argument('--set_resolve',
                        type=str2bool,
                        default=True,
                        help='post-processing for resolving the same outputs')
    parser.add_argument('--m_niter',
                        type=int,
                        default=4,
                        help='number of outer iteration')
    parser.add_argument('--clean_lam',
                        type=float,
                        default=1.0,
                        help='clean input regularization')
    parser.add_argument("--mixup_parallel",
                        type=str2bool,
                        default=False,
                        help="mixup_process parallelization")
    return parser
