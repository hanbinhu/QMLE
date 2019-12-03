
def get_argparser():
    import argparse
    parser = argparse.ArgumentParser(prog="qmle", description="Quantum Machine Learning Experiments")
    parser.add_argument('-v', '--version', action='version', version='%(prog)s 0.1')

    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument('--seed', metavar='<N>', type=int, default=0,
            help='Random seed (default: 0)')

    logger_parser = argparse.ArgumentParser(add_help=False)
    logger_parser.add_argument('--prefix', metavar='<DIR>', default='.',
            help='Prefix path for output files (default: \'.\')')
    logger_parser.add_argument('--log', metavar='<FILE>', default='output.log',
            help='Log file name (default: \'output.log\')')
    logger_parser.add_argument('--log-level', metavar='<LEVELNAME>', choices=['INFO', 'DEBUG'], default='INFO',
            help='Log file output level ({"INFO", "DEBUG"})')

    subparsers = parser.add_subparsers(title='commands', dest='command')

    # Parser for reproducing the paper
    parser_rep = subparsers.add_parser('reproduce', parents=[logger_parser, common_parser])
    parser_rep.add_argument('-p', '--data-path', metavar='<DIR>', required=True,
            help='Path to the dataset folder.')

    # Parser for run the experiments
    parser_run = subparsers.add_parser('run', parents=[logger_parser, common_parser])
    parser_run.add_argument('-d', '--dataset', metavar='<DATASET>',
            choices=['MNIST', 'Fashion-MNIST', 'CIFAR10'], required=True,
            help='Input datasets ({"MNIST", "Fashion-MNIST", "CIFAR10"}).')
    parser_run.add_argument('-p', '--data-path', metavar='<DIR>', required=True,
            help='Path to the dataset folder.')

    return parser

def config_logger(args):
    import os
    import logging
    logger = logging.getLogger('qmle')
    isPrefixExist = True
    if not os.path.exists(args.prefix):
        isPrefixExist = False
        os.makedirs(args.prefix)
    log_file_name = os.path.join(args.prefix, args.log)

    fh = logging.FileHandler(log_file_name, mode='w')
    ch = logging.StreamHandler()
    fh.setLevel(logging.INFO if args.log_level == 'INFO' else logging.DEBUG)
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt='[%(asctime)s][%(levelname)7s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)

    if isPrefixExist:
        logger.warning(f"Prefix path '{args.prefix}' already exists.")

def set_seed(seed):
    import random
    import numpy as np
    import torch
    import logging
    logger = logging.getLogger('qmle')
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    logger.info(f'Seed: {seed}')

def get_args():
    parser = get_argparser()
    args = parser.parse_args()
    if args.command is not None:
        config_logger(args)
        set_seed(args.seed)
    else:
        print('Warning: Please specify the action ({"reproduce", "run"}).')
    return args
