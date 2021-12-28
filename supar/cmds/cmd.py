# -*- coding: utf-8 -*-

import pdb
import torch
from supar.utils import Config
from supar.utils.logging import init_logger, logger
from supar.utils.parallel import init_device


def parse(parser):
    parser.add_argument('--conf', '-c', help='path to config file')
    parser.add_argument('--path', '-p', help='path to model file')
    parser.add_argument('--device',
                        '-d',
                        default='-1',
                        help='ID of GPU to use')
    parser.add_argument('--seed',
                        '-s',
                        default=1,
                        type=int,
                        help='seed for generating random numbers')
    parser.add_argument('--threads',
                        '-t',
                        default=16,
                        type=int,
                        help='max num of threads')
    parser.add_argument('--batch-size',
                        default=5000,
                        type=int,
                        help='batch size')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help='node rank for distributed training')
    parser.add_argument("--itp",
                        type=float,
                        default=0.06)
    parser.add_argument('--update_steps',
                        default=1,
                        type=int)             
    parser.add_argument('--transformer_layers',
                        default=6,
                        type=int) 
    parser.add_argument('--p_layerdrop', '-p_l',  
                        default=0.5, 
                        type=float, 
                        help='hyper-parameters of the layerdropout')
    parser.add_argument("--dropout",
                        default=0.3,
                        type=float,
                        help="normal dropout")    
    # parser.add_argument('--methods',
    #                     choices=['layerdropout', 'dropout', 'both',"nodropout"], 
    #                     default='nodropout', help='which methods to use') 
    parser.add_argument("--if_layerdrop",
                        default=False,
                        type=bool,
                        help="whether to use LayerDrop")   
    parser.add_argument("--if_selfattdrop",
                        default=False,
                        type=bool,
                        help="whether to use Attention Dropout")    
    parser.add_argument("--p_attdrop",
                        default=0.5,
                        type=float,
                        help="dropout for self-attention")    
    # args, unknown = parser.parse_known_args()
    # args, unknown = parser.parse_known_args(unknown, args)
    # args = Config.load(**vars(args), unknown=unknown)
    # Parser = args.pop('Parser')

    args, unknown = parser.parse_known_args()
    args, _ = parser.parse_known_args(unknown, args)
    args = Config(**vars(args))
    Parser = args.pop('Parser')

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    init_device(args.device, args.local_rank)
    init_logger(logger, f"{args.path}.{args.mode}.log")
    logger.info('\n' + str(args))

    if args.mode == 'train':
        parser = Parser.build(**args)
        parser.train(**args)
    elif args.mode == 'evaluate':
        parser = Parser.load(args.path)
        parser.evaluate(**args)
    elif args.mode == 'predict':
        parser = Parser.load(args.path)
        parser.predict(**args)
    elif args.mode == 'filter':
        parser = Parser.load(args.path)
        parser.filter(**args)
