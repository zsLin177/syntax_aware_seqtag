# -*- coding: utf-8 -*-

import argparse

from supar import TeacherSeqTagParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='Gnn SRL.')
    parser.set_defaults(Parser=TeacherSeqTagParser)
    subparsers = parser.add_subparsers(title='Commands', dest='mode')
    # train
    subparser = subparsers.add_parser('train', help='Train a parser.')
    subparser.add_argument('--feat', '-f', choices=['tag', 'char', 'lemma', 'bert'], nargs='+', help='features to use')
    subparser.add_argument('--build', '-b', action='store_true', help='whether to build the model first')
    subparser.add_argument('--encoder', choices=['lstm', 'bert', 'transformer'], default='lstm', help='encoder to use')
    subparser.add_argument('--max-len', type=int, help='max length of the sentences')
    subparser.add_argument('--buckets', default=32, type=int, help='max num of buckets to use')
    subparser.add_argument('--train', default='data/sdp/DM/train.conllu', help='path to train file')
    subparser.add_argument('--dev', default='data/sdp/DM/dev.conllu', help='path to dev file')
    subparser.add_argument('--test', default='data/sdp/DM/test.conllu', help='path to test file')
    subparser.add_argument('--embed', default='data/glove.6B.100d.txt', help='path to pretrained embeddings')
    subparser.add_argument('--unk', default='unk', help='unk token in pretrained embeddings')
    subparser.add_argument('--n-embed', default=100, type=int, help='dimension of embeddings')
    subparser.add_argument('--bert', default='bert-base-cased', help='which bert model to use')
    subparser.add_argument('--lr_rate', default=1, type=int)
    subparser.add_argument('--if_openDrop_e', default=False, type=bool, help='whether to open the dropout during training')
    subparser.add_argument('--self_uncer', action='store_true', help='whether to use self uncer')
    subparser.add_argument('--aux', action='store_true', help='whether to have the aux model')
    subparser.add_argument('--aux_path', default='exp/transformer-pos-teacher-base/model', help='path to aux model')
    subparser.add_argument('--threshold', default=0.03, type=float, help='path to aux model')
       
    # subparser.add_argument('--policy_grad', action='store_true', help='whether to train with policy_grad')
    # subparser.add_argument('--times', default=3, type=int, help='sample times during training')
    # subparser.add_argument('--pg_start_epoch', default=20, type=int, help='without aux model, then after this epochs to use the reward computed by self and need grad')

    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    subparser.add_argument('--if_openDrop_e', default=False, type=bool, help='whether to open the dropout during evaluate')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--task', default='05', choices=['05', '09', '12'], help='which dataset')
    subparser.add_argument('--gold',
                           default='data/conll05-original-style/sc-wsj.final')
    subparser.add_argument('--if_openDrop_p', default=False, type=bool, help='whether to open the dropout during predict')

    # filter
    subparser = subparsers.add_parser('filter', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')  
    subparser.add_argument('--output_data_full', default="data-error/output-full.txt", help="output the full results of possible mislabeled data.")
    subparser.add_argument('--if_openDrop_p', default=False, type=bool, help='whether to open the dropout during predict')
    subparser.add_argument('--times', default=10, type=int, help="how many times to predict.")  
    subparser.add_argument('--if_T', default=False, type=bool, help='whether to open the dropout during predict')  
    # subparser.add_argument('--method', default='kl-avg-gold', help='evaluate method')   
    subparser.add_argument('--method', choices=['kl-avg-gold', 'kl-avg-wo-gold', "kl-vote-gold", 'kl-vote-without-gold', "var-w-gold", "var-wo-gold"], nargs='+', help='method to use')  
    # parse(parser)

    # analysis
    subparser = subparsers.add_parser('analysis', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=1, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')  
    subparser.add_argument('--output_data_full', default="data-error/output-full.txt", help="output the full results of possible mislabeled data.")
    subparser.add_argument('--if_openDrop_p', default=False, type=bool, help='whether to open the dropout during predict')
    subparser.add_argument('--times', default=10, type=int, help="how many times to predict.")  
    subparser.add_argument('--if_T', default=False, type=bool, help='whether to open the dropout during predict')   
    
    parse(parser)


if __name__ == "__main__":
    main()
