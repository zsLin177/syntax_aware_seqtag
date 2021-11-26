# -*- coding: utf-8 -*-

import argparse

from supar import CrfSeqTagParser
from supar.cmds.cmd import parse


def main():
    parser = argparse.ArgumentParser(description='xxx.')
    parser.set_defaults(Parser=CrfSeqTagParser)
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
    subparser.add_argument('--inference', default='mfvi', choices=['mfvi', 'lbp'], help='approximate inference methods')
    subparser.add_argument('--lr_rate', default=1, type=int)
    subparser.add_argument('--split',
                           action='store_true',
                           help='whether to use different mlp for predicate and arg')
    subparser.add_argument('--use_syntax',
                           action='store_true',
                           help='whether to use syntax to help seqtag')
    subparser.add_argument('--mix',
                           action='store_true',
                           help='whether to use mixed syntax info to help seqtag')
    subparser.add_argument('--synatax_path', default='JointParser/parser/save/joint-ctb7/ctb7.joint.bigram/', help='path of used syntax model')
    # evaluate
    subparser = subparsers.add_parser('evaluate', help='Evaluate the specified parser and dataset.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    # predict
    subparser = subparsers.add_parser('predict', help='Use a trained parser to make predictions.')
    subparser.add_argument('--buckets', default=8, type=int, help='max num of buckets to use')
    subparser.add_argument('--data', default='data/sdp/DM/test.conllu', help='path to dataset')
    subparser.add_argument('--pred', default='pred.conllu', help='path to predicted result')
    subparser.add_argument('--prob', action='store_true', help='whether to output probs')
    subparser.add_argument('--task', default='05', choices=['05', '09', '12'], help='which dataset')
    subparser.add_argument('--gold',
                           default='data/conll05-original-style/sc-wsj.final')
    subparser.add_argument('--vtb',
                           action='store_true',
                           default=False,
                           help='whether to use viterbi')
    parse(parser)


if __name__ == "__main__":
    main()
