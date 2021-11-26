
import pdb
from typing import no_type_check_decorator
from torch import autograd
from torch._C import dtype
import torch.nn as nn
from supar.models.model import Model
from supar.modules import MLP
from supar.utils import Config
from supar.utils.common import MIN
from supar.structs import CRFLinearChain
from supar.modules.scalar_mix import ScalarMix
import torch
from torch.nn.utils.rnn import pad_sequence
# from parser.JParser import JParser

class SimpleSeqTagModel(Model):
    r"""
    TODO: introduction
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 syntax=False,
                 synatax_path='',
                 n_syntax=800,
                 mix=False,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp=400,
                 n_mlp_argument=300,
                 n_mlp_relation=300,
                 repr_mlp_dropout=.2,
                 scorer_mlp_dropout=.2,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 split=True,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_labels = n_labels
        self.n_mlp = n_mlp
        self.syntax = syntax
        self.n_synatax = n_syntax
        self.mix = mix
        
        self.repr_mlp = MLP(n_in=self.args.n_hidden, n_out=n_mlp, dropout=repr_mlp_dropout)
        self.scorer = MLP(n_in=n_mlp, n_out=n_labels, activation=False)
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, words, sens_lst=None, feats=None):
        # [batch_size, seq_len, n_hidden]
        x = self.encode(words, feats)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(self.repr_mlp(x))
        return score

    def decode(self, score):
        """
        TODO:introduce
        """
        return score.argmax(-1)

    def loss(self, score, gold_labels, mask):
        """
        TODO:introduce
        """
        return self.ce_criterion(score[mask], gold_labels[mask[:, 1:]])


class CrfSeqTagModel(Model):
    r"""
    TODO: introduction
    """
    def __init__(self,
                 n_words,
                 n_labels,
                 syntax=False,
                 synatax_path='',
                 n_syntax=800,
                 mix=False,
                 n_tags=None,
                 n_chars=None,
                 n_lemmas=None,
                 feat=['tag', 'char', 'lemma'],
                 n_embed=100,
                 n_pretrained=125,
                 n_feat_embed=100,
                 n_char_embed=50,
                 n_char_hidden=400,
                 char_pad_index=0,
                 char_dropout=0.33,
                 bert=None,
                 n_bert_layers=4,
                 mix_dropout=.0,
                 bert_pooling='mean',
                 bert_pad_index=0,
                 freeze=True,
                 embed_dropout=.2,
                 n_lstm_hidden=300,
                 n_lstm_layers=3,
                 encoder_dropout=.33,
                 n_mlp=400,
                 n_mlp_argument=300,
                 n_mlp_relation=300,
                 repr_mlp_dropout=.2,
                 scorer_mlp_dropout=.2,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 split=True,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_labels = n_labels
        self.n_mlp = n_mlp
        self.syntax = syntax
        self.n_synatax = n_syntax
        self.mix = mix
        
        self.repr_mlp = MLP(n_in=self.args.n_hidden, n_out=n_mlp, dropout=repr_mlp_dropout)
        self.scorer = MLP(n_in=n_mlp, n_out=n_labels, activation=False)
        self.trans = nn.Parameter(torch.zeros(n_labels+1, n_labels+1))
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, words, sens_lst=None, feats=None):
        # batch_size, seq_len, _ = words.shape
        # [batch_size, seq_len, n_hidden]
        x = self.encode(words, feats)

        # [batch_size, seq_len, n_labels]
        score = self.scorer(self.repr_mlp(x))
        return score

    def decode(self, score, mask):
        """
        TODO:introduce
        """
        dist = CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans)
        return dist.argmax.argmax(-1)

    def loss(self, score, gold_labels, mask):
        """
        TODO:introduce
        """
        batch_size, seq_len = mask.shape
        loss = -CRFLinearChain(score[:, 1:], mask[:, 1:], self.trans).log_prob(gold_labels).sum() / seq_len
        return loss



        