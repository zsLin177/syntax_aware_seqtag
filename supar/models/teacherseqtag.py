
import pdb, math
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

class TeacherSeqTagModel(Model):
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

    def forward(self, words, sens_lst=None, feats=None, if_layerdrop=False, p_layerdrop=0.5, if_selfattdrop=False, p_attdrop=0.5):
        # [batch_size, seq_len, n_hidden]
        x = self.encode(words, feats, if_layerdrop, p_layerdrop, if_selfattdrop, p_attdrop)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(self.repr_mlp(x))
        return score


    def multi_forward(self, words, sens_lst=None, feats=None, times=3, if_T=True, if_layerdrop=False, p_layerdrop=0.5, if_selfattdrop=False, p_attdrop=0.5):
        '''
        To produce the set of predicted distributions Q={q1, q2, q3, ...}
        '''
        q_set = []
        score_T = words.new_tensor([self.n_mlp], dtype=torch.float).sqrt()
        for i in range(times):
            tmp_feats = feats[:]
            # [batch_size, seq_len, n_hidden]
            x = self.encode(words, tmp_feats, if_layerdrop, p_layerdrop, if_selfattdrop, p_attdrop)
            # [batch_size, seq_len, n_labels]
            score = self.scorer(self.repr_mlp(x))
            if if_T:
                score /= score_T
            q_set.append(score.softmax(-1))
        # [batch_size, seq_len, times, n_labels]
        q = torch.stack(q_set).permute(1, 2, 0, 3)
        return q

    def skl_div(self, q, p=None):
        '''
        q: predicted distributions
            [batch_size, seq_len, times, n_labels]
        p: the "gold" label
            [batch_size, seq_len]
        '''
        batch_size, seq_len, times, _ = q.shape
        if p != None:
            # return the kl with "gold"
            # return : [batch_size, seq_len, times]
            # [batch_size, seq_len, n_labels, times]
            q = -q.permute(0, 1, 3, 2).log()
            index = p.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, times)
            # [batch_size, seq_len, times]
            kl = torch.gather(q, 2, index).squeeze(2)
            return self.sig_scale(kl)
        else:
            # return the kl between predicted
            # return : [batch_size, seq_len, times, times]
            # [times, n_labels, batch_size*seq_len]
            q = q.reshape(batch_size*seq_len, times, -1).permute(1, 2, 0)
            # [times, times, batch_size*seq_len]
            neg_hp = (q * q.log()).sum(1).unsqueeze(1).expand(-1, times, -1)
            # [times, times, n_labels, batch_size*seq_len]
            a = q.unsqueeze(1).expand(-1, times, -1, -1)
            b = q.unsqueeze(0).expand(times, -1, -1, -1)
            # [times, times, batch_size*seq_len]
            cro_h = (a*b.log()).sum(2)
            # [batch_size, seq_len, times, times]
            kl = (neg_hp-cro_h).permute(2, 0, 1).reshape(batch_size, seq_len, times, -1)
            return self.sig_scale(kl)


    def avg_metric(self, q, p=None, threshold=0.5):
        '''
        return : [batch_size, seq_len]
        '''
        skl = self.skl_div(q, p)
        if p != None:
            # skl:[batch_size, seq_len, times]
            res = skl.mean(-1)
            res_mask = res.gt(threshold)
            return res, res_mask
        else:
            # skl:[batch_size, seq_len, times, times]
            mask = skl.new_ones(skl.shape[-1], skl.shape[-1]).bool().triu(1)
            res = skl.permute(2, 3, 0, 1)[mask].mean(0)
            res_mask = res.gt(threshold)
            return res, res_mask
    
    def vote_metric(self, q, p=None, threshold=0.5, vote_rate=0.5):
        skl = self.skl_div(q, p)
        times = skl.shape[-1]
        bound_num = math.floor(times * vote_rate)
        if p != None:
            # skl:[batch_size, seq_len, times]
            return skl.gt(threshold).sum(-1).ge(bound_num)
        else:
            # skl:[batch_size, seq_len, times, times]
            mask = skl.new_ones(skl.shape[-1], skl.shape[-1]).bool().triu(1)
            return skl.permute(2, 3, 0, 1)[mask].gt(threshold).sum(0).ge(bound_num)

    def var_metric(self, q, p=None, varhold=0.1):
        skl = self.skl_div(q, p)
        if p != None:
            # skl:[batch_size, seq_len, times]
            res = skl.std(-1)
            res_mask = res.gt(varhold)
            return res, res_mask
        else:
            # skl:[batch_size, seq_len, times, times]
            mask = skl.new_ones(skl.shape[-1], skl.shape[-1]).bool().triu(1)
            res = skl.permute(2, 3, 0, 1)[mask].std(0)
            res_mask = res.gt(varhold)
            return res, res_mask
        
    
    def sig_scale(self, kl):
        '''
        scale the kl based on sigmoid:
            skl = 2 * (sigmoid(kl) - 0.5)
            0<skl<1
        '''
        return 2*(kl.sigmoid()-0.5)   

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

 
class CrfTeacherSeqTagModel(Model):
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



        