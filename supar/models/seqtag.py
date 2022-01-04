
import pdb, math
from typing import no_type_check_decorator
from torch import autograd
from torch._C import dtype
import torch.nn as nn
from torch.nn.init import normal_
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
                 repr_mlp_dropout=.3,
                 scorer_mlp_dropout=.2,
                 interpolation=0.1,
                 pad_index=0,
                 unk_index=1,
                 self_uncertain=False,
                 sample_t=50,
                 **kwargs):
        super().__init__(**Config().update(locals()))
        self.n_labels = n_labels
        self.n_mlp = n_mlp
        self.syntax = syntax
        self.n_synatax = n_syntax
        self.mix = mix
        self.self_uncertain = self_uncertain
        self.sample_t = sample_t
        
        self.repr_mlp = MLP(n_in=self.args.n_hidden, n_out=n_mlp, dropout=repr_mlp_dropout)
        self.scorer = MLP(n_in=n_mlp, n_out=n_labels, activation=False)
        if not self_uncertain:
            self.ce_criterion = nn.CrossEntropyLoss()
        else:
            self.uncer_mlp = nn.Sequential(MLP(n_in=self.args.n_hidden, n_out=self.args.n_hidden//2, dropout=repr_mlp_dropout),
                                            MLP(n_in=self.args.n_hidden//2, n_out=n_mlp, dropout=repr_mlp_dropout))
            self.uncer_scorer = MLP(n_in=n_mlp, n_out=1, activation=False)

    def forward(self, words, sens_lst=None, feats=None):
        # [batch_size, seq_len, n_hidden]
        x = self.encode(words, feats)
        # [batch_size, seq_len, n_labels]
        score = self.scorer(self.repr_mlp(x))
        if not self.self_uncertain:
            return score, None
        else:
            # [batch_size, seq_len]
            uncer = self.uncer_scorer(self.uncer_mlp(x)).squeeze(-1)
            return score, uncer
    
    def su_metric(self, uncer, threshold=0.5):
        ssu = self.sig_scale(uncer.pow(2))
        ssu_mask = ssu.gt(threshold)
        return ssu, ssu_mask

    @torch.no_grad()
    def multi_forward(self, words, sens_lst=None, feats=None, times=3, if_T=True):
        '''
        To produce the set of predicted distributions Q={q1, q2, q3, ...}
        '''
        q_set = []
        score_T = words.new_tensor([self.n_mlp], dtype=torch.float).sqrt()
        for i in range(times):
            tmp_feats = feats[:]
            # [batch_size, seq_len, n_hidden]
            x = self.encode(words, tmp_feats)
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
            # first compute an average distribution
            # then return skl with this avg distribution
            # return : [batch_size, seq_len, times]
            # [batch_size, seq_len, n_labels]
            avg_dis = q.mean(2)
            # [batch_size, seq_len]
            neg_avg_e = (avg_dis * avg_dis.log()).sum(-1)
            expanded_avg = avg_dis.unsqueeze(2).expand(-1, -1, times, -1)
            # [batch_size, seq_len, times]
            cross_e = -(expanded_avg * q.log()).sum(-1)
            kl = neg_avg_e.unsqueeze(-1) + cross_e
            return self.sig_scale(kl)

    def avg_metric(self, q, p=None, threshold=0.5):
        '''
        return : [batch_size, seq_len]
        '''
        skl = self.skl_div(q, p)
        # skl:[batch_size, seq_len, times]
        res = skl.mean(-1)
        res_mask = res.gt(threshold)
        return res, res_mask
    
    def vote_metric(self, q, p=None, vote_low_rate=0.3, vote_up_rate=0.9):
        '''
        q: predicted distributions
            [batch_size, seq_len, times, n_labels]
        p: the "gold" label
            [batch_size, seq_len]
        '''
        times = q.shape[2]
        preds = q.argmax(-1)
        if p is not None:
            # [batch_size, seq_len]
            vote = times - p.unsqueeze(-1).expand(-1, -1, times).eq(preds).sum(-1)
            # if the 'gold' label gets votes more than vote_rate, then we think this may be a mistake
            mask = (vote/times).ge(vote_low_rate) & (vote/times).le(vote_up_rate)
            return vote, mask
        else:
            # [batch_size, seq_len]
            mode = torch.mode(preds, -1)[0]
            uncer = 1 - (mode.unsqueeze(-1).expand(-1, -1, times).eq(preds).sum(-1)/times)
            mask = uncer.ge(vote_low_rate)
            return uncer, mask



    def var_metric(self, q, p=None, varhold=0.1):
        skl = self.skl_div(q, p)
        # skl:[batch_size, seq_len, times]
        res = skl.std(-1)
        res_mask = res.gt(varhold)
        return res, res_mask
        
    def sig_scale(self, kl):
        '''
        scale the kl based on sigmoid:
            skl = 2 * (sigmoid(kl) - 0.5)
            0<skl<1
        '''
        return 2*(kl.sigmoid()-0.5)

    def decode(self, score, uncer=None):
        """
        TODO:introduce
        """
        if not self.self_uncertain:
            return score.argmax(-1)
        else:
            # [batch_size, seq_len, n_labels]
            sample = torch.empty_like(score).normal_(mean=0, std=1)
            f_score = score + uncer.unsqueeze(-1) * sample
            return f_score.argmax(-1)


    def loss(self, score, gold_labels, mask, uncer=None):
        """
        TODO:introduce
        """
        if not self.self_uncertain:
            return self.ce_criterion(score[mask], gold_labels[mask[:, 1:]])
        else:
            # [k, n_labels], [k]
            score, uncer = score[mask], uncer[mask]
            k, n_labels = score.shape[0], score.shape[1]
            # [k, t, n_labels]
            sample = torch.empty((k, self.sample_t, n_labels), device=score.device).normal_(mean=0, std=1)
            sample = uncer.unsqueeze(-1).expand(-1, self.sample_t).unsqueeze(-1) * sample
            t_f_score = score.unsqueeze(1).expand(-1, self.sample_t, -1) + sample
            # [t, k ,n_labels]
            t_f_score = t_f_score.permute(1, 0, 2)
            # [t, k]
            t_gold_score = torch.gather(t_f_score, 2, gold_labels[mask[:, 1:]].unsqueeze(0).expand(self.sample_t, -1).unsqueeze(-1)).squeeze(-1)
            # [t, k]
            t_logsumexp = torch.logsumexp(t_f_score, 2)
            t_p = (t_gold_score - t_logsumexp).exp()
            # [k]
            neg_log_avg_p = -t_p.mean(0).log()
            return neg_log_avg_p.mean()


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



        