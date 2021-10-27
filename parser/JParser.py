# -*- coding: utf-8 -*-
import os
import pdb

from parser.modules import MLP, BertEmbedding, Biaffine, BiLSTM
from parser.modules.dropout import IndependentDropout, SharedDropout
from parser.utils.alg import crf, cky_mask, cky_simple
from parser.utils.common import coarse_productions
from parser.utils.fn import build, compose, pad

import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)
from nltk.tree import Tree

class JParser(nn.Module):

    def __init__(self, args):
        super(JParser, self).__init__()

        self.args = args
        # self.fields = fields
        self.pretrained = False
        # the embedding layer
        self.char_embed = nn.Embedding(num_embeddings=args.n_chars,
                                       embedding_dim=args.n_embed)
        n_lstm_input = args.n_embed
        if args.feat == 'bert':
            self.feat_embed = BertEmbedding(model=args.bert_model,
                                            n_layers=args.n_bert_layers,
                                            n_out=args.n_feat_embed)
            n_lstm_input += args.n_feat_embed
        if self.args.feat in {'bigram', 'trigram'}:
            self.bigram_embed = nn.Embedding(num_embeddings=args.n_bigrams,
                                             embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed
        if self.args.feat == 'trigram':
            self.trigram_embed = nn.Embedding(num_embeddings=args.n_trigrams,
                                              embedding_dim=args.n_embed)
            n_lstm_input += args.n_embed

        self.embed_dropout = IndependentDropout(p=args.embed_dropout)

        # the lstm layer
        self.lstm = BiLSTM(input_size=n_lstm_input,
                           hidden_size=args.n_lstm_hidden,
                           num_layers=args.n_lstm_layers,
                           dropout=args.lstm_dropout)
        self.lstm_dropout = SharedDropout(p=args.lstm_dropout)

        # the MLP layers
        self.mlp_span_l = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_span_r = MLP(n_in=args.n_lstm_hidden*2,
                              n_out=args.n_mlp_span,
                              dropout=args.mlp_dropout)
        self.mlp_label_l = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)
        self.mlp_label_r = MLP(n_in=args.n_lstm_hidden*2,
                               n_out=args.n_mlp_label,
                               dropout=args.mlp_dropout)

        # the Biaffine layers
        self.span_attn = Biaffine(n_in=args.n_mlp_span,
                                  n_out=args.n_sublabels,
                                  bias_x=True,
                                  bias_y=True)

        self.label_attn = Biaffine(n_in=args.n_mlp_label,
                                   n_out=args.n_labels,
                                   bias_x=True,
                                   bias_y=True)

        # self.crf = TreeCRFLoss(n_labels=args.n_sublabels) 
        
        self.pad_index = args.pad_index
        self.unk_index = args.unk_index

    def load_pretrained(self, embed_dict=None):
        embed = embed_dict['embed'] if isinstance(
            embed_dict, dict) and 'embed' in embed_dict else None
        if embed is not None:
            self.pretrained = True
            self.char_pretrained = nn.Embedding.from_pretrained(embed)
            nn.init.zeros_(self.char_embed.weight)
            if self.args.feat == 'bigram':
                embed = embed_dict['bi_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(embed)
                nn.init.zeros_(self.bigram_embed.weight)
            elif self.args.feat == 'trigram':
                bi_embed = embed_dict['bi_embed']
                tri_embed = embed_dict['tri_embed']
                self.bi_pretrained = nn.Embedding.from_pretrained(bi_embed)
                self.tri_pretrained = nn.Embedding.from_pretrained(tri_embed)
                nn.init.zeros_(self.bigram_embed.weight)
                nn.init.zeros_(self.trigram_embed.weight)
        return self

    def forward(self, feed_dict):
        chars = feed_dict["chars"]
        batch_size, seq_len = chars.shape
        # get the mask and lengths of given batch
        mask = chars.ne(self.pad_index)
        lens = mask.sum(dim=1)
        ext_chars = chars
        # set the indices larger than num_embeddings to unk_index
        if self.pretrained:
            ext_mask = chars.ge(self.char_embed.num_embeddings)
            ext_chars = chars.masked_fill(ext_mask, self.unk_index)

        # get outputs from embedding layers
        char_embed = self.char_embed(ext_chars)
        if self.pretrained:
            char_embed += self.char_pretrained(chars)

        if self.args.feat == 'bert':
            feats = feed_dict["feats"]
            feat_embed = self.feat_embed(*feats)
            char_embed, feat_embed = self.embed_dropout(char_embed, feat_embed)
            embed = torch.cat((char_embed, feat_embed), dim=-1)
        elif self.args.feat == 'bigram':
            bigram = feed_dict["bigram"]
            ext_bigram = bigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
            char_embed, bigram_embed = self.embed_dropout(
                char_embed, bigram_embed)
            embed = torch.cat((char_embed, bigram_embed), dim=-1)
        elif self.args.feat == 'trigram':
            bigram = feed_dict["bigram"]
            trigram = feed_dict["trigram"]
            ext_bigram = bigram
            ext_trigram = trigram
            if self.pretrained:
                ext_mask = bigram.ge(self.bigram_embed.num_embeddings)
                ext_bigram = bigram.masked_fill(ext_mask, self.unk_index)
                ext_mask = trigram.ge(self.trigram_embed.num_embeddings)
                ext_trigram = trigram.masked_fill(ext_mask, self.unk_index)
            bigram_embed = self.bigram_embed(ext_bigram)
            trigram_embed = self.trigram_embed(ext_trigram)
            if self.pretrained:
                bigram_embed += self.bi_pretrained(bigram)
                trigram_embed += self.tri_pretrained(trigram)
            char_embed, bigram_embed, trigram_embed = self.embed_dropout(
                char_embed, bigram_embed, trigram_embed)
            embed = torch.cat(
                (char_embed, bigram_embed, trigram_embed), dim=-1)
        else:
            embed = self.embed_dropout(char_embed)[0]

        x = pack_padded_sequence(embed, lens.cpu(), True, False)
        # pdb.set_trace()
        x, _, each_layer_out = self.lstm(x)
        x, _ = pad_packed_sequence(x, True, total_length=seq_len)
        x = self.lstm_dropout(x)

        for i in range(len(each_layer_out)):
            each_layer_out[i] = pad_packed_sequence(each_layer_out[i], True, total_length=seq_len)[0]
        
        x_f, x_b = x.chunk(2, dim=-1)
        n_x = torch.cat((x_f[:, :-1], x_b[:, 1:]), -1)
        # apply MLPs to the BiLSTM output states
        span_l = self.mlp_span_l(n_x)
        span_r = self.mlp_span_r(n_x)
        label_l = self.mlp_label_l(n_x)
        label_r = self.mlp_label_r(n_x)

        # [batch_size, seq_len, seq_len, n_sublabels]
        s_span = self.span_attn(span_l, span_r).permute(0, 2, 3, 1)
        # [batch_size, seq_len, seq_len, n_labels]
        s_label = self.label_attn(label_l, label_r).permute(0, 2, 3, 1)

        return s_span, s_label, x, torch.stack(each_layer_out)
    @classmethod
    def load(cls, path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # load model
        model = cls.load_model(path)
        # load fields
        model.load_fields(path, model.args)
        # load mask matrix
        coarse_mask, unary_mask = model.CHART.get_coarse_mask(coarse_productions)
        model.transitions = coarse_mask.to(device)
        model.start_transitions = unary_mask.to(device)

        return model
    
    @classmethod
    def load_model(cls, path):

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_path = os.path.join(path, 'model')
        state = torch.load(model_path, map_location=device)
        model = cls(state['args'])

        model.load_pretrained(state['pretrained'])
        model.load_state_dict(state['state_dict'], False)
        model.to(device)

        return model

    def load_fields(self, path, args):
        field_path = os.path.join(path, 'fields')
        self.fields = torch.load(field_path)
        self.TREE = self.fields.TREE
        if args.feat == 'bert':
            self.CHAR, self.FEAT = self.fields.CHAR
        elif args.feat == 'bigram':
            self.CHAR, self.BIGRAM = self.fields.CHAR
        elif args.feat == 'trigram':
            self.CHAR, self.BIGRAM, self.TRIGRAM = self.fields.CHAR
        else:
            self.CHAR = self.fields.CHAR
        self.POS = self.fields.POS
        self.CHART = self.fields.CHART


    def save(self, path):
        state_dict, pretrained = self.state_dict(), None
        if self.pretrained:
            pretrained = {'embed': state_dict.pop('char_pretrained.weight')}
            if hasattr(self, 'bi_pretrained'):
                pretrained.update(
                    {'bi_embed': state_dict.pop('bi_pretrained.weight')})
            if hasattr(self, 'tri_pretrained'):
                pretrained.update(
                    {'tri_embed': state_dict.pop('tri_pretrained.weight')})
        state = {
            'args': self.args,
            'state_dict': state_dict,
            'pretrained': pretrained
        }
        torch.save(state, path)

    @torch.no_grad()
    def api(self, sentences, bert_embed=None):
        """
        Args:
            sentences (list): a list of sentences.
            bert_embed (Tensor): bert embeddings from a pretrained BERT model.

        Returns:
            preds (list(tuple)): [(word segment, parsing), ...]
        """

        # eval mode
        self.eval()

        # sequences to indices
        trees = self.get_trees(sentences)
        if self.args.feat == 'bert':
            chars, feats = self.get_indice(sentences)
            feed_dict = {"chars": chars, "feats": feats}
        elif self.args.feat == 'bigram':
            chars, bigram = self.get_indice(sentences)
            feed_dict = {"chars": chars, "bigram": bigram}

        # encoder
        batch_size, seq_len = chars.shape
        lens = chars.ne(self.args.pad_index).sum(1) - 1
        mask = lens.new_tensor(range(seq_len - 1)) < lens.view(-1, 1, 1)
        mask = mask & mask.new_ones(seq_len-1, seq_len-1).triu_(1)
        s_span, s_label, x, each_layer_out = self.forward(feed_dict)

        # # decoder
        # if self.args.marg:
        #     s_span = crf(s_span, self.transitions, self.start_transitions, mask, marg=True, mask_inside=self.args.mask_inside)
        # preds = self.decode(s_span, s_label, self.transitions, self.start_transitions, mask, mask_cky=True)
        # # char-level
        # preds = [build(tree,
        #                 [(i, j, self.CHART.vocab.itos[label])
        #                 for i, j, label in pred])
        #             for tree, pred in zip(trees, preds)]
        # # word-level
        # preds = [compose(tree) for tree in preds]
        # # split word segment and parsing
        # preds = [(tree.leaves(), tree) for tree in preds]

        # return preds
        return x, each_layer_out

    def get_trees(self, sentences):
        """
        Transform a charater sequence into a char-level flatten tree.
        """

        trees = []
        for sentence in sentences:
            tree = Tree('TOP', [Tree('CHAR', [char]) for char in sentence])
            trees.append(tree)

        return trees

    def get_indice(self, sentences):
        """
        Transform characters into indices.
        
        Returns:
            chars (Tensor)
            feats (Tensor)
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # lists
        chars = self.CHAR.transform(sentences)
        chars = pad(chars, self.CHAR.pad_index).to(device)
        # features
        if self.args.feat == 'bert':
            feats = self.FEAT.transform(sentences)
            feats = [pad(f, self.FEAT.pad_index).to(device)
                     for f in zip(*feats)]
        elif self.args.feat == 'bigram':
            feats = self.BIGRAM.transform(sentences)
            feats = pad(feats, self.BIGRAM.pad_index).to(device)

        return chars, feats


    def decode(self, s_span, s_label, transitions, start_transitions, mask, mask_cky=False):

        if mask_cky:
            pred_spans = cky_mask(s_span, transitions, start_transitions, mask)
        else:
            pred_spans = cky_simple(s_span, mask)
        
        batch_size, seq_len, _, label_size = s_label.shape

        corase = torch.nn.functional.one_hot(torch.tensor([self.CHART.sublabel_cluster(label) for label in self.CHART.vocab.itos]), self.args.n_sublabels).float().log().to(self.args.device)

        pred_labels = (s_label.view(batch_size, seq_len, seq_len, label_size, 1) + corase.view(1, 1, 1, label_size, self.args.n_sublabels)).argmax(-2)

        preds = [[(i, j, labels[i][j][l]) for i, j, l in spans]
                for spans, labels in zip(pred_spans, pred_labels)]

        return preds

    
