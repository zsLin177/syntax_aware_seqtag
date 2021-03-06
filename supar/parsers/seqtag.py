# -*- coding: utf-8 -*-

import os

import pdb
from re import X

import torch
import torch.nn as nn
from supar.models import (SimpleSeqTagModel, CrfSeqTagModel)
from supar.parsers.parser import Parser
from supar.utils import Config, Dataset, Embedding
from supar.utils.common import bos, pad, unk, eos
from supar.utils.field import Field, SubwordField
from supar.utils.logging import get_logger, progress_bar
from supar.utils.metric import SeqTagMetric
from supar.utils.transform import CoNLL

logger = get_logger(__name__)


class SimpleSeqTagParser(Parser):
    r"""
    TODO:introduction
    """
    NAME = 'SimpleSeqTag'
    MODEL = SimpleSeqTagModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), SeqTagMetric(self.LABEL.vocab)

        for i, batch in enumerate(bar, 1):
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            loss = self.model.loss(score, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            preds = self.model.decode(score)[:, 1:]
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")


    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)

        for batch in loader:
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            preds = self.model.decode(score)[:, 1:]
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
        # total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)
        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        # for words, *feats, labels in loader:
        for batch in loader:
            # pdb.set_trace()
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            output = self.model.decode(score)[:, 1:]
            mask = mask[:, 1:]
            metric(output.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            lens = mask.sum(-1).tolist()
            preds['labels'].extend(out[0:i].tolist() for i, out in zip(lens, output))
        preds['labels'] = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None] for i in out]) for out in preds['labels']]
        # total_loss /= len(loader)
        print(metric)
        return preds

    @classmethod
    def build(cls, path, min_freq=3, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if args.encoder == 'bert':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField(
                'words',
                pad=t.pad_token,
                unk=t.unk_token,
                bos=t.bos_token or t.cls_token,
                tokenize=t.tokenize,
                fn=None if not isinstance(t,
                                          (GPT2Tokenizer, GPT2TokenizerFast))
                else lambda x: ' ' + x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=bos)
            if 'char' in args.feat:
                CHAR = SubwordField('chars',
                                    pad=pad,
                                    unk=unk,
                                    bos=bos,
                                    fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField(
                    'bert',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    fix_len=args.fix_len,
                    tokenize=t.tokenize,
                    fn=None
                    if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                BERT.vocab = t.get_vocab()
        # LABEL = ChartField('labels', fn=CoNLL.get_labels)
        # word:[bos, seq, eos] spans:[bos, seq, eos]
        LABEL = Field('labels')
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=LABEL)

        train = Dataset(transform, args.train)
        if args.encoder != 'bert':
            WORD.build(
                train, args.min_freq,
                (Embedding.load(args.embed, args.unk) if args.embed else None))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        # LABEL.build(train)
        LABEL.build(train)
        args.update({
            'n_words':
            len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'bos_index':
            WORD.bos_index,
            'unk_index':
            WORD.unk_index,
            'interpolation': args.itp,
            'syntax': args.use_syntax,
            'synatax_path': args.synatax_path,
            'mix': args.mix
        })

        if(args.encoder == 'bert'):
            args.update({
            'lr':
            5e-5,
            'epochs': 20, 
            'warmup':
            0.1
        })
        elif args.encoder == 'lstm':
            args.update({
            'lr':
            1e-3,
            'epochs': 5000, 
            'mu': .0,
            'nu': .95,
            'eps': 1e-12,
            'weight_decay': 3e-9,
            'decay': .75,
            'decay_steps': 5000
        })
        elif args.encoder =='transformer':
            args.update({
            'lr':
            0.04,
            'epochs': 5000, 
            'warmsteps':1000
        })

        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)


class CrfSeqTagParser(Parser):
    r"""
    TODO:introduction
    """
    NAME = 'CrfSeqTag'
    MODEL = CrfSeqTagModel

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.WORD, self.CHAR, self.BERT = self.transform.FORM
        self.LEMMA = self.transform.LEMMA
        self.TAG = self.transform.POS
        self.LABEL = self.transform.PHEAD

    def train(self,
              train,
              dev,
              test,
              buckets=32,
              batch_size=5000,
              update_steps=1,
              verbose=True,
              **kwargs):
        r"""
        Args:
            train/dev/test (list[list] or str):
                Filenames of the train/dev/test datasets.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            update_steps (int):
                Gradient accumulation steps. Default: 1.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments for updating training configurations.
        """

        return super().train(**Config().update(locals()))

    def evaluate(self,
                 data,
                 buckets=8,
                 batch_size=5000,
                 verbose=True,
                 **kwargs):
        r"""
        Args:
            data (str):
                The data for evaluation, both list of instances and filename are allowed.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for evaluation.

        Returns:
            The loss scalar and evaluation results.
        """

        return super().evaluate(**Config().update(locals()))

    def predict(self,
                data,
                pred=None,
                lang='en',
                buckets=8,
                batch_size=5000,
                verbose=True,
                **kwargs):
        r"""
        Args:
            data (list[list] or str):
                The data for prediction, both a list of instances and filename are allowed.
            pred (str):
                If specified, the predicted results will be saved to the file. Default: ``None``.
            lang (str):
                Language code (e.g., 'en') or language name (e.g., 'English') for the text to tokenize.
                ``None`` if tokenization is not required.
                Default: ``en``.
            buckets (int):
                The number of buckets that sentences are assigned to. Default: 32.
            batch_size (int):
                The number of tokens in each batch. Default: 5000.
            prob (bool):
                If ``True``, outputs the probabilities. Default: ``False``.
            verbose (bool):
                If ``True``, increases the output verbosity. Default: ``True``.
            kwargs (dict):
                A dict holding the unconsumed arguments that can be used to update the configurations for prediction.

        Returns:
            A :class:`~supar.utils.Dataset` object that stores the predicted results.
        """

        return super().predict(**Config().update(locals()))

    def _train(self, loader):
        self.model.train()

        bar, metric = progress_bar(loader), SeqTagMetric(self.LABEL.vocab)

        for i, batch in enumerate(bar, 1):
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            loss = self.model.loss(score, labels, mask)
            loss = loss / self.args.update_steps
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)
            if i % self.args.update_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            preds = self.model.decode(score, mask)
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            bar.set_postfix_str(
                f"lr: {self.scheduler.get_last_lr()[0]:.4e} - loss: {loss:.4f} - {metric}"
            )
        logger.info(f"{bar.postfix}")


    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)

        # for words, *feats, labels in loader:
        for batch in loader:
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            preds = self.model.decode(score, mask)
            mask = mask[:, 1:]
            metric(preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
        # total_loss /= len(loader)

        return metric

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()

        total_loss, metric = 0, SeqTagMetric(self.LABEL.vocab)
        preds = {'labels': [], 'probs': [] if self.args.prob else None}
        # for words, *feats, labels in loader:
        for batch in loader:
            # pdb.set_trace()
            sentences_lst = [s.words for s in batch.sentences]
            words, *feats, labels = batch
            word_mask = words.ne(self.args.pad_index) & words.ne(self.args.bos_index)
            mask = word_mask if len(words.shape) < 3 else word_mask.any(-1)
            score = self.model(words, sentences_lst, feats)
            output = self.model.decode(score, mask)
            mask = mask[:, 1:]
            metric(output.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
            lens = mask.sum(-1).tolist()
            preds['labels'].extend(out[0:i].tolist() for i, out in zip(lens, output))
        preds['labels'] = [CoNLL.build_relations([[self.LABEL.vocab[i] if i >= 0 else None] for i in out]) for out in preds['labels']]
        # total_loss /= len(loader)
        print(metric)
        return preds

    @classmethod
    def build(cls, path, min_freq=3, fix_len=20, **kwargs):
        r"""
        Build a brand-new Parser, including initialization of all data fields and model parameters.

        Args:
            path (str):
                The path of the model to be saved.
            min_freq (str):
                The minimum frequency needed to include a token in the vocabulary. Default:7.
            fix_len (int):
                The max length of all subword pieces. The excess part of each piece will be truncated.
                Required if using CharLSTM/BERT.
                Default: 20.
            kwargs (dict):
                A dict holding the unconsumed arguments.
        """

        args = Config(**locals())
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and not args.build:
            parser = cls.load(**args)
            parser.model = cls.MODEL(**parser.args)
            parser.model.load_pretrained(parser.WORD.embed).to(args.device)
            return parser

        logger.info("Building the fields")
        
        TAG, CHAR, LEMMA, BERT = None, None, None, None
        if args.encoder == 'bert':
            from transformers import (AutoTokenizer, GPT2Tokenizer,
                                      GPT2TokenizerFast)
            t = AutoTokenizer.from_pretrained(args.bert)
            WORD = SubwordField(
                'words',
                pad=t.pad_token,
                unk=t.unk_token,
                bos=t.bos_token or t.cls_token,
                tokenize=t.tokenize,
                fn=None if not isinstance(t,
                                          (GPT2Tokenizer, GPT2TokenizerFast))
                else lambda x: ' ' + x)
            WORD.vocab = t.get_vocab()
        else:
            WORD = Field('words', pad=pad, unk=unk, bos=bos, lower=True)
            if 'tag' in args.feat:
                TAG = Field('tags', bos=bos)
            if 'char' in args.feat:
                CHAR = SubwordField('chars',
                                    pad=pad,
                                    unk=unk,
                                    bos=bos,
                                    fix_len=args.fix_len)
            if 'lemma' in args.feat:
                LEMMA = Field('lemmas', pad=pad, unk=unk, bos=bos, lower=True)
            if 'bert' in args.feat:
                from transformers import (AutoTokenizer, GPT2Tokenizer,
                                          GPT2TokenizerFast)
                t = AutoTokenizer.from_pretrained(args.bert)
                BERT = SubwordField(
                    'bert',
                    pad=t.pad_token,
                    unk=t.unk_token,
                    bos=t.bos_token or t.cls_token,
                    eos=t.sep_token,
                    fix_len=args.fix_len,
                    tokenize=t.tokenize,
                    fn=None
                    if not isinstance(t, (GPT2Tokenizer, GPT2TokenizerFast))
                    else lambda x: ' ' + x)
                BERT.vocab = t.get_vocab()
        # LABEL = ChartField('labels', fn=CoNLL.get_labels)
        # word:[bos, seq, eos] spans:[bos, seq, eos]
        LABEL = Field('labels')
        transform = CoNLL(FORM=(WORD, CHAR, BERT),
                          LEMMA=LEMMA,
                          POS=TAG,
                          PHEAD=LABEL)

        train = Dataset(transform, args.train)
        if args.encoder != 'bert':
            WORD.build(
                train, args.min_freq,
                (Embedding.load(args.embed, args.unk) if args.embed else None))
            if TAG is not None:
                TAG.build(train)
            if CHAR is not None:
                CHAR.build(train)
            if LEMMA is not None:
                LEMMA.build(train)
        # LABEL.build(train)
        LABEL.build(train)
        args.update({
            'n_words':
            len(WORD.vocab) if args.encoder == 'bert' else WORD.vocab.n_init,
            'n_labels':
            len(LABEL.vocab),
            'n_tags':
            len(TAG.vocab) if TAG is not None else None,
            'n_chars':
            len(CHAR.vocab) if CHAR is not None else None,
            'char_pad_index':
            CHAR.pad_index if CHAR is not None else None,
            'n_lemmas':
            len(LEMMA.vocab) if LEMMA is not None else None,
            'bert_pad_index':
            BERT.pad_index if BERT is not None else None,
            'pad_index':
            WORD.pad_index,
            'bos_index':
            WORD.bos_index,
            'unk_index':
            WORD.unk_index,
            'interpolation': args.itp,
            'syntax': args.use_syntax,
            'synatax_path': args.synatax_path,
            'mix': args.mix
        })

        if(args.encoder == 'bert'):
            args.update({
            'lr':
            5e-5,
            'epochs': 20, 
            'warmup':
            0.1
        })
        elif args.encoder == 'lstm':
            args.update({
            'lr':
            1e-3,
            'epochs': 5000, 
            'mu': .0,
            'nu': .95,
            'eps': 1e-12,
            'weight_decay': 3e-9,
            'decay': .75,
            'decay_steps': 5000
        })
        elif args.encoder =='transformer':
            args.update({
            'lr':
            0.04,
            'epochs': 5000, 
            'warmsteps':1000
        })

        logger.info(f"{transform}")

        logger.info("Building the model")
        model = cls.MODEL(**args).load_pretrained(
            WORD.embed if hasattr(WORD, 'embed') else None).to(args.device)
        logger.info(f"{model}\n")

        return cls(args, model, transform)