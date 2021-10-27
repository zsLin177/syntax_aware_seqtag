# -*- coding: utf-8 -*-

from functools import reduce

import torch
from supar.utils.common import MIN


class Semiring(object):
    r"""
    A semiring is defined by a tuple `<K, +, ×, 0, 1>` :cite:`goodman-1999-semiring`.
    `K` is a set of values;
    `+` is commutative, associative and has an identity element `0`;
    `×` is associative, has an identity element `1` and distributes over `+`.
    """

    zero = 0
    one = 1

    @classmethod
    def add(cls, x, y, out=None):
        return torch.add(x, y, out=out)

    @classmethod
    def add_(cls, x, y):
        return cls.add(x, y, out=x)

    @classmethod
    def sum(cls, x, dim=-1):
        return x.sum(dim)

    @classmethod
    def mul(cls, x, y, out=None):
        return torch.mul(x, y, out=out)

    @classmethod
    def mul_(cls, x, y):
        return cls.mul(x, y, out=x)

    @classmethod
    def dot(cls, x, y, dim=-1):
        return cls.sum(cls.mul(x, y), dim)

    @classmethod
    def prod(cls, x, dim=-1):
        return x.prod(dim)

    @classmethod
    def times(cls, *x):
        return reduce(lambda i, j: cls.mul(i, j), x)

    @classmethod
    def zero_(cls, x):
        return x.fill_(cls.zero)

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)

    @classmethod
    def zero_mask(cls, x, mask):
        return x.masked_fill(mask, cls.zero)

    @classmethod
    def zero_mask_(cls, x, mask):
        return x.masked_fill_(mask, cls.zero)

    @classmethod
    def one_mask(cls, x, mask):
        return x.masked_fill(mask, cls.one)

    @classmethod
    def one_mask_(cls, x, mask):
        return x.masked_fill_(mask, cls.one)

    @classmethod
    def convert(cls, x):
        return x

    @classmethod
    def unconvert(cls, x):
        return x


class LogSemiring(Semiring):
    r"""
    Log-space semiring: `<logsumexp, +, -inf, 0>`.
    """

    zero = MIN
    one = 0

    @classmethod
    def add(cls, x, y, out=None):
        return torch.logaddexp(x, y, out=out)

    @classmethod
    def sum(cls, x, dim=-1):
        return x.logsumexp(dim)

    @classmethod
    def mul(cls, x, y, out=None):
        return torch.add(x, y, out=out)

    @classmethod
    def prod(cls, x, dim=-1):
        return x.sum(dim)


class MaxSemiring(LogSemiring):
    r"""
    Max semiring `<max, +, -inf, 0>`.
    """

    @classmethod
    def sum(cls, x, dim=-1):
        return x.max(dim)[0]


def KMaxSemiring(k):
    r"""
    k-max semiring `<kmax, +, [-inf, -inf, ...], [0, -inf, ...]>`.
    """

    class KMaxSemiring(LogSemiring):

        @classmethod
        def convert(cls, x):
            return torch.cat((x.unsqueeze(0), cls.zero_(x.new_empty(k - 1, *x.shape))))

        @classmethod
        def sum(cls, x, dim=-1):
            dim = dim if dim >= 0 else x.dim() + dim
            x = x.permute(dim, *range(dim), *range(dim + 1, x.dim()))
            return x.reshape(-1, *x.shape[2:]).topk(k, 0)[0]

        @classmethod
        def mul(cls, x, y):
            return (x.unsqueeze(0) + y.unsqueeze(1)).reshape(-1, *x.shape[1:]).topk(k, 0)[0]

        @classmethod
        def one_(cls, x):
            x[:1].fill_(cls.one)
            x[1:].fill_(cls.zero)
            return x

    return KMaxSemiring


class EntropySemiring(LogSemiring):
    """
    Entropy expectation semiring: `<logsumexp, +, -inf, 0>` :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def convert(cls, x):
        return torch.stack((x, torch.zeros_like(x)))

    @classmethod
    def unconvert(cls, x):
        return x[-1]

    @classmethod
    def sum(cls, x, dim=-1):
        p = x[0].logsumexp(dim)
        r = x[0] - p.unsqueeze(dim)
        r = r.exp().mul((x[1] - r)).sum(dim)
        return torch.stack((p, r))

    @classmethod
    def mul(cls, x, y):
        return x + y

    @classmethod
    def zero_(cls, x):
        x[:-1].fill_(cls.zero)
        x[-1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)


class CrossEntropySemiring(LogSemiring):
    """
    Cross entropy expectation semiring: `<logsumexp, +, -inf, 0>` :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def convert(cls, x):
        return torch.cat((x, cls.one_(torch.empty_like(x[:1]))))

    @classmethod
    def unconvert(cls, x):
        return x[-1]

    @classmethod
    def sum(cls, x, dim=-1):
        p = x[:-1].logsumexp(dim)
        r = x[:-1] - p.unsqueeze(dim)
        r = r[0].exp().mul((x[-1] - r[1])).sum(dim)
        return torch.cat((p, r.unsqueeze(0)))

    @classmethod
    def mul(cls, x, y):
        return x + y

    @classmethod
    def zero_(cls, x):
        x[:-1].fill_(cls.zero)
        x[-1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)


class KLDivergenceSemiring(LogSemiring):
    """
    KL divergence expectation semiring: `<logsumexp, +, -inf, 0>` :cite:`li-eisner-2009-first`.
    """

    @classmethod
    def convert(cls, x):
        return torch.cat((x, cls.one_(torch.empty_like(x[:1]))))

    @classmethod
    def unconvert(cls, x):
        return x[-1]

    @classmethod
    def sum(cls, x, dim=-1):
        p = x[:-1].logsumexp(dim)
        r = x[:-1] - p.unsqueeze(dim)
        r = r[0].exp().mul((x[-1] - r[1] + r[0])).sum(dim)
        return torch.cat((p, r.unsqueeze(0)))

    @classmethod
    def mul(cls, x, y):
        return x + y

    @classmethod
    def zero_(cls, x):
        x[:-1].fill_(cls.zero)
        x[-1].fill_(cls.one)
        return x

    @classmethod
    def one_(cls, x):
        return x.fill_(cls.one)

class VarianceSemiring(LogSemiring):

    pass