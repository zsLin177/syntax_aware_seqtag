import torch
import copy
import torch.nn as nn
import math
from torch.nn import (TransformerEncoder, TransformerEncoderLayer, MultiheadAttention)
from torch.nn import functional as F
from torch.nn import Linear, LayerNorm, Dropout
import pdb

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) /
                        emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size)).cuda()
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        # pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.pos_embedding = pos_embedding
        # self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        # token_embedding [batch_size, seq_len, dim]
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(1), :])


class SelfAttentionEncoder_Layerdrop(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 emb_size: int,
                 num_heads: int = 8,
                 dim_feedforward: int = 600,
                 dropout: float = 0.1,
                 position_embed = True,
                 batch_first = True,
                 norm_first = False):
        super(SelfAttentionEncoder_Layerdrop, self).__init__()
        self.num_layers = num_encoder_layers
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.position_embed = position_embed
        self.batch_first = batch_first
        self.norm_first = norm_first

        encoder_layer = TransformerEncoderLayer_Drop(
            d_model=emb_size, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=batch_first, dropout=dropout, norm_first= norm_first)
        self.transformer_encoder = TransformerEncoder_Drop(
            encoder_layer, num_layers=num_encoder_layers)
        self.positional_encoding = PositionalEncoding(emb_size,
                                                      dropout=dropout)

    def forward(self, src, pad_mask, if_layerdrop=False, p_layerdrop=0.3, if_selfattdrop=False, p_attdrop=0.5):
        '''
        src: [batch_size, seq_len, dim]
        pad_mask: [batch_size, seq_len] for pad, true means need to be pad
        '''
        if(self.position_embed):
            src_emb = self.positional_encoding(src)
        else:
            src_emb = src
        if(not self.batch_first):
            memory = self.transformer_encoder(src_emb.transpose(0, 1),
                                            mask=None,
                                            src_key_padding_mask=pad_mask,
                                            if_layerdrop=if_layerdrop,
                                            p_layerdrop=p_layerdrop,
                                            if_selfattdrop=if_selfattdrop,
                                            p_attdrop=p_attdrop)
            memory = memory.transpose(0, 1)
        else:
            memory = self.transformer_encoder(src_emb,
                                            mask=None,
                                            src_key_padding_mask=pad_mask,
                                            if_layerdrop=if_layerdrop,
                                            p_layerdrop=p_layerdrop,
                                            if_selfattdrop=if_selfattdrop,
                                            p_attdrop=p_attdrop)                                          
        return memory

    def __repr__(self):
        s = f"{self.emb_size}, num_layers={self.num_layers}, num_heads={self.num_heads}, dim_ffn={self.dim_feedforward}, position_embed={self.position_embed}, batch_first={self.batch_first}, norm_first={self.norm_first}"
        if self.dropout > 0:
            s += f", dropout={self.dropout}"

        return f"{self.__class__.__name__}({s})"

class TransformerEncoder_Drop(TransformerEncoder):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__(encoder_layer, num_layers)
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        # hyper-parameter

    def forward(self, src, mask=None, src_key_padding_mask=None, if_layerdrop=False, p_layerdrop=0.5, if_selfattdrop=False, p_attdrop=0.5):
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # pdb.set_trace()
        output = src
        if if_layerdrop == False:
            for mod in self.layers:
                output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, if_selfattdrop=if_selfattdrop, p_attdrop=p_attdrop)
        else:
            # 将tensor用从均匀分布中抽样得到的值填充 uniform distribution(0-1)
            dropout_probs = torch.empty(len(self.layers)).uniform_()
            # print() #预测的时候再打开
            # print("均匀分布概率：", dropout_probs)
            # temp_num = dropout_probs.gt(p_layerdrop)
            # print("drop的个数：",torch.sum(temp_num==0) )

            for i, mod in enumerate(self.layers):
                # pdb.set_trace()
                if dropout_probs[i] > p_layerdrop:
                    output =  mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, if_selfattdrop=if_selfattdrop, p_attdrop=p_attdrop)

        if self.norm is not None:
            output = self.norm(output)

        return output
def _get_clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoderLayer_Drop(TransformerEncoderLayer):
    
    __constants__ = ['batch_first', 'norm_first']
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,layer_norm_eps=1e-5, 
                batch_first=False, norm_first=False,device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,layer_norm_eps, 
                batch_first, norm_first, device, dtype)
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, if_selfattdrop=False, p_attdrop=0.5):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask, if_selfattdrop, p_attdrop)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask, if_selfattdrop, p_attdrop))
            x = self.norm2(x + self._ff_block(x))
        return x


    # self-attention block
    def _sa_block(self, x,
                  attn_mask, key_padding_mask, if_selfattdrop=False, p_attdrop=0.5):
        if if_selfattdrop == False:
            x = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=False)[0]
        else:
            # get Q*K attention weights
            attention = self.self_attn(x, x, x,
                            attn_mask=attn_mask,
                            key_padding_mask=key_padding_mask,
                            need_weights=True)[1]
            mask = torch.empty_like(attention).uniform_()
            # write to GPU
            # mask = mask.lt(float(p_attdrop)).to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            mask = mask.lt(p_attdrop)
            # '_' represent modify the variable attention
            attention.masked_fill_(mask, 0)
            x= torch.matmul(attention, x)
            
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)    

