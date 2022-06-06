# This file contains Transformer network
# Most of the code is copied from http://nlp.seas.harvard.edu/2018/04/03/attention.html

# The cfg name correspondance:
# N=num_layers
# d_model=input_encoding_size
# d_ff=rnn_size
# h is always 8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils

import copy
import math
import numpy as np

from .CaptionModel import CaptionModel
from .AttModel import sort_pack_padded_sequence, pad_unsort_packed_sequence, pack_wrapper, AttModel


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, fc_feats):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src), src_mask,
                           tgt, tgt_mask, fc_feats)

    def encode(self, src):
        return self.encoder(self.src_embed(src))

    def decode(self, memory, src_mask, tgt, tgt_mask, fc_feats,final=False):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, fc_feats,final)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.norm(x + self.dropout(sublayer(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        #self.no_mask_layer=clones(layer, 1)
        #self.layers.requires_grad=False
        self.norm = LayerNorm(layer.size )
        # self.alpha = nn.Parameter(torch.Tensor(6), requires_grad=True)
        # self.gamma = nn.Parameter(torch.Tensor(1, 1), requires_grad=True)
        # torch.nn.init.constant(self.alpha, 1.0)
        # torch.nn.init.constant(self.gamma, 1.0)
        self.dropout = nn.Dropout(p=0.2)
        self.mlp=MlpBlock(768,3072,768)
        # self.fc = nn.Linear(1024, 512)
    def forward(self, x, memory, src_mask, tgt_mask, fc_feats,final=False):
        # for layer in self.layers:
        #     x = layer(x, memory, src_mask, tgt_mask)
        # return self.norm(x)

        # o_list = []
        # for layer in self.layers:
        #     x, o = layer(x, memory, src_mask, tgt_mask, fc_feats)
        #     o_list.append(o)
        # # return self.norm(x)
        # return self.linear_sum(o_list, self.alpha, self.gamma)

        o_list = []

        # memory=memory*src_mask
        memory=self.mlp(memory)
        memory = memory[:, 1:]
        #memory = memory * src_mask
        for layer in self.layers:
            x, o = layer(x, memory, src_mask, tgt_mask, fc_feats)
            o_list.append(o)
        # if x.shape[1]==17 or final:
        #     x,o=self.no_mask_layer[0](x,memory,src_mask,None,fc_feats)
        # # return self.norm(x)
        # return self.linear_sum(o_list, self.alpha, self.gamma)
        #feats = torch.cat([o_list[-1], o_list[-2]], dim=2)
        feats=x
        return self.norm(self.dropout(feats))

    # def linear_sum(self, x, alpha, gamma):
    #     alpha_softmax = F.softmax(alpha)
    #     for i in range(len(x)):
    #         t = x[i] * alpha_softmax[i] * gamma
    #         if i == 0:
    #             res = t
    #         else:
    #             res += t
    #     return res


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn,att_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.att_attn=att_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 4)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x, memory, src_mask, tgt_mask, fc_feats):
        "Follow Figure 1 (right) for connections."
        m = memory
        s = x

        x_kv = torch.cat((fc_feats.unsqueeze(1), x), dim=1)
        # # print(tgt_mask.size())
        # # print(x.size())

        t = tgt_mask
        # print(tgt_mask.size())
        l = tgt_mask.new_ones((x.size(0), tgt_mask.size(1), 1))
        # print(l.size())
        b = tgt_mask.new_ones((x.size(0), 1, tgt_mask.size(2)+1))
        #tgt_mask = torch.cat((l, tgt_mask), dim=-1)
        # print(tgt_mask.size())
        #tgt_mask = torch.cat((tgt_mask, b), dim=1)
        # tgt_mask = torch.ones((x.size(0), 32, 32), device=x.device)
        # tgt_mask[:, :-1, 1:] = t
        # print(x.size())
        # print(tgt_mask.size())
        # src_mask=src_mask[:,1:]
        #m=m[:,1:]
        #m=m*src_mask
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x_kv, x_kv, tgt_mask))
        # m_x=torch.cat((m,x),dim=1)
        # m_x = self.sublayer[3](m_x, lambda m_x: self.att_attn(m_x, m_x, m_x))
        #x = x[:, :-1, :]
        #tgt_mask = t
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        #x = self.src_attn(x, m, m)
        # return self.sublayer[2](x, self.feed_forward)
        x = self.dropout(self.sublayer[2](x, self.feed_forward))
        return s + x, x


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class MlpBlock(nn.Module):
    """ Transformer Feed-Forward Block """
    def __init__(self, in_dim, mlp_dim, out_dim, dropout_rate=0.1):
        super(MlpBlock, self).__init__()

        # init layers
        self.fc1 = nn.Linear(in_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, out_dim)
        self.act = nn.GELU()
        if dropout_rate > 0.0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)
        else:
            self.dropout1 = None
            self.dropout2 = None

    def forward(self, x):

        out = self.fc1(x)
        out = self.act(out)
        if self.dropout1:
            out = self.dropout1(out)

        out = self.fc2(out)
        out = self.dropout2(out)
        return out

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        #qk_weight=self.attn.cpu().detach().numpy()
        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(AttModel):

    def make_model(self, src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        model = EncoderDecoder(
            #Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), 1),
            lambda x: x,
            Decoder(DecoderLayer(d_model, c(attn), c(attn),c(attn),
                                 c(ff), dropout), N),
            # nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
            lambda x: x,
            nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
            Generator(d_model, tgt_vocab))

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, opt):
        super(TransformerModel, self).__init__(opt)
        self.opt = opt
        # self.config = yaml.load(open(opt.config_file))
        # d_model = self.input_encoding_size # 512

        delattr(self, 'att_embed')
        self.att_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) + (nn.Linear(self.att_feat_size, self.input_encoding_size),
                                                                              nn.ReLU(),
                                                                              nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ())))
        delattr(self, 'fc_embed')
        self.fc_embed = nn.Sequential(nn.Linear(self.att_feat_size, self.input_encoding_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))

        delattr(self, 'embed')
        self.embed = lambda x: x
        delattr(self, 'logit')
        del self.ctx2att

        tgt_vocab = self.vocab_size + 1
        self.model = self.make_model(0, tgt_vocab,
                                     N=opt.num_layers,
                                     d_model=opt.input_encoding_size,
                                     d_ff=opt.rnn_size)

    def logit(self, x):  # unsafe way
        return self.model.generator.proj(x)

    def init_hidden(self, bsz):
        return None

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        att_feats, seq, att_masks, seq_mask, fc_feats = self._prepare_feature_forward(
            att_feats, fc_feats, att_masks)
        memory = self.model.encode(att_feats)

        return fc_feats, att_feats[..., :1], memory, att_masks

    def _prepare_feature_forward(self, att_feats, fc_feats, att_masks=None, seq=None):
        #att_feats, att_masks = self.clip_att(att_feats, att_masks)
        #
        # att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)
        #
        # fc_feats = self.fc_embed(fc_feats)

        # if att_masks is None:
        #     att_masks = att_feats.new_ones(
        #         att_feats.shape[:2], dtype=torch.long)
        # att_masks = att_masks.unsqueeze(-2)

        if seq is not None:
            # crop the last one
            # seq=seq.view(-1,1,seq.size(2))
            seq = seq[:, :-1]
            #seq=seq[:,1:]
            # seq=seq.unsqueeze(1)
            seq_mask = (seq.data > 0).long()
            seq_mask[:, 0] += 1

            seq_mask = seq_mask.unsqueeze(-2)
            seq_mask = seq_mask & subsequent_mask(seq.size(-1)).to(seq_mask)
            random_mask = torch.rand((seq.size(0), seq.size(1))).cuda()
            # random_eye=torch.eye(seq_mask.size(1)).unsqueeze(0).cuda()
            # random_eye=random_eye.repeat(seq_mask.size(0),1,1)
            # random_mask=random_mask+random_eye
            random_mask = (random_mask > 0.3).long()

            seq=seq*random_mask

            #seq_mask = seq_mask * random_mask
            # mask=np.array(seq_mask.cpu())
        else:
            seq_mask = None

        return att_feats, seq, att_masks, seq_mask, fc_feats

    def _forward(self, fc_feats, att_feats, seq, att_masks=None):
        att_feats, seq, att_masks, seq_mask, fc_feats = self._prepare_feature_forward(
            att_feats, fc_feats, att_masks, seq)

        out = self.model(att_feats, seq, att_masks, seq_mask, fc_feats)

        outputs = self.model.generator(out)
        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def core(self, it, fc_feats_ph, att_feats_ph, memory, state, mask,final=False):
        """
        state = [ys.unsqueeze(0)]
        """
        if state is None:
            ys = it.unsqueeze(1)
        else:
            ys = torch.cat([state[0][0], it.unsqueeze(1)], dim=1)
        # print(ys.size())
        tmp_mask = subsequent_mask(ys.size(1)).to(memory.device)
        tmp_mask = tmp_mask.expand(ys.size(0), tmp_mask.size(1), tmp_mask.size(2))
        #tmp_mask=None
        # print(tmp_mask.size())
        out = self.model.decode(memory, mask,
                                ys,
                                tmp_mask,
                                fc_feats_ph,final)
        return out[:, -1], [ys.unsqueeze(0)]
