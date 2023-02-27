##########################################################
# Copyright 2019 Oath Inc.
# Licensed under the terms of the MIT license.
# Please see LICENSE file in the project root for terms.
##########################################################
# This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import BoxRelationalEmbedding, BoxTimeRelationalEmbedding, BoxTimeIdRelationalEmbedding
import copy
import math
import numpy as np

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

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
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, h, d_model, d_ff, trignometric_embedding, legacy_extra_skip, dropout, box_feats):
        super(EncoderLayer, self).__init__()
        self.self_attn = BoxMultiHeadedAttention(h, d_model, trignometric_embedding, legacy_extra_skip, dropout, box_feats)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, box, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, box, mask))
        return self.sublayer[1](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def box_attention(query, key, value, box_relation_embds_matrix, mask=None, dropout=None):
    '''
    Compute 'Scaled Dot Product Attention as in paper Relation Networks for Object Detection'.
    Follow the implementation in https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1026-L1055
    '''

    N = value.size()[:2]
    dim_k = key.size(-1)
    dim_g = box_relation_embds_matrix.size()[-1]

    w_q = query
    w_k = key.transpose(-2, -1)
    w_v = value

    #attention weights
    scaled_dot = torch.matmul(w_q,w_k)
    scaled_dot = scaled_dot / np.sqrt(dim_k)
    if mask is not None:
        scaled_dot = scaled_dot.masked_fill(mask == 0, -1e4)

    #w_g = box_relation_embds_matrix.view(N,N)
    w_g = box_relation_embds_matrix
    w_a = scaled_dot

    # multiplying log of geometric weights by feature weights
    w_mn = torch.log(torch.clamp(w_g, min = 1e-6)) + w_a
    w_mn = torch.nn.Softmax(dim=-1)(w_mn)
    if dropout is not None:
        w_mn = dropout(w_mn)

    output = torch.matmul(w_mn,w_v)

    return output, w_mn

class BoxMultiHeadedAttention(nn.Module):
    '''
    Self-attention layer with relative position weights.
    Following the paper "Relation Networks for Object Detection" in https://arxiv.org/pdf/1711.11575.pdf
    '''

    def __init__(self, h, d_model, trignometric_embedding=True, legacy_extra_skip=False, dropout=0.1, box_feats=6):
        "Take in model size and number of heads."
        super(BoxMultiHeadedAttention, self).__init__()

        assert d_model % h == 0
        self.trignometric_embedding=trignometric_embedding
        self.legacy_extra_skip = legacy_extra_skip

        # We assume d_v always equals d_k
        self.h = h
        self.d_k = d_model // h
        self.box_feats = box_feats
        if self.trignometric_embedding:
            self.dim_g = 16*self.box_feats
        else:
            self.dim_g = self.box_feats
        geo_feature_dim = self.dim_g

        #matrices W_q, W_k, W_v, and one last projection layer
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.WGs = clones(nn.Linear(geo_feature_dim, 1, bias=True),h)

        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input_query, input_key, input_value, input_box, mask=None):
        "Implements Figure 2 of Relation Network for Object Detection"
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = input_query.size(0)

        #tensor with entries R_mn given by a hardcoded embedding of the relative position between bbox_m and bbox_n
        if self.box_feats == 4:
            relative_geometry_embeddings = BoxRelationalEmbedding(input_box, trignometric_embedding= self.trignometric_embedding)
        elif self.box_feats == 5:
            relative_geometry_embeddings = BoxTimeRelationalEmbedding(input_box, trignometric_embedding= self.trignometric_embedding)
        elif self.box_feats == 6:
            relative_geometry_embeddings = BoxTimeIdRelationalEmbedding(input_box, trignometric_embedding= self.trignometric_embedding)
        #relative_geometry_embeddings = BoxRelationalEmbedding(input_box, trignometric_embedding= self.trignometric_embedding)
        flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1,self.dim_g)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (input_query, input_key, input_value))]
        box_size_per_head = list(relative_geometry_embeddings.shape[:3])
        box_size_per_head.insert(1, 1)
        relative_geometry_weights_per_head = [l(flatten_relative_geometry_embeddings).view(box_size_per_head) for l in self.WGs]
        relative_geometry_weights = torch.cat((relative_geometry_weights_per_head),1)
        relative_geometry_weights = F.relu(relative_geometry_weights)

        # 2) Apply attention on all the projected vectors in batch.
        x, self.box_attn = box_attention(query, key, value, relative_geometry_weights, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)

        if self.legacy_extra_skip:
            x = input_value + x

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

class RelationTransformerModel(nn.Module):

    def __init__(self, opt):
        super(RelationTransformerModel, self).__init__()
        self.box_trignometric_embedding = True
        self.legacy_extra_skip = False
        self.input_encoding_size = 2048
        self.rnn_size = 2048
        self.box_feats = opt.BOX_FEATS
        self.heads = opt.HEADS
        self.layers = opt.LAYERS

        self.layer1 = EncoderLayer(h=self.heads, d_model=self.input_encoding_size,
                                   d_ff=self.rnn_size,
                                   trignometric_embedding=self.box_trignometric_embedding,
                                   legacy_extra_skip=self.legacy_extra_skip, 
                                   dropout=0.1,
                                   box_feats=self.box_feats)
        self.norm1 = LayerNorm(self.input_encoding_size)

        for p in self.layer1.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for p in self.norm1.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        if self.layers > 1:
            self.layer2 = EncoderLayer(h=self.heads, d_model=self.input_encoding_size,
                                       d_ff=self.rnn_size,
                                       trignometric_embedding=self.box_trignometric_embedding,
                                       legacy_extra_skip=self.legacy_extra_skip,
                                       dropout=0.1,
                                       box_feats=self.box_feats)
            self.norm2 = LayerNorm(self.input_encoding_size)

            for p in self.layer2.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.norm2.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        if self.layers > 2:
            self.layer3 = EncoderLayer(h=self.heads, d_model=self.input_encoding_size,
                                       d_ff=self.rnn_size,
                                       trignometric_embedding=self.box_trignometric_embedding,
                                       legacy_extra_skip=self.legacy_extra_skip,
                                       dropout=0.1,
                                       box_feats=self.box_feats)
            self.norm3 = LayerNorm(self.input_encoding_size)

            for p in self.layer3.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            for p in self.norm3.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def forward(self, att_feats, boxes, att_masks=None):
        x = self.norm1(self.layer1(att_feats, boxes, att_masks))/10.
        if self.layers > 1:
            x = x + self.norm2(self.layer2(x, boxes, att_masks))/10.
            if self.layers > 2:
                x = x + self.norm3(self.layer3(x, boxes, att_masks))/10.
        return x
