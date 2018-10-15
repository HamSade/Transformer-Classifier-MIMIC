# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:46:24 2018

@author: hamed
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer.Models import Encoder
#from transformer.SubLayers import PositionwiseFeedForward


class ffn_compressed(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, d_out, dropout=0.1):
        super(ffn_compressed, self).__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_out, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output)
        return output
        
        
class model(nn.Module):
    
    d_src_vec = 1440
    d_emb_vec = 304
    len_max_seq = 10
    n_layers=3
    n_head = 8
    dropout = 0.1
    d_inner = 2048
        
    def __init__(self, d_src_vec=d_inner,            
                 len_max_seq=len_max_seq,
                 d_emb_vec=d_emb_vec,
                 n_layers = n_layers,
                 n_head=n_head, d_k=d_emb_vec//n_head,
                 d_v=d_emb_vec//n_head, d_model=d_emb_vec,
                 d_inner=d_inner, dropout=dropout):
        
        self.d_src_vec = d_src_vec
        self.d_emb_vec = d_emb_vec
        self.len_max_seq = len_max_seq
        self.n_layers= n_layers
        self.n_head = n_head
        self.dropout = dropout
        self.d_inner = d_inner
        
        self.ffn = ffn_compressed(d_in=self.d_src_vec, d_hid=self.d_inner,
                                  d_out=self.d_emb_vec)
        
        self.encoder = Encoder(len_max_seq=self.len_max_seq, d_word_vec=self.d_word_vec,
            n_layers=self.n_layers, n_head=self.n_head, d_k=self.d_src_vec//self.n_head,
            d_v=self.d_src_vec//self.n_head, d_model=self.d_word_vec, d_inner=self.d_inner,
            dropout=self.dropout)

#        self.average_pooling()
        self.FC1 = nn.Linear(512, 64)  
        self.FC2 = nn.Linear(64, 8)
        self.FC3 = nn.Linear(8, 2)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x, x_pos):
        x = self.ffn(x)
        x, enc_slf_attn_list = self.encoder(x, x_pos, return_attns=True)
        x = self.FC1(x)
        x = self.FC2(x)
        x = self.FC3(x)
        return self.softmax(x), enc_slf_attn_list
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
