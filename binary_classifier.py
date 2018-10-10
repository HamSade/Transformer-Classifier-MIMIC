# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:46:24 2018

@author: hamed
"""

import torch
from transformer.Models import Encoder
from transformer.SubLayers import PositionwiseFeedForward

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output
        

class model(nn.Module):
    
    def __init__(self,d_src_vec=1440, len_max_seq=10, d_word_vec=512,
            n_head=8, d_k=512//8, d_v=512//8,
            d_model=512, d_inner=2048, dropout=0.1):
                
        self.fnn = PositionwiseFeedForward()
        
        self.encoder = Encoder(d_src_vec=d_src_vec, len_max_seq=len_max_seq, d_word_vec=d_word_vec,
            n_layers=3, n_head=n_head, d_k=d_src_vec//n_head, d_v=d_src_vec//n_head,
            d_model=512, d_inner=2048, dropout=0.1)

#        self.average_pooling()
        self.FC1 = nn.Linear(512, 64)  
        self.FC2 = nn.Linear(64, 8)
        self.FC3 = nn.Linear(8, 2)
        self.softmax = nn.sof
        
    def forward(self, x):
        
        
        
        