# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 16:46:24 2018

@author: hamed
"""

import torch
from transformer.Models import Encoder



class model(nn.Module):
    
    def __init__(self,):
        self.model = Encoder(...)
        self.FC = nn.Linear(..., ...)        
        self.average_pooling()
        
    def forward(self, ):
                
    
        
        