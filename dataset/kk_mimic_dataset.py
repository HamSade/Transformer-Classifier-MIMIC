# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:40:35 2018

@author: hamed 
"""

import torch
from torch.utils import data
import numpy as np
#import sklearn
from sklearn import datasets

#%%
class kk_mimic_dataset(data.Dataset):
    
    def __init__(self, phase="train", seq_len=10, data_norm=True):
        
        super(kk_mimic_dataset, self).__init__()
        if phase == "train": 
            data_path = "../../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_TRAIN"
            data = datasets.load_svmlight_file(data_path)           
 
        else:
            data_path = "../../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_VALID"
            data = np.array(datasets.load_svmlight_file(data_path))
           
            if  phase == "valid":#               
                data = [ data[0][:data[1].shape[0]//10], data[1][:data[1].shape[0]//10] ]  #TODO 10% for validation
            else:            
                data = [ data[0][data[1].shape[0]//10:], data[1][data[1].shape[0]//10:] ]  #TODO 90% for test
                
        # ONLY for fast debugging
        data = [ data[0][:data[1].shape[0]//100], data[1][:data[1].shape[0]//100] ] #Only 1/100 of all patients
        
        self.d_feat = 14400
        self.seq_len = seq_len
        self.features = np.array(data[0].todense())
        self.labels = np.array(data[1])
        
        # Removing last irrelevant features
        self.temporal_features = np.split(self.features[:,:self.d_feat], self.seq_len, axis=1)
        self.temporal_features = np.reshape(self.temporal_features, (-1,self.seq_len, self.d_feat//self.seq_len))
        self.fixed_features = self.features[:,self.d_feat:]                
        
        #Data normalization 
        if data_norm:
            file_name = "stats.npy"
            stats = np.load(file_name)  #stats = ("mean_", "scale_", "min_", "max_", "var_") * 1440
            mean_ = stats[0,:]
            scale_ = stats[1,:]
            self.temporal_features = np.divide( np.subtract(self.temporal_features, mean_), scale_)
        
    #%%   
    def __len__(self):
        return self.labels.shape[0]
       
    #%%
    def __getitem__(self, index):
        src_seq = self.temporal_features[index]
        src_fixed_feats = self.fixed_features[index]
        gold = self.labels[index]        
        src_pos = np.array([pos_i for pos_i, _ in enumerate(src_seq)])  #TODO pos_i <- pos_i + 1 ??!        
        src_seq = torch.FloatTensor(src_seq)
        src_pos = torch.LongTensor(src_pos)
        src_fixed_feats = torch.FloatTensor(src_fixed_feats)
        gold = torch.LongTensor( [gold] )
        return src_seq, src_pos, gold, src_fixed_feats

#%% Data loader
        
def loader(dataset, batch_size=64, shuffle=True, num_workers=1):
    params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers':num_workers}
    return data.DataLoader(dataset, **params) #, collate_fn=collate_fn_temp)
                



