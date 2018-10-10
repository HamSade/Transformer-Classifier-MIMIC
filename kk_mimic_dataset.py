# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 10:40:35 2018

@author: hamed
"""

import torch
from torch.utils import data
#import numpy as np
import sklearn.datasets as datasets
#import matplotlib.pyplot as plt


class kk_mimic_dataset(data.Dataset):
    
    def __init__(self, training=True):
        if training: 
            data_path = "../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_TRAIN"
        else:
            data_path = "../mimic-libsvm/" + "PATIENTS_SPLIT_XGB_VALID"
            
        data = datasets.load_svmlight_file(data_path)
        self.features = data[0].todense()
        self.labels = data[1]
        
        # Removing last irrelevant features
        self.temporal_features = self.features[:,:14400]
        self.fixed_features = self.features[:,14400:]                
        
        print(self.features.shape)
        
    def __len__(self):
        return self.labels.shape[0]
        
    def __getitem__(self, index):
        return self.temporal_features[index], self.labels[index], self.fixed_features[index]
    
    
    
#%% Data loader
        
def loader(dataset, training=True, batch_size=64, shuffle=True, num_workers=1):
    params = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers':num_workers}
    return data.DataLoader(dataset, **params)
                

#%% Test dataloader

#training_set = kk_mimic_dataset()
#data_loader = loader(training_set)















