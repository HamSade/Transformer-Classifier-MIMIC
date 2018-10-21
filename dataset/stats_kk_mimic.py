# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 15:49:04 2018

@author: hamed
"""

import numpy as np

#import matplotlib.pyplot as plt
from tqdm import tqdm

#import pdb
from kk_mimic_dataset import kk_mimic_dataset

#%% Test dataloader
                
training_set = kk_mimic_dataset(data_norm=False)
validation_set = kk_mimic_dataset(phase='validation',data_norm=False)
test_set = kk_mimic_dataset(phase='test',data_norm=False)

print("len(training_set)", len(training_set))
print("len(validation_set)", len(validation_set))
print("len(test_set)", len(test_set))

#%%

num_feats = 1440  #per seq token

max_ = [-np.inf]*num_feats
min_ = [np.inf]*num_feats

sum_ = [0.]*num_feats
sum2_ = [0.]*num_feats

training_counter = 0
for x in tqdm(training_set):
    x = x[0]
    training_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j] #src_seq
        max_[j] = temp.max()
        min_[j] = temp.min()
        sum_[j] = temp.sum()
        sum2_[j] = np.power(temp,2).sum()
        
validation_counter = 0    
for x in tqdm(validation_set):
    x = x[0]
    validation_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j] #src_seq
        max_[j] = temp.max()
        min_[j] = temp.min()
        sum_[j] = temp.sum()
        sum2_[j] = np.power(temp,2).sum()

test_counter = 0        
for x in tqdm(test_set):
    x = x[0]
    test_counter += 1
    
    for j in range(num_feats):
        temp = x[:,j] #src_seq
        max_[j] = temp.max()
        min_[j] = temp.min()
        sum_[j] = temp.sum()
        sum2_[j] = np.power(temp,2).sum()
        
mean_ = np.divide (sum_, ( len(training_set) + len(validation_set) + len(test_set)) )
ex2_ = np.divide ( sum2_ , ( len(training_set) + len(validation_set) + len(test_set)) ) 
  
var_ = ex2_ - np.power(mean_, 2)

#%%
print("max(max_)", max(max_))
print("min(min_)", min(min_))

scale_ = np.maximum(np.abs(max_), np.abs(min_))

#%%
file_name = 'stats/stats.npy'
stats = (mean_, scale_, min_, max_, var_)
np.save(file_name, stats, allow_pickle=True, fix_imports=True)










