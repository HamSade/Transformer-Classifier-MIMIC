# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 17:52:45 2018

@author: hamed
"""

from kk_mimic_dataset import kk_mimic_dataset
import numpy as np
import tqdm

import matplotlib.pyplot as plt

#%%
training_set = kk_mimic_dataset()
validation_set = kk_mimic_dataset(phase='validation')
test_set = kk_mimic_dataset(phase='test')

#%%

max_train = [-np.inf]*1440
min_train = [np.inf]*1440
sum_train = [0.]*1440

max_valid = [-np.inf]*1440
min_valid = [np.inf]*1440
sum_valid = [0.]*1440

max_test = [-np.inf]*1440
min_test = [np.inf]*1440
sum_test = [0.]*1440

#%%

training_counter = 0
for x in tqdm(training_set):
    x = x[0]
    training_counter += 1
    
    for j in range(1440):
        temp = x[:,j] #src_seq
        max_train[j] = temp.max()
        min_train[j] = temp.min()
        sum_train[j] = temp.sum()

#%%        
validation_counter = 0    
for x in tqdm(validation_set):
    x = x[0]
    validation_counter += 1
    
    for j in range(1440):
        temp = x[:,j] #src_seq
        max_valid[j] = temp.max()
        min_valid[j] = temp.min()
        sum_valid[j] = temp.sum()
        
#%%
test_counter = 0        
for x in tqdm(test_set):
    x = x[0]
    test_counter += 1
    
    for j in range(1440):
        temp = x[:,j] #src_seq
        max_test[j] = temp.max()
        min_test[j] = temp.min()
        sum_test[j] = temp.sum()
        
mean_train = np.divide (sum_train, ( len(training_set) + len(validation_set) + len(test_set)) )  
scale_train = np.maximum(np.abs(max_train), np.abs(min_train))

mean_valid = np.divide (sum_train, len(validation_set) )  
scale_valid = np.maximum(np.abs(max_valid), np.abs(min_valid))

mean_test = np.divide (sum_test, len(test_set) )  
scale_test = np.maximum(np.abs(max_test), np.abs(min_test))

#%%

plt.figure(1)
plt.title("{}".format(phase[i]))
plt.plot(mean_train)
plt.plot(scale_train)
plt.plot(mean_train)


print("mean_train", mean_)
print("", )
print("", )

print("", )
print("", )
print("", )

print("", )
print("", )
print("", )





