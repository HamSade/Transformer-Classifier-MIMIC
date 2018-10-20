# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 17:41:17 2018

@author: hamed
"""

import numpy as np
import matplotlib.pyplot as plt


file_name = "stats.npy"
stats = ("mean_", "scale_", "min_", "max_", "var_")


x = np.load(file_name)

print("x.shape = ", x.shape)

for i in range(5):
    plt.figure(i)
    plt.title("{}".format(stats[i]))
    plt.plot(np.arange(1440), x[i,:])


