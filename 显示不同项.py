# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 09:51:56 2018

@author: maiquer
"""

import numpy as np
import pandas as pd
from collections import Counter


t1 = pd.read_csv('./resnet/desnet169_99.2.csv',header=None, index_col=None)
t2 = pd.read_csv('./resnet/desnet169_666_99.2.csv',header=None, index_col=None)
t3 = pd.read_csv('./resnet/3best_sub99.4.csv',header=None, index_col=None)
t4 = pd.read_csv('./resnet/3_sub99.6.csv',header=None, index_col=None)
t5 = pd.read_csv('./resnet/99.3_169_666_3fold.csv',header=None, index_col=None)
t6 = pd.read_csv('./resnet/vote169_256_99.2.csv',header=None, index_col=None)
t7 = pd.read_csv('./resnet/acc100.csv',header=None, index_col=None)
t8 = pd.read_csv('./resnet/submission.csv',header=None, index_col=None)
t9 = pd.read_csv('./resnet/3.csv',header=None, index_col=None)
t10 = pd.read_csv('./des/6.csv',header=None, index_col=None)

final1 = pd.merge(t8,t7,how='left',on=[0])
final1[3]=0
for i in range(1000):
    if(final1['1_x'][i]!=final1['1_y'][i]):
        print(i)