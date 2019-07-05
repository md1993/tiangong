# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:52:38 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

#submission = pd.read_csv('submission.csv',header=None, index_col=None)
res1 = pd.read_csv('./des/vote169_256_99.2.csv',header=None, index_col=None)
res2 = pd.read_csv('./des/vote_OHEM_0.7_99.4_.csv',header=None, index_col=None)

res3 = pd.read_csv('./des/vote161_256-OHEM-0.8-99.2.csv',header=None, index_col=None)

res4 = pd.read_csv('./des/seresnet152_456_cross_3_99.2.csv',header=None, index_col=None)
res5 = pd.read_csv('./des/seresnet152_123_OHEM_3_99.2.csv',header=None, index_col=None)

res6 = pd.read_csv('./des/resnet101_1113_99.2.csv',header=None, index_col=None)
res7 = pd.read_csv('./des/resnet101_123_cross_3_99.csv',header=None, index_col=None)

res8 = pd.read_csv('./des/single_169_256_666_99.1.csv',header=None, index_col=None)
res9 = pd.read_csv('./des/resnet152_456_99.1.csv',header=None, index_col=None)

t = pd.read_csv('./resnet/best5_99.8.csv',header=None, index_col=None)
final = pd.DataFrame({ 0: res1[0], 
                                   '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1],
                                   '4label': res4[1],
                                   '5label': res5[1],
                                   '6label': res6[1],
                                   '7label': res4[1],
                                   '8label': res8[1],
                                   '9label': res9[1], 
                                   'xlabel': t[1],                       
                                   })

final[1]=0
for i in range(1000):
    final[1][i] = Counter([
                           final['1label'][i],
                           final['2label'][i],
                           final['3label'][i],
                           final['4label'][i],
                           final['5label'][i],
                           final['6label'][i],
                           final['7label'][i],
                           final['8label'][i],
                           final['9label'][i],
                           final['xlabel'][i],
                           ]).most_common(1)[0][0]
    
result= final[[0,1]]
result.to_csv('./des/6.csv',index=False,header=False)
