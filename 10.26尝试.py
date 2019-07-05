# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:56:00 2018

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 09:51:30 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

#submission = pd.read_csv('submission.csv',header=None, index_col=None)
res1 = pd.read_csv('./resnet/desnet169_99.2.csv',header=None, index_col=None)
res2 = pd.read_csv('./resnet/desnet169_666_99.2.csv',header=None, index_col=None)
res3 = pd.read_csv('./resnet/fc_res152_seed666_submission98.8.csv',header=None, index_col=None)
res4 = pd.read_csv('./resnet/169_555_99.1.csv',header=None, index_col=None)
res5 = pd.read_csv('./resnet/desner169_123_99.csv',header=None, index_col=None)
res6 = pd.read_csv('./resnet/99.3_169_666_3fold.csv',header=None, index_col=None)
res7 = pd.read_csv('./resnet/resnet152_888_submission98.6.csv',header=None, index_col=None)
res8 = pd.read_csv('./resnet/resnet152_888_98.9_submission.csv',header=None, index_col=None)

final = pd.DataFrame({ 0: res1[0], '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1],
                                   '4label': res4[1], 
                                   '5label': res5[1],
                                   '6label': res6[1],
                                   '7label': res7[1], 
                                   '8label': res8[1]
                                   
                                   
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
                           final['8label'][i]
                  
                           ]).most_common(1)[0][0]
    
result= final[[0,1]]
result.to_csv('./resnet/8.csv',index=False,header=False)
