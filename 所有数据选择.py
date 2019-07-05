# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 11:35:21 2018

@author: Administrator
"""

import numpy as np
import pandas as pd
from collections import Counter

t1 = pd.read_csv('./B/Bnew_resenet152_456_OHEM0.7_256_96.5.csv',header=None, index_col=None)
t2 = pd.read_csv('./B/Bnew_resnet152_456_ohem0.7_256.csv',header=None, index_col=None)
t3 = pd.read_csv('./B/Bnew_restart_resnet152_456_cross.csv',header=None, index_col=None)
t4 = pd.read_csv('./B/Bnew_restart_resnet152_666_cross.csv',header=None, index_col=None)
t5 = pd.read_csv('./B/Bnew_se_resnet101_456_cross_256.csv',header=None, index_col=None)
t6 = pd.read_csv('./B/Bnew_se_resnet101_456_ohem0.7_256_96.csv',header=None, index_col=None)

t7 = pd.read_csv('./B/B01_resnet152_456_ohem0.7_256.csv',header=None, index_col=None)
t8 = pd.read_csv('./B/B01_resnet152_456.csv',header=None, index_col=None)
t9 = pd.read_csv('./B/B01_se_resnet101_456_ohem0.7_256_color_dididi.csv',header=None, index_col=None)
t10 = pd.read_csv('./B/B01_se_resnet152_123_ohem0.7_256_color.csv',header=None, index_col=None)
t11 = pd.read_csv('./B/B01_se-resnet101_456_cross_256.csv',header=None, index_col=None)

t12 = pd.read_csv('./B/4_98.5.csv',header=None, index_col=None)
t13 = pd.read_csv('./B/5_98.6.csv',header=None, index_col=None)
t14 = pd.read_csv('./B/4des_98.4.csv',header=None, index_col=None)

t15 = pd.read_csv('./B/B02_resnet101_456_ohem0.7_256.csv',header=None, index_col=None)
t16 = pd.read_csv('./B/B02_RESNET152_456_OHEM0.7_256.csv',header=None, index_col=None)
t17 = pd.read_csv('./B/B02_resnet152_456.csv',header=None, index_col=None)
t18 = pd.read_csv('./B/B02_se_resenet101_456_cross_256.csv',header=None, index_col=None)

t = pd.read_csv('./B/99.5.csv',header=None, index_col=None)
final = pd.DataFrame({ 0: t1[0], 
                                   '1label': t1[1], 
                                   '2label': t2[1], 
                                   '3label': t3[1],
                                   '4label': t4[1],
                                   '5label': t5[1], 
                                   '6label': t6[1],
                                   '7label': t7[1],
                                   '8label': t8[1],
                                   '9label': t9[1],
                                   '10label': t10[1],
                                   '11label': t11[1],
                                   '12label': t12[1],
                                   '13label': t13[1],
                                   '14label': t14[1],
                                   '15label': t15[1],
                                   '16label': t16[1],
                                   '17label': t17[1],
                                   '18label': t18[1],
#                                   'xlabel': t[1],                       
                                   })

final[1]=0
for i in range(1000):
    final[1][i] = Counter([
                           final['1label'][i],
#                           final['2label'][i],
#                           final['3label'][i],
                           final['4label'][i],
#                           final['5label'][i],
#                           final['6label'][i],
#                           final['7label'][i],
#                           final['8label'][i],
#                           final['9label'][i],
#                           final['10label'][i],
                           final['11label'][i],
#                           final['12label'][i],
#                           final['13label'][i],
#                           final['14label'][i],
#                           final['15label'][i],
#                           final['16label'][i],
#                           final['17label'][i],
#                           final['18label'][i],
#                           final['xlabel'][i],
                           ]).most_common(1)[0][0]
    
result= final[[0,1]]
result.to_csv('./B/5.csv',index=False,header=False)
