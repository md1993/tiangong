import numpy as np
import pandas as pd
from collections import Counter

#submission = pd.read_csv('submission.csv',header=None, index_col=None)
res1 = pd.read_csv('./resnet/desnet169_99.2.csv',header=None, index_col=None)
res2 = pd.read_csv('./resnet/desnet169_666_99.2.csv',header=None, index_col=None)
res3 = pd.read_csv('./resnet/3best_sub99.4.csv',header=None, index_col=None)
res4 = pd.read_csv('./resnet/169_555_99.1.csv',header=None, index_col=None)
res5 = pd.read_csv('./resnet/desner169_123_99.csv',header=None, index_col=None)

final = pd.DataFrame({ 0: res1[0], '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1],
                                   '4label': res4[1],
                                   '5label': res5[1]
                                   })

final[6]=0
for i in range(1000):
    final[6][i] = Counter([
                           final['1label'][i],
                           final['2label'][i],
                           final['3label'][i],
                           final['4label'][i],
                           final['5label'][i]
                           ]).most_common(1)[0][0]
    
result= final[[0,6]]
result.to_csv('./resnet/5_sub.csv',index=False,header=False)
