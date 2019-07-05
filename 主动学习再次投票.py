import numpy as np
import pandas as pd
from collections import Counter


#res1 = pd.read_csv('./B/(1,2,3,4)98.8.csv',header=None, index_col=None)
#res2 = pd.read_csv('./B/(5,6,7)98.8.csv',header=None, index_col=None)
#res3 = pd.read_csv('./B/(8,9,10)98.9.csv',header=None, index_col=None)
#
#res4 = pd.read_csv('./B/learn_res101_123_cross_98.8.csv',header=None, index_col=None)
#res5 = pd.read_csv('./B/learn_res101_666_cross_98.6.csv',header=None, index_col=None)
#res6 = pd.read_csv('./B/learn_sam_res152_456_cross_98.6.csv',header=None, index_col=None)

res1 = pd.read_csv('./B/base_des161_123_cross_98.7.csv',header=None, index_col=None)
res2 = pd.read_csv('./B/learn_des121_cross_98.7.csv',header=None, index_col=None)
res3 = pd.read_csv('./B/learn_des161_cross_98.7.csv',header=None, index_col=None)

res4 = pd.read_csv('./B/learn_des169_456_cross_99.csv',header=None, index_col=None)
res5 = pd.read_csv('./B/learn_res101_123_cross_98.8.csv',header=None, index_col=None)
res6 = pd.read_csv('./B/learn_res101_666_cross_98.6.csv',header=None, index_col=None)
res7 = pd.read_csv('./B/learn_sam_des161_123_cross_98.6.csv',header=None, index_col=None)
res8 = pd.read_csv('./B/learn_sam_des169_456_cross_98.9.csv',header=None, index_col=None)
res9 = pd.read_csv('./B/learn_sam_des161_123_cross_98.6.csv',header=None, index_col=None)
res10= pd.read_csv('./B/learn_sam_res152_456_cross_98.6.csv',header=None, index_col=None)

t = pd.read_csv('./B/100.csv',header=None, index_col=None)

final = pd.DataFrame({ 0: res1[0], '1label': res1[1], 
                                   '2label': res2[1], 
                                   '3label': res3[1],
                                   '4label': res4[1], 
                                   '5label': res5[1], 
                                   '6label': res6[1], 
                                   '7label': res7[1],
                                   '8label': res8[1], 
                                   '9label': res9[1], 
                                   '10label': res10[1],
                                   'xlabel': t[1],         
                                   })

final[1]=0
for i in range(1000):
    final[1][i] = Counter([final['1label'][i],
                           final['2label'][i],
                           final['3label'][i],
                           final['4label'][i],
                           final['5label'][i],
                           final['6label'][i], 
                           final['7label'][i],
                           final['8label'][i],
                           final['9label'][i],
                           final['10label'][i], 
#                           final['xlabel'][i],
                           ]).most_common(1)[0][0]
    
pic_name_list = []
label_list =[]
n = 0 
result= final[[0,1]]
for j in range(1000):
    if (final['1label'][j]==final['2label'][j]==
        final['3label'][j]==final['4label'][j]==
        final['5label'][j]==final['6label'][j] == 
        final['7label'][j]==final['8label'][j]==
        final['9label'][j]==final['10label'][j]
#        == final['xlabel'][j]
        ) :
        pic_name_list.append(final[0][j])
        label_list.append(final['1label'][j])
        n=n+1
    else:
        final[1][j]=0
        
result= final[[0,1]]
result.to_csv('./B/re_learn.csv',index=False,header=False)

sampel_train = list(zip(pic_name_list,label_list))
pretrain = pd.DataFrame(data = sampel_train)  
pretrain.to_csv('./vote/re_pre_testb.csv')     
print(n)
#result.to_csv('allvote.csv',index=False,header=False)

