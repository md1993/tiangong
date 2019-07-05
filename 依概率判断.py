# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 11:25:23 2018

@author: maiquer
"""

import pandas as pd
import numpy as np

des = pd.read_csv('./pro/rand999_desnet_96.10_prob.csv')
res = pd.read_csv('./pro/rand888_resnet_95.43_prob.csv')
v4 = pd.read_csv('./pro/v4_rand666_94.9_prob.csv')


def csv2csv(result):
    tf = []
    for index, row in result.iterrows():
        temp = row['probability']
        t = temp.split("[")[1].split("]")[0].split(",")
        for j in range(12):
            tf.append(float(t[j]))
    n = 12  #大列表中几个数据组成一个小列表
    child = [tf[i:i + n] for i in range(0, len(tf), n)]
    cnp = np.array(child)
    a = list(result['filename'].values)
    df = pd.DataFrame(cnp,index=a)
    return df

des = csv2csv(des)
v4 = csv2csv(v4)
res = csv2csv(res)


des_index = list(des.index)


#tmp = pd.merge(des,v4,how='left')
all_list=[]
for pic_name in list(des.index):
    one_list= []
    for j in range((des.iloc[0]).shape[0]):
        one_list.append(des.loc[pic_name][j] + v4.loc[pic_name][j] + res.loc[pic_name][j])
    all_list.append(np.array(one_list))
all_list = np.array(all_list)
class_num= np.argmax(all_list,axis = 1)
sub_label=[]
for i in class_num:
            if i == 0:
                sub_label.append('norm')
            else:
                sub_label.append('defect%d' % i)

a =pd.DataFrame(sub_label,index=des.index)   
a.to_csv('./pro/result/conv9+8+6.csv',header=False)
        
