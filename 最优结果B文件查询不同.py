import numpy as np
import pandas as pd
from collections import Counter

t1 = pd.read_csv('./B/100.csv',header=None, index_col=None)
t2 = pd.read_csv('./B/5.csv',header=None, index_col=None)

res_final = pd.merge(t2 , t1, how='left', on=[0])
res_final[1] = 0
n = 0
for i in range(1000):
    if(res_final['1_x'][i]!=res_final['1_y'][i]):
        n = n+1
        print(i)
print(n)