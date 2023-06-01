import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_csv('./train.tsv',sep='\t',header=None,
               index_col=False,names=["s1", "s2", "label", "ans"])

check_nan=df['s1'].isnull().sum().sum()
check_nan2=df['s2'].isnull().sum().sum()
check_nan3=df['ans'].isnull().sum().sum()
print(check_nan)
print(check_nan2)
print(check_nan3)
# df.fillna(value='nothing',inplace=True)
#
# df.to_csv('./train1.tsv',index=None,header=None,sep='\t')