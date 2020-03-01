# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 23:19:00 2020

@author: folde
"""

import pandas as pd
import numpy as np
import glob
import scipy.stats as sp
import os
from pathlib import Path

path = os.getcwd() + '/tfrbm/logs/rbms/raw/params' + "/*.csv" # use your path
path = Path(path)
all_files = glob.glob(path.as_posix())

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

params = pd.concat(li, axis=0, ignore_index=True)
params = params.convert_dtypes()

path = os.getcwd() + '/tfrbm/logs/rbms/raw/vals' + "/*.csv" # use your path
path = Path(path)
all_files = glob.glob(path.as_posix())

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

vals = pd.concat(li, axis=0, ignore_index=True)
vals = vals.convert_dtypes()

#%% 

vals['num'] = vals['num'].str.strip('[]').str.split(',')
tmp = vals['num'].apply(pd.Series).astype('float')
tmp = tmp.add_prefix('i')
#vals['num'] = tmp.values.tolist()
vals = vals.join(tmp)
vals = vals.drop(['num'], axis=1)


tmp2 = pd.wide_to_long(vals[vals["Val"]=="meanError"], ["i"], i="Timestamp", j="Epoch")
tmp2 = tmp2.reset_index(level=[0,1])

tmp3 = pd.wide_to_long(vals[vals["Val"]=="pearson"], ["i"], i="Timestamp", j="Epoch")
tmp3 = tmp3.reset_index(level=[0,1])

tmp4 = tmp2.append(tmp3)

tmp4 = tmp4.rename(columns={"i": "Performance"})
tmp4 = tmp4.set_index('Timestamp').join(params.set_index('Timestamp'))

#tmp5 = tmp4[tmp4['rbmName'].str.contains('|'.join(['Xavier','REMERGE','Regularized']))]
#tmp5 = tmp4

import seaborn as sns;
import matplotlib.pyplot as plt

sns.relplot(x="Epoch", y="Performance", hue="rbmName", row = "Val", col= "stddev_par", kind="line", data=tmp4[tmp4['n_visible']==5])

#%% Plotting singular values

#path = r'C:\Users\folde\Desktop\iac\tensorfow-rbm-master\logs\rbms\raw\svd'
#all_files = glob.glob(path + "\*.csv")

path = os.getcwd() + '/tfrbm/logs/rbms/raw/svd' + "/*.csv" # use your path
path = Path(path)
all_files = glob.glob(path.as_posix())

li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

svd_s = pd.concat(li, axis=0, ignore_index=True)
svd_s = svd_s.drop(columns=["u","v"])

tmp = svd_s['s'].str.strip('[]').str.split(' ')
tmp = [[s for s in element if s!='' and s!=' ' and s!=None] for element in tmp]

svd_s['s'] = tmp
tmp = pd.DataFrame(svd_s.s.values.tolist())
tmp = tmp.rename(columns={x:y for x,y in zip(tmp.columns, range(1,len(tmp.columns)+1))})
tmp = tmp.astype('float')
tmp = tmp.add_prefix("a")

svd_s = pd.concat([svd_s, tmp], axis=1)
svd_s = svd_s.drop('s', axis = 1)
svd_s['Index'] = svd_s.index

svd_s = pd.wide_to_long(svd_s, ["a"], i= "Index", j="Value")
svd_s = svd_s.reset_index(level=[0,1])


svd_s = svd_s.set_index('Timestamp').join(params.set_index('Timestamp'))

import seaborn as sns;
import matplotlib.pyplot as plt

sns.relplot(x="Epoch", y="a", hue="rbmName", style="Value", col= "stddev_par", kind="line", data=svd_s)

#%% Plotting REMERGE results

path = os.getcwd() + '/tfrbm/logs/rbms/raw/weight' + "/*_weight.csv" # use your path
path = Path(path)
all_files = glob.glob(path.as_posix())

li = []

import ast
import numpy as np
def from_np_array(array_string):
    array_string = ','.join(array_string.replace('[ ', '[').split())
    try:
        return np.array(ast.literal_eval(array_string))
    except:
        return float('NaN')

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
    
weights = pd.concat(li, axis=0, ignore_index=True)
weights['num'] = weights['num'].apply(from_np_array)

weights_filtered = weights[weights['num'].notnull()]


