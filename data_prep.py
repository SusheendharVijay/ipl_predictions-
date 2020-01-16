# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tqdm
pd.set_option('display.max_columns',100)
pd.set_option('display.max_rows',100)
import random
import time 
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt 
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
matches = pd.read_csv('/kaggle/input/ipldata/matches.csv')
deli = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')

# %% [code]
matches.head()

# %% [markdown]
# #----------------------------------------

# %% [code]
def preprocessing_data(deli_1):
        batsman = deli_1.groupby(['batsman'])
        good_batsman = {'batsman':[],'good/bad':[],'target':[]}
        batsman_list = list(deli_1.batsman.unique())
        for name in batsman_list:
            group = batsman.get_group(name)
            good_batsman['batsman'].append(name)
            if group.shape[0]>0:
                good_batsman['good/bad'].append(group.shape[0]>12)
            if group.shape[0]==0:
                good_batsman['good/bad'].append(False)

            if (group.shape[0]>12) and (group.shape[0]<=18):
                good_batsman['target'].append(1)
            else:
                good_batsman['target'].append(0)


        good_batsman  = pd.DataFrame(data=good_batsman,columns=['batsman','good_bad','target']) 
        deli_1 = pd.merge(deli_1,good_batsman,how='left',on=['batsman'])
        deli_1.drop(['is_super_over','wide_runs','bye_runs','legbye_runs','noball_runs','penalty_runs','extra_runs'],inplace=True,axis=1)

        deli_good_1 = deli_1.loc[deli_1['good_bad']].copy()
        good_batsman = deli_good_1.groupby(['batsman']).copy()




        for name in deli_good_1.batsman.unique():
            group = good_batsman.get_group(name)
            group = group.iloc[:12,:]
            train_data['batsman'].append(name)
            train_data['total_runs_12'].append(group.total_runs.sum())
            train_data['batsman_runs_12'].append(group.batsman_runs.sum())
            train_data['mean_12'].append(group.batsman_runs.mean())
            train_data['var_12'].append(group.batsman_runs.var())
            train_data['target'].append(group.target.unique()[0])

# %% [code]
train_data = {'batsman':[],'total_runs_12':[],'batsman_runs_12':[],'mean_12':[],'var_12':[],'target':[]}
for match_num in tqdm.trange(1,637):    
    deli_1 = deli.loc[deli.match_id==match_num]
    preprocessing_data(deli_1)

for match_num in tqdm.trange(7894,7954):    
    deli_1 = deli.loc[deli.match_id==match_num]
    preprocessing_data(deli_1)

for match_num in tqdm.trange(11137,11416):    
    deli_1 = deli.loc[deli.match_id==match_num]
    preprocessing_data(deli_1)
    
    

X_train = pd.DataFrame(train_data)
X_train.head()