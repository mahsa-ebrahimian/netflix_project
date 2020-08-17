#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(2)

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, MaxPooling2D, Conv1D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from sklearn.metrics import classification_report, confusion_matrix

import random
import tensorflow as tf


sns.set(style='white', context='notebook', palette='deep')

from datetime import datetime

# datetime object containing current date and time
start_time = datetime.now()
import gc


# In[2]:


# Movie lens 100 K dataset
import os
names = ['user_id', 'item_id', 'rating', 'timestamp']

movie_db = pd.read_csv('netflix_project/netflix_sample_complete.csv',skiprows=1, names = ['user_id','rating','date','item_id'],header = None)
movie_db=movie_db[['user_id','item_id','rating','date']]


# In[5]:


item_info=movie_db.groupby('item_id', as_index=False).rating.agg({np.mean, np.var})
item_info.reset_index(level=0, inplace=True)
target_size=100
selected_item_size=20
performance=pd.DataFrame(columns=['attack_model','attack size','filler size','test_or_train','accuracy','recall','F1'])

x=0.01 ## percentage of fillers for AOP attack
avg_rating=movie_db['rating'].mean() 
std_rating=movie_db['rating'].std() 
#selected_items to be chosen from highly popular items , items that are rated by most of users ( highest number of ratings)
#item_info1=
#movie_db.groupby('item_id', as_index=False)['rating'].mean()
attack_date=random.sample(list(movie_db['date'].unique()),1)
print('attack day is: ',attack_date)


# In[6]:



attack_model='AOP'#attack_i
filler_size= 0.01#filler_i # from 1% to 10% which is 1% of items excluding target items
attack_size= 0.1
fake_db = pd.DataFrame( columns=['user_id', 'item_id', 'rating', 'date'])
fake_profile=[]
filler_db=pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'date'])
target_db=pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'date'])
selected_db=pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'date'])
# defining fake profiles
for i in range (round(attack_size*movie_db['user_id'].nunique())):
    fake_profile.append(movie_db['user_id'].max()+(i+1))
print('number of fake profiles generated is: \n',round(attack_size*movie_db['user_id'].nunique()))

#defining target items
#30 items are choosen randomly out of 1682 items
target_items=random.sample(list(movie_db['item_id'].unique()),target_size)
print('number of target items selected is: \n',target_size)


#defining filler items
fillers = [x for x in list(movie_db['item_id']) if x not in target_items]
#randomly choosed filler items based on filler size
filler_items=random.sample(fillers,round(filler_size*len(set(fillers))))




#defining selected items
selected=movie_db.groupby('item_id', as_index=False)['rating'].count()
selected=selected[selected['rating']>round(0.4*movie_db['user_id'].nunique())]
selected_items=[x for x in selected['item_id'] if x not in target_items if x not in filler_items]
print('number of selected items  is: \n',len(set(selected_items)))

#insert ratings for filler items
filler_db.drop(filler_db.index, inplace=True)
for i in filler_items:
    if attack_model=='random':
        filler_db=filler_db.append(pd.DataFrame({'user_id': random.sample(fake_profile,1), 'item_id': i ,'rating':round(np.random.normal(avg_rating, std_rating),2),'date':[attack_date[0]]}), ignore_index=True)
    if attack_model=='average':
        filler_db=filler_db.append(pd.DataFrame({'user_id': random.sample(fake_profile,1), 'item_id': i ,'rating':item_info.loc[item_info['item_id']==i,'mean'],'date':[attack_date[0]]}), ignore_index=True)
    if attack_model=='bandwagon':
        filler_db=filler_db.append(pd.DataFrame({'user_id': random.sample(fake_profile,1), 'item_id': i ,'rating':round(np.random.normal(avg_rating, std_rating),2),'date':[attack_date[0]]}), ignore_index=True)


#insert ratings for target items
target_db.drop(target_db.index, inplace=True)
for i in target_items:
    #print(random.sample(fake_profile,1),i,5)
    target_db=target_db.append(pd.DataFrame({'user_id':random.sample(fake_profile,1),'item_id':i,'rating':5,'date':[attack_date[0]]}),ignore_index=True)

#insert rating for selected items
if attack_model=='bandwagon':
    for i in selected_items:
        selected_db=selected_db.append(pd.DataFrame({'user_id': random.sample(fake_profile,1), 'item_id': i ,'rating':5,'date':[attack_date[0]]}), ignore_index=True)


#fillers selection for AOP attacks

AOP_filler_db=pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'date'])
AOP_fillers=movie_db.groupby('item_id', as_index=False)['rating'].count()
AOP_fillers=AOP_fillers.sort_values('rating',ascending=False)
AOP_fillers=AOP_fillers.head(int(len(AOP_fillers)*(x)))
AOP_fillers=[x for x in AOP_fillers['item_id']]


if attack_model=='AOP':
    for i in AOP_fillers:
        AOP_filler_db=AOP_filler_db.append(pd.DataFrame({'user_id':random.sample(fake_profile,1), 'item_id': i ,'rating':np.random.normal(item_info.loc[item_info['item_id']==i,'mean'],item_info.loc[item_info['item_id']==i,'var']),'date':[attack_date[0]]}), ignore_index=True)
    filler_db=AOP_filler_db

if attack_model=='AOP':
    print('number of filler items selected for AOP attack is: \n',round(len(set(AOP_fillers))))
else:
    print('number of filler items selected is: \n',round(filler_size*len(set(fillers))))

# to first create a complete dataframe and then reshape to array
injected_db=pd.DataFrame()
injected_db=injected_db.append(target_db)
injected_db=injected_db.append(filler_db)
if attack_model=='bandwagon':
    injected_db=injected_db.append(selected_db)

movie_db=movie_db.append(injected_db,sort=True)
movie_db.fillna(0, inplace=True)

iix_n = pd.MultiIndex.from_product([np.unique(movie_db.user_id), np.unique(movie_db.date)])
arr = (movie_db.pivot_table('rating', ['user_id', 'date'], 'item_id', aggfunc='sum')
         .reindex(iix_n,copy=False).to_numpy()
         .reshape(movie_db.user_id.nunique(),movie_db.date.nunique(),-1))
arr_y=[1  for i in arr[:2000,:,:]]
arr_y2=[0  for i in arr[2000:,:,:] ]########??????????? how to give target to them????
arr_y3=arr_y+arr_y2

inds = np.where(np.isnan(arr))
#Place column means in the indices. Align the arrays using take
arr[inds] = 0
X_train, X_test, Y_train, Y_test = train_test_split(arr,arr_y3,test_size=0.30, random_state=40)


# In[ ]:


end_time = datetime.now()
print("started at: ",start_time)
print("ended at: ",end_time)

