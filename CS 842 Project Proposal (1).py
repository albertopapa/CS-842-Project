#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


# In[2]:


pip install joblib


# In[3]:


train = pd.read_csv('/Users/bruger/Desktop/Social_Media_Data.csv', header = None)
train.columns = ['MR1','MR2','MR3','MR4','SOC1','SOC2','SOC3','SOC4','SOC5','EXT1','EXT2','EXT3','EXT4','INT1','INT2','INT3','INT4','COM1','COM2','COM3','COM4','TA1','TA2','TA3','TA4','KTC1','KTC2','KTC3','KTC4','KTC5','KTC6','KTC7','KTC8','Class']
print('Shape of the dataset: ' + str(train.shape))
print(train.head())


# In[4]:


# features and target
target = 'Class'
features = ['MR1','MR2','MR3','MR4']


# In[5]:


X = train[features]
y = train[target]


# In[6]:


X


# In[7]:


y


# In[8]:


# model 
model = svm.SVC()
model.fit(X, y)
model.score(X, y)


# In[9]:


import pickle
pickle.dump(model, open('model.pkl', 'wb'))


# In[10]:


data = {  'MR1': 3
             , 'MR2': 2
             , 'MR3': 1
             , 'MR4': 2}


# In[11]:


data.update((x, [y]) for x, y in data.items())


# In[12]:


data


# In[13]:


data_df = pd.DataFrame.from_dict(data)
data_df


# In[14]:


result = model.predict(data_df)


# In[15]:


rtype=type(result)
result


# In[16]:


rtype


# In[17]:


str(result[0])


# In[18]:


# send back to browser
output = {'results': str(result[0])}


# In[19]:


output

