#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
print('Libraries Imported')


# In[2]:


pip install joblib


# In[46]:


train = pd.read_csv('/Users/bruger/Desktop/Social_Media_Data.csv', header = None)
train.columns = ['MR1','MR2','MR3','MR4','SOC1','SOC2','SOC3','SOC4','SOC5','EXT1','EXT2','EXT3','EXT4','INT1','INT2','INT3','INT4','COM1','COM2','COM3','COM4','TA1','TA2','TA3','TA4','KTC1','KTC2','KTC3','KTC4','KTC5','KTC6','KTC7','KTC8','Class']
print('Shape of the dataset: ' + str(train.shape))
print(train.head())


# In[47]:


# features and target
target = 'Class'
features = ['MR1','MR2','MR3','MR4']


# In[48]:


X = train[features]
y = train[target]


# In[49]:


X


# In[50]:


y


# In[52]:


# model 
model = svm.SVC()
model.fit(X, y)
model.score(X, y)


# In[53]:


import pickle
pickle.dump(model, open('model.pkl', 'wb'))


# In[54]:


data = {  'MR1': 3
             , 'MR2': 2
             , 'MR3': 1
             , 'MR4': 2}


# In[55]:


data.update((x, [y]) for x, y in data.items())


# In[56]:


data


# In[57]:


data_df = pd.DataFrame.from_dict(data)
data_df


# In[58]:


result = model.predict(data_df)


# In[59]:


rtype=type(result)
result


# In[60]:


rtype


# In[61]:


str(result[0])


# In[62]:


# send back to browser
output = {'results': str(result[0])}


# In[63]:


output


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




