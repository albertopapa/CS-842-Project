#!/usr/bin/env python
# coding: utf-8

# In[11]:


import sys
import math
import numpy as np
data = [int(x) for x in sys.stdin.readlines()[0].rstrip().split()][0]
# put your python code here
plot1 = [1,1]
plot2 = [5,4]
euclidean_distance = math.sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
print(euclidean_distance) # Change this


# In[4]:


import math


# In[5]:



plot1 = [1,3]
plot2 = [2,5]
euclidean_distance = math.sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
euclidean_distance


# In[7]:


import sys
from sklearn.metrics.pairwise import euclidean_distances
data = [int(x) for x in sys.stdin.readlines()]
X = [[0, 1], [1, 1]]
                        # distance between rows of X
euclidean_distances(X, X)
print (euclidean_distances(X, X))


# In[8]:


import sys
data = [int(x) for x in sys.stdin.readlines()]
# put your python code here
a,b,m,n = map(int,input().split())
euclidean_distance= math.sqrt((a-n)**2+(b-m)**2)
print(euclidean_distance) # Change this


# In[2]:


import numpy as np


# In[3]:


np.array([1,4,2,5,3], dtype="float32")


# In[5]:


np.array([range(i,i+3) for i in [2,4,6]])


# In[15]:


import sys
import numpy as np
data = [int(x) for x in sys.stdin.readlines()[0].rstrip().split()][0]
np.random.seed(1)
# put your python code here
n=40
np.random.random ((n,2))


# In[24]:


import sys
import numpy as np
np.random.seed(1)
# put your python code here

x=np.array ([40,2])
print (x.shape)


# In[31]:


import sys
import numpy as np
data = [int(x) for x in sys.stdin.readlines()[0].rstrip().split()][0]
np.random.seed(1)
# put your python code here
x=np.random.randint(2, size=(data,2))
print ("x shape", x.shape)


# In[33]:


x3 = np.random.randint(10, size=(3, 4, 5))
x3


# In[38]:




x3.nbytes


# In[39]:


x3.itemsize


# In[40]:


x3.dtype


# In[44]:


x=np.arange(10)
x


# In[47]:


x[5::-1]


# In[48]:


x[::-1]


# In[50]:


x[::-3]


# In[51]:


x[4:-2]


# In[53]:


x[:-4]


# In[56]:


x2=np.random.randint(10,size=(3,4))
x2


# In[66]:


x2[2,-1]=30
x2


# In[5]:


grid=np.arange(1,10)
grid


# In[9]:


grid=np.arange(1,10).reshape((3,3))
grid


# In[ ]:




