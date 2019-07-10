#!/usr/bin/env python
# coding: utf-8

# In[65]:


import numpy as np
from pyspark.mllib.random import RandomRDDs
from pyspark.mllib.clustering import KMeans, KMeansModel


# In[66]:


c1_v = RandomRDDs.normalVectorRDD(sc, 20, 2,numPartitions=2, seed=1).map(lambda v:np.add([1,5],v))


# In[67]:


c1_v.stats()


# In[68]:


c2_v = RandomRDDs.normalVectorRDD(sc, 20, 2,numPartitions=2, seed=1).map(lambda v:np.add([5,1],v))


# In[69]:


c2_v.stats()


# In[70]:


c3_v = RandomRDDs.normalVectorRDD(sc, 20, 2,numPartitions=2, seed=1).map(lambda v:np.add([4,6],v))


# In[71]:


c12 = c1_v.union(c2_v)


# In[72]:


mydata = c12.union(c3_v)


# ### K = 1

# In[73]:


my_kmModel = KMeans.train(mydata, k=1,maxIterations=10, runs=1,initializationMode='k-means||')


# In[74]:


my_kmModel.computeCost(mydata)


# In[53]:


my_kmModel.clusterCenters


# In[58]:


my_kmModel = KMeans.train(mydata, k=3,maxIterations=10, runs=1,initializationMode='k-means||')


# In[59]:


my_kmModel.computeCost(mydata)


# In[60]:


my_kmModel.clusterCenters


# In[61]:


mydata.stats()


# In[62]:


my_kmModel = KMeans.train(mydata, k=4,maxIterations=10, runs=1,initializationMode='k-means||')


# In[63]:


my_kmModel.computeCost(mydata)


# In[64]:


my_kmModel.clusterCenters


# In[ ]:




