#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.regression import LabeledPoint

rawdata=[
['sunny',85,85,'FALSE',0],
['sunny',80,90,'FALSE',0],
['overcast',83,86,'TRUE',1],
['rainy',70,96,'FALSE',1],
['rainy',68,80,'FALSE',1],
['rainy',65,70,'TRUE',0],
['overcast',64,65,'TRUE',1],
['sunny',72,95,'TRUE',0],
['sunny',69,70,'FALSE',1],
['rainy',75,80,'FALSE',1],
['sunny',75,70,'TRUE',1],
['overcast',72,90,'TRUE',1],
['overcast',81,75,'FALSE',1],
['rainy',71,91,'FALSE',0],
['sunny',84,85,'FALSE',1],
['sunny',81,89,'TRUE',1],
['overcast',84,87,'FALSE',0],
['rainy',72,95,'TRUE',0],
['rainy',70,82,'FALSE',0],
['rainy',63,71,'TRUE',1],
['overcast',65,75,'TRUE',0],
['sunny',74,93,'FALSE',1],
['sunny',70,75,'FALSE',0],
['rainy',80,79,'FALSE',0],
['sunny',80,75,'TRUE',0],
['overcast',75,85,'TRUE',0],
['overcast',85,80,'FALSE',0],
['rainy',75,91,'TRUE',1],

    
['rainy',80,80,'FALSE',1],
['sunny',75,68,'FALSE',0],
['sunny',80,90,'TRUE',0],
['overcast',90,94,'TRUE',1],
['rainy',85,96,'FALSE',1],
['rainy',92,65,'TRUE',1],
['rainy',69,79,'FALSE',0],
['overcast',69,80,'TRUE',1],
['sunny',78,92,'FALSE',0],
['sunny',79,89,'TRUE',1],
['rainy',91,99,'FALSE',1],
['sunny',90,85,'TRUE',1],
['overcast',89,86,'FALSE',1],
['overcast',93,88,'TRUE',1],
['rainy',76,93,'TRUE',0]
]

sc = SparkContext.getOrCreate()
from pyspark import SparkConf, SparkContext



from pyspark.sql import SQLContext,Row
sqlContext = SQLContext(sc)

data_df=sqlContext.createDataFrame(rawdata,
   ['outlook','temp','humid','windy','play'])

#transform categoricals into indicator variables
out2index={'sunny':[1,0,0],'overcast':[0,1,0],'rainy':[0,0,1]}

#make RDD of labeled vectors
def newrow(dfrow):
    outrow = list(out2index.get((dfrow[0])))  #get dictionary entry for outlook
    outrow.append(dfrow[1])   #temp
    outrow.append(dfrow[2])   #humidity
    if dfrow[3]=='TRUE':      #windy
        outrow.append(1)
    else:
        outrow.append(0)
    return (LabeledPoint(dfrow[4],outrow))

datax_rdd=data_df.rdd.map(newrow)
#


# In[7]:


splits = datax_rdd.randomSplit([0.7, 0.3],1234)


# In[8]:


train = splits[0]
test = splits[1]


# In[9]:


nbmodel = NaiveBayes.train(train)


# In[ ]:





# In[10]:


test_data = test.collect()


# In[11]:


predict_test =[]
for i in range(test.count()):
    predict_test.append(nbmodel.predict(test_data[:][i].features))


# In[12]:


predict_test


# In[13]:


test.collect()


# In[14]:


train_data = train.collect()
predict_train =[]
for i in range(train.count()):
    predict_train.append(nbmodel.predict(train_data[:][i].features))


# In[15]:


conf_mat = [ [0,0],[0,0] ]
for i in range(train.count()):
    conf_mat[int(train_data[:][i].label)][int(predict_train[i])] +=1


# In[16]:


conf_mat


# In[17]:


acuracy = float((conf_mat[0][0] + conf_mat[1][1]) / train.count())


# In[18]:


acuracy * 100


# In[32]:


Misclassification = float((conf_mat[0][1] + conf_mat[1][0]) /  train.count())


# In[33]:


Misclassification * 100


# In[19]:


conf_mat_test = [[0,0],[0,0]]
for j in range(test.count()):
    conf_mat_test[int(test_data[:][j].label)][int(predict_test[j])] +=1


# In[20]:


conf_mat_test


# In[27]:


acuracy = float((conf_mat_test[0][0] + conf_mat_test[1][1]) / test.count())


# In[28]:


acuracy *100


# In[30]:


Misclassification = float((conf_mat_test[0][1] + conf_mat_test[1][0]) / test.count())


# In[31]:


Misclassification * 100


# ### Decision Tree 

# In[36]:


from pyspark.ml.classification import DecisionTreeClassifier


# In[69]:


DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5,
                            minInstancesPerNode=20, impurity='entropy')


# In[70]:


from pyspark.mllib.tree import DecisionTree


# In[71]:


tree_model = DecisionTree.trainClassifier(datax_rdd, numClasses=2,categoricalFeaturesInfo={},
                                          minInstancesPerNode=2 )


# In[ ]:





# In[72]:


predict_test_tree =[]
for i in range(test.count()):
    predict_test_tree.append(tree_model.predict(test_data[:][i].features))


# In[73]:


predict_test_tree


# In[74]:


predict_train_tree = []
for i in range(train.count()):
    predict_train_tree.append(tree_model.predict(train_data[:][i].features))


# In[75]:


conf_mat_tree = [ [0,0],[0,0] ]
for i in range(train.count()):
    conf_mat_tree[int(train_data[:][i].label)][int(predict_train_tree[i])] +=1


# In[76]:


conf_mat_tree


# In[77]:


acuracy = float((conf_mat_tree[0][0] + conf_mat_tree[1][1]) / train.count()) * 100
acuracy


# In[78]:


conf_mat_tree2 = [ [0,0],[0,0] ]
for i in range(test.count()):
    conf_mat_tree2[int(test_data[:][i].label)][int(predict_test_tree[i])] +=1


# In[79]:


conf_mat_tree2


# In[80]:


acuracy = float((conf_mat_tree2[0][0] + conf_mat_tree2[1][1]) / test.count()) * 100
acuracy


# In[ ]:




