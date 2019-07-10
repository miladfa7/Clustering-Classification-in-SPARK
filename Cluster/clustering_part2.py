#!/usr/bin/env python
# coding: utf-8

# In[3]:


from pyspark.sql import SQLContext
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
import utils
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:





# In[5]:


sqlContext = SQLContext(sc)
dataset = sqlContext.read.load('/media/milad/Linux/bigdata/minute_weather.csv', 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')


# In[6]:


dataset.count()


# In[10]:


filtere_dataset = dataset.filter((dataset.rowID % 10) == 0)
filtere_dataset.count()


# In[11]:


filtere_dataset.describe().toPandas().transpose()


# In[12]:


filtere_dataset.filter(filtere_dataset.rain_accumulation == 0.0).count() 


# In[13]:


filtere_dataset.filter(filtere_dataset.rain_duration == 0.0).count() 


# In[14]:


drop_data = filtere_dataset.drop('rain_accumulation').drop('rain_duration').drop('hpwren_timestamp')


# In[15]:


before = drop_data.count()
drop_data = drop_data.na.drop()
after = drop_data.count()
before - after


# In[16]:


drop_data.columns


# In[18]:


featuresUsed = ['air_pressure', 'air_temp', 'avg_wind_direction', 'avg_wind_speed', 'max_wind_direction', 
        'max_wind_speed','relative_humidity']
assembler = VectorAssembler(inputCols=featuresUsed, outputCol="features_unscaled")
assembled = assembler.transform(drop_data)


# In[21]:


scale = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
scale_model = scale.fit(assembled)
scale_data = scale_model.transform(assembled)


# In[23]:


scale_data = scale_data.select("features", "rowID")

elbow_set = scale_data.filter((scale_data.rowID % 3) == 0).select("features")
#elbow_set.persist()


# In[25]:


clusters = range(2,31)
wsseList = utils.elbow(elbow_set, clusters)


# In[26]:


utils.elbow_plot(wsseList, clusters)


# In[31]:


scale_data = scale_data.select("features")
scale_data.persist()


# In[32]:


kmeans = KMeans(k=12, seed=1)
model = kmeans.fit(scale_data)
transformed = model.transform(scale_data)


# In[33]:


centers = model.clusterCenters()
centers

