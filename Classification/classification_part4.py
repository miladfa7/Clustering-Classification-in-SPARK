#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.sql import SQLContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier,NaiveBayes
from pyspark.ml.feature import Binarizer,VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics


# In[2]:


sqlContext = SQLContext(sc)
data = sqlContext.read.load('/media/milad/Linux/bigdata/daily_weather.csv', 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')


# In[3]:


featureColumns = ['air_pressure_9am','air_temp_9am','avg_wind_direction_9am','avg_wind_speed_9am',
        'max_wind_direction_9am','max_wind_speed_9am','rain_accumulation_9am',
        'rain_duration_9am']


# In[4]:


data = data.drop('number')


# In[5]:


data = data.na.drop() 


# In[6]:


binarizer = Binarizer(threshold=25.0, inputCol="relative_humidity_3pm", outputCol="label")
binarizedDF = binarizer.transform(data)


# In[7]:


binarizedDF.select("relative_humidity_3pm","label").show(4)


# In[8]:


assembler = VectorAssembler(inputCols=featureColumns, outputCol="features")
assembled = assembler.transform(binarizedDF)


# In[9]:


(train, test) = assembled.randomSplit([0.8,0.2], seed = 13234 )


# In[10]:


train.count(), test.count()


# In[11]:


D_tree = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=5,
                            minInstancesPerNode=20, impurity="gini")


# In[12]:


pipeline = Pipeline(stages=[D_tree])
model = pipeline.fit(train)


# In[13]:


predictions = model.transform(test)


# In[14]:


t = train.collect()


# In[15]:


predictions.select("prediction", "label").show(10)


# In[16]:


evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test set accuracy =  " + str(accuracy*100))


# In[18]:


predictions.select("prediction", "label").write.save(path="/media/milad/Linux/bigdata/predictions.csv",
                                                     format="com.databricks.spark.csv",
                                                     header='true')


# #### Naive bayes

# In[19]:


model_nb = NaiveBayes(smoothing=1.0, modelType="multinomial")


# In[20]:


model_naive = model_nb.fit(train)


# In[21]:


predictions_nb = model_naive.transform(test)


# In[22]:


predictions_nb.select("prediction", "label").show(15)


# In[23]:


evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                              metricName="accuracy")
accuracy = evaluator.evaluate(predictions_nb)
print("Test set accuracy = " + str(accuracy*100))


# In[24]:


predictions_nb.select("prediction", "label").write.save(path="/media/milad/Linux/bigdata/predictions_nb.csv",
                                                     format="com.databricks.spark.csv",
                                                     header='true')


# ### Evaluation

# #### Decision tree

# In[30]:


sqlContext = SQLContext(sc)
predictions = sqlContext.read.load("/media/milad/Linux/bigdata/predictions.csv", 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')


# In[32]:


conf_tree = MulticlassMetrics(predictions.rdd.map(tuple))


# In[35]:


conf_tree.confusionMatrix().toArray().transpose()


# #### Naive bayse

# In[36]:


sqlContext = SQLContext(sc)
predictions_nb = sqlContext.read.load("/media/milad/Linux/bigdata/predictions_nb.csv", 
                          format='com.databricks.spark.csv', 
                          header='true',inferSchema='true')


# In[37]:


conf_tree_nb = MulticlassMetrics(predictions.rdd.map(tuple))


# In[38]:


conf_tree_nb.confusionMatrix().toArray().transpose()


# In[ ]:




