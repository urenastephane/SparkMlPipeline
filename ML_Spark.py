import time
start_time = time.time()

from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import *
from coalesce import coalesce
from pyspark.sql.functions import col
from pyspark.ml.feature import *
from pyspark.ml.regression import RandomForestRegressor, DecisionTreeRegressor
from pyspark.ml import *
import time
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.storagelevel import StorageLevel

import sys                                                                                                                                                        
reload(sys)                                                                                                                                                       
sys.setdefaultencoding('utf8') 

print("------ CONFIG -------")

# CONF CLUSTER
conf=SparkConf()\
.set('spark.network.timeout','5000000s')\
.set('spark.executor.heartbeatInterval','4500000s')

sc = SparkContext(conf=conf)
sc.setCheckpointDir("checkpointdir2")
sqlContext=SQLContext(sc)

print("------ reading data -------")
data=sqlContext.read.parquet("data.parquet")

print(sc.uiWebUrl)
print("------ train test split --------")
(trainingData,testData)=data.randomSplit([0.7,0.3])

print("--------- Indexing categorical values for OHE------------")

codeIndexer = StringIndexer(inputCol='code', outputCol='code_index').setHandleInvalid("keep")
lieuIndexer = StringIndexer(inputCol='lieu', outputCol='lieu_index').setHandleInvalid("keep")

#putting indexed data into a OHE for a better modelling approach
ohe = OneHotEncoderEstimator(inputCols=['lieu_index','code_index'],outputCols=['lieu_category','code_category']).setHandleInvalid("keep")

#Here i combine numerical features into a vector, rescale them between 0 and 1 and then merge the new scaled numerical features with features from the OHE
numericalAssembler=VectorAssembler(inputCols=['time'],outputCol='numerical_features')
numericalScaler=MinMaxScaler(inputCol='numerical_features',outputCol='scaled_numerical_features')
assembler= VectorAssembler(inputCols=['lieu_category','code_category','scaled_numerical_features'],outputCol='features')
labelCol="label"

print("--------------- Random Forest Model definition ----------------")
rf=RandomForestRegressor(labelCol=labelCol, featuresCol='features', predictionCol='prediction', maxBins=100, numTrees=100, maxDepth=11, subsamplingRate=0.1)

print("------------- transformer pipeline --------------")
transformerStages=[codeIndexer,lieuIndexer,ohe,numericalAssembler,numericalScaler,assembler]
transformerPipeline=Pipeline(stages=transformerStages)
transformer=transformerPipeline.fit(trainingData)
transformedTrainingData=transformer.transform(trainingData).checkpoint()
trainingData.unpersist()

print("--------------- TRANSFORMER serialization ----------------")
transformer.write().overwrite().save("TransformerTestCluster")

print("------------ REGRESSION MODEL ---------------")
modelPipeline=Pipeline(stages=[rf])
model=modelPipeline.fit(transformedTrainingData)
model.write().overwrite().save("ModelTestCluster")

print("------------- READING SERIALIZED TRANSFORMER AND MODEL --------------")
rfTransformerPipeline=PipelineModel.load("TransformerTestCluster")
rfModel=PipelineModel.load("ModelTestCluster")
 
print("---------- Prediction on Test Data -------------")
transformedTestData = rfTransformerPipeline.transform(testData)
predictions=rfModel.transform(transformedTestData)


print("---------- MODEL EVALUATION ------------")
RMSE=RegressionEvaluator(predictionCol="prediction",labelCol=labelCol,metricName="rmse")

R2=RegressionEvaluator(predictionCol="prediction",labelCol=labelCol,metricName="r2")

MAE=RegressionEvaluator(predictionCol="prediction",labelCol=labelCol,metricName="mae")

# Calculation of 3 metrics to check performances of the regression approach (you can change the RF with other regression models)
# see other regression algorithms here: https://spark.apache.org/docs/2.2.0/ml-classification-regression.html#regression
errorRMSE=RMSE.evaluate(predictions)
errorR2=R2.evaluate(predictions)
errorMAE=MAE.evaluate(predictions)


print("-------- RMSE ---------")
print(errorRMSE)
print("-------- R2 ---------")
print(errorR2)
print("-------- MAE ---------")
print(errorMAE)

#check time into the terminal (but can be checked with tools from the cluster)
print("--- %s seconds ---" % (time.time() - start_time))
