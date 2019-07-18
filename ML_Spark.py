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
data=sqlContext.read.parquet("6.1.2.FinalSeq2J2EvtsToKeep10perCent.parquet")

print(sc.uiWebUrl)
print("------ train test split --------")
(trainingData,testData)=data.randomSplit([0.7,0.3])

print("--------- factorization of categorical values ------------")

codeIndexer = StringIndexer(inputCol='code_evt', outputCol='code_evt_index').setHandleInvalid("keep")
firstCodeIndexer = StringIndexer(inputCol='first_code_evt', outputCol='first_code_evt_index').setHandleInvalid("keep")
lieuIndexer = StringIndexer(inputCol='lieu_evt', outputCol='lieu_evt_index').setHandleInvalid("keep")
firstLieuIndexer = StringIndexer(inputCol='first_lieu_evt', outputCol='first_lieu_evt_index').setHandleInvalid("keep")
lastLieuIndexer = StringIndexer(inputCol='lieu_dernier_evt', outputCol='last_lieu_evt_index').setHandleInvalid("keep")
codeServiceIndexer = StringIndexer(inputCol='code_service', outputCol='code_service_index').setHandleInvalid("keep")

ohe = OneHotEncoderEstimator(inputCols=['dow','first_dow','seq','code_evt_index','first_code_evt_index','lieu_evt_index','first_lieu_evt_index','last_lieu_evt_index','code_service_index'],outputCols=['dow_category','first_dow_category','seq_category','code_evt_category','first_code_evt_category','lieu_evt_category','first_lieu_evt_category','last_lieu_evt_category','code_service_category']).setHandleInvalid("keep")

numericalAssembler=VectorAssembler(inputCols=['time','timestamp_delta','contractual_time_difference'],outputCol='numerical_features')
numericalScaler=MinMaxScaler(inputCol='numerical_features',outputCol='scaled_numerical_features')
assembler= VectorAssembler(inputCols=['dow_category','first_dow_category','seq_category','code_evt_category','lieu_evt_category','first_lieu_evt_category','last_lieu_evt_category','code_service_category','scaled_numerical_features'],outputCol='features')
labelCol="label"

print("--------------- Random Forest Model definition ----------------")
rf=RandomForestRegressor(labelCol=labelCol, featuresCol='features', predictionCol='prediction', maxBins=100, numTrees=100, maxDepth=11, subsamplingRate=0.1)

print("------------- transformer pipeline --------------")
transformerStages=[codeIndexer,firstCodeIndexer,lieuIndexer,firstLieuIndexer,lastLieuIndexer,codeServiceIndexer,ohe,numericalAssembler,numericalScaler,assembler]
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


errorRMSE=RMSE.evaluate(predictions)
errorR2=R2.evaluate(predictions)
errorMAE=MAE.evaluate(predictions)


print("-------- RMSE ---------")
print(errorRMSE)
print("-------- R2 ---------")
print(errorR2)
print("-------- MAE ---------")
print(errorMAE)

print("--- %s seconds ---" % (time.time() - start_time))
