# SparkMlPipeline
Complete pipeline developed with pyspark to run ML algorithms through Hadoop Clusters (here a random forest is run without ParamGrid or Cross Validations, but those can be added easily. See example here: https://spark.apache.org/docs/2.2.0/ml-tuning.html#model-selection-aka-hyperparameter-tuning )

This file contains a full ML pipeline ready to use. 
Nevertheless, it is not optimized to automatically integrate any column of a df (provided as a list). For now, you juste manually add String indexers and the pipeline. I automated this job in the repository dealing with Feature Importance calculation with Spark.
