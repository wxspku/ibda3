package com.ibda.spark.regression;

import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.classification.RandomForestClassificationModel;

public class RandomForestClassificationTest extends SparkMLTest<RandomForestClassifier, RandomForestClassificationModel>{

    @Override
    public void initTrainingParams(){
        trainingParams.put("numTrees",100);
        trainingParams.put("impurity", "gini");
        trainingParams.put("minInstancesPerNode", 10); //default 1
        trainingParams.put("minInfoGain", 0.0d);
        trainingParams.put("maxDepth", 10); //default 5
        trainingParams.put("maxBins", 64);  //default 32 必须是2的整数幂

        trainingParams.put("bootstrap",true);
        trainingParams.put("subsamplingRate",0.5d);
    }
}
