package com.ibda.spark.regression;

import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;

public class GBTClassificationTest extends SparkMLTest<GBTClassifier, GBTClassificationModel>{

    @Override
    public void initTrainingParams(){
        trainingParams.put("numTrees",100d); //default 50
        trainingParams.put("impurity", "variance");
        trainingParams.put("minInstancesPerNode", 10); //default 1
        trainingParams.put("minInfoGain", 0.001d);
        trainingParams.put("maxDepth", 10); //default 5
        trainingParams.put("maxBins", 64);  //default 32 必须是2的整数幂

        trainingParams.put("bootstrap",true);
        trainingParams.put("subsamplingRate",0.7d);
        trainingParams.put("maxIter",50);
        trainingParams.put("featureSubsetStrategy","all");
        trainingParams.put("stepSize",0.05d); //default 0.1
    }
}
