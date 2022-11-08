package com.ibda.spark.classification;

import org.apache.spark.ml.classification.RandomForestClassificationModel;
import org.apache.spark.ml.classification.RandomForestClassifier;

public class RandomForestClassificationTest extends SparkClassificationTest<RandomForestClassifier, RandomForestClassificationModel> {

    @Override
    public void initTrainingParams() {

        trainingParams.put("impurity", "gini");
        trainingParams.put("minInstancesPerNode", 10); //default 1
        trainingParams.put("minInfoGain", 0.0d);
        trainingParams.put("maxDepth", 10); //default 5
        trainingParams.put("maxBins", 64);  //default 32 必须是2的整数幂

        trainingParams.put("numTrees", 100);
        trainingParams.put("bootstrap", true);
        trainingParams.put("subsamplingRate", 0.5d);
    }
}
