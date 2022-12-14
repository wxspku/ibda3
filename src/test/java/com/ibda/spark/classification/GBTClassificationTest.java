package com.ibda.spark.classification;

import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.classification.GBTClassifier;

/**
 * GBTClassifier was given dataset with invalid label 2.0.  Labels must be in {0,1}
 */
public class GBTClassificationTest extends SparkClassificationTest<GBTClassifier, GBTClassificationModel> {

    @Override
    public void initTrainingParams() {
        trainingParams.put("numTrees", 100d); //default 50
        trainingParams.put("impurity", "variance");
        trainingParams.put("minInstancesPerNode", 10); //default 1
        trainingParams.put("minInfoGain", 0.001d);
        trainingParams.put("maxDepth", 10); //default 5
        trainingParams.put("maxBins", 64);  //default 32 必须是2的整数幂

        trainingParams.put("bootstrap", true);
        trainingParams.put("subsamplingRate", 0.7d);
        trainingParams.put("maxIter", 50);
        trainingParams.put("featureSubsetStrategy", "all");
        trainingParams.put("stepSize", 0.05d); //default 0.1
    }
}
