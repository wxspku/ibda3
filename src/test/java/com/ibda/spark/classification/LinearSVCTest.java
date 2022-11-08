package com.ibda.spark.classification;

import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;

public class LinearSVCTest extends SparkClassificationTest<LinearSVC, LinearSVCModel> {

    @Override
    public void initTrainingParams() {
        trainingParams.put("aggregationDepth", 3); //default 2
        trainingParams.put("regParam", 0.01);
        trainingParams.put("maxIter", 100);
        trainingParams.put("tol", 1.0E-8); //default 1
    }
}
