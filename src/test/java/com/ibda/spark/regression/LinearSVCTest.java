package com.ibda.spark.regression;

import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.LinearSVCModel;

public class LinearSVCTest extends SparkMLTest<LinearSVC, LinearSVCModel> {
    @Override
    public void prepareData() {
        loadBinomialData();
        trainingParams.put("aggregationDepth",3); //default 2
        trainingParams.put("regParam", 0.01);
        trainingParams.put("maxIter",100);
        trainingParams.put("tol", 1.0E-8); //default 1
    }
}
