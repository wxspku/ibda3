package com.ibda.spark.regression;

import org.apache.spark.ml.classification.FMClassificationModel;
import org.apache.spark.ml.classification.FMClassifier;

public class FMClassificationTest extends SparkMLTest<FMClassifier, FMClassificationModel> {
    @Override
    public void prepareData() {
        loadBinomialData();

        trainingParams.put("stepSize",0.001d); //default 0.1
        trainingParams.put("regParam",0.001d);
        trainingParams.put("tol",1.0E-8);
        trainingParams.put("maxIter",500);
        trainingParams.put("fitLinear",false);
    }
}
