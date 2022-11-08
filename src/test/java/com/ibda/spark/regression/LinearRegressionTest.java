package com.ibda.spark.regression;

import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;

import java.io.IOException;

public class LinearRegressionTest extends SparkRegressionTest<LinearRegression, LinearRegressionModel> {
    @Override
    public void initTrainingParams() {
        trainingParams.put("maxIter", 100);
        trainingParams.put("tol", 1E-8);
        trainingParams.put("regParam", 0.2);
        trainingParams.put("elasticNetParam", 0.7);
    }

    @Override
    public void test02MachineLearning() throws IOException {
        //GBTClassificationModel、LinearSVCModel只支持二分类
        super.test02MachineLearning();
    }
}