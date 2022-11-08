package com.ibda.spark.regression;

import org.apache.spark.ml.regression.GeneralizedLinearRegression;
import org.apache.spark.ml.regression.GeneralizedLinearRegressionModel;

/**
 * 广义线性回归
 */
public class GLRegressionTest extends SparkRegressionTest<GeneralizedLinearRegression, GeneralizedLinearRegressionModel> {
    @Override
    public void initTrainingParams() {
        trainingParams.put("family","gaussian");
        trainingParams.put("link","identity");
        trainingParams.put("maxIter",100);
        trainingParams.put("regParam",0.3d);
    }
}
