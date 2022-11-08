package com.ibda.spark.regression;

import org.apache.spark.ml.regression.RandomForestRegressionModel;
import org.apache.spark.ml.regression.RandomForestRegressor;

public class RandomForestRegressionTest extends SparkRegressionTest<RandomForestRegressor, RandomForestRegressionModel> {

    @Override
    public void initTrainingParams() {
        trainingParams.put("impurity", "variance");
        trainingParams.put("minInfoGain", 0.0d);
        trainingParams.put("maxDepth", 10); //default 5
        trainingParams.put("maxBins", 1024);  //default 32 必须是2的整数幂
        trainingParams.put("minInstancesPerNode", 10); //default 1

        trainingParams.put("numTrees", 100);
        trainingParams.put("bootstrap", true);
        trainingParams.put("featureSubsetStrategies()", "sqrt");
    }

    @Override
    protected void loadTest02Data() {
        super.loadTest02Data();
        trainingParams.put("maxBins", 64);
    }
}
