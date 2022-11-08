package com.ibda.spark.regression;

import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;

public class DecisionTreeRegressionTest extends SparkRegressionTest<DecisionTreeRegressor, DecisionTreeRegressionModel> {

    @Override
    public void initTrainingParams() {
        trainingParams.put("impurity", "variance");
        trainingParams.put("minInfoGain", 0.0d);
        trainingParams.put("maxDepth", 10); //default 5
        trainingParams.put("maxBins", 1024);  //default 32 必须是2的整数幂
        trainingParams.put("minInstancesPerNode", 10); //default 1


    }

    @Override
    protected void loadTest02Data() {
        super.loadTest02Data();
        trainingParams.put("maxBins", 64);
    }
}
