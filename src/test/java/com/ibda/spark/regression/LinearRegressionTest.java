package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;

import java.io.IOException;

public class LinearRegressionTest extends SparkRegressionTest<LinearRegression, LinearRegressionModel> {
    @Override
    public void initTrainingParams() {
        trainingParams.put("maxIter", 200);
        trainingParams.put("tol", 1E-8);
        trainingParams.put("regParam", 0.2);
        trainingParams.put("elasticNetParam", 0.8);
    }

    @Override
    protected void loadTest02Data() {
        //car,age,gender,inccat,ed,marital
        modelColumns = new ModelColumns(
                new String[]{"age"},
                new String[]{"gender","inccat", "ed", "marital"},
                new String[]{"gender"},
                "car");
        loadDataSet(FilePathUtil.getAbsolutePath("data/car_decision_tree.csv", false), "csv");
    }
}