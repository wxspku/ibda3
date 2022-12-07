package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
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

    @Override
    public void testValidationSplitTuning() throws IOException {
        testTuning(false, RegressionEvaluator.class);
    }

    @Override
    public void testCrossValidationTuning() throws IOException {
        testTuning(true,RegressionEvaluator.class);
    }

    @Override
    protected void initTuningGrid() {
        tuningParamGrid.put("regParam",new Double[]{0.3d,0.4d,0.5d});
        tuningParamGrid.put("elasticNetParam",new Double[]{0d,0.05d,0.1d});
    }
}