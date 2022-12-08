package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;

import java.io.IOException;

public class LinearRegressionTest extends SparkRegressionTest<LinearRegression, LinearRegressionModel> {
    @Override
    public void initTrainingParams() {
        trainingParams.put("maxIter", 500);
        trainingParams.put("tol", 1E-20);
        trainingParams.put("regParam", 0.4);
        trainingParams.put("elasticNetParam", 0.1);
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
    public void test01LearningEvaluatingPredicting() throws IOException{
        super.test01LearningEvaluatingPredicting();
    }

    @Override
    protected void initTuningGrid() {
        tuningParamGrid.put("regParam",new Double[]{0.3d,0.4d,0.5d,0.6d});
        tuningParamGrid.put("elasticNetParam",new Double[]{0d,0.1d,0.15d,0.2d,0.25d,0.3d});
    }

    @Override
    public void testValidationSplitTuning() throws IOException {
        testTuning(false, RegressionEvaluator.class);
    }

    @Override
    public void testCrossValidationTuning() throws IOException {
        testTuning(true,RegressionEvaluator.class);
    }
}