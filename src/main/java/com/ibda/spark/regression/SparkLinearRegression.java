package com.ibda.spark.regression;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Map;

public class SparkLinearRegression extends SparkRegression{

    public SparkLinearRegression(String appName) {
        super(appName);
    }

    @Override
    public RegressionHyperModel fit(Dataset<Row> trainingData, ModelColumns modelCols, PipelineModel preProcessModel, Map<String, Object> params) {
        return null;
    }

    @Override
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel, PredictionModel model) {
        return null;
    }

    @Override
    public double predict(Object features) {
        return 0;
    }
}
