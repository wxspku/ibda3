package com.ibda.spark.regression;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Map;

public class LogicRegression extends SparkRegression{

    public LogicRegression(String appName) {
        super(appName);
    }

    @Override
    public RegressionHyperModel fit(Dataset<Row> trainingData, ModelColumns modelCols, PipelineModel preProcessModel, Map<String, Object> params) {
        //预处理
        Dataset<Row> training = modelCols.transform(trainingData,preProcessModel);
        LogisticRegression lr = new LogisticRegression()
                .setFeaturesCol(modelCols.featuresCol)
                .setLabelCol(modelCols.labelCol)
                .setPredictionCol(modelCols.predictCol)
                .setProbabilityCol(modelCols.probabilityCol);

        ParamMap paramMap = buildParams(lr.uid(),params);
        // Fit the model
        LogisticRegressionModel lrModel = lr.fit(training,paramMap);
        LogicRegressionHyperModel result = new LogicRegressionHyperModel(lrModel,preProcessModel,modelCols);

        return result;
    }


    @Override
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel, PredictionModel model) {
        LogicRegressionHyperModel hyperModel = new LogicRegressionHyperModel((LogisticRegressionModel) model, preProcessModel, modelCols);
        return hyperModel.predict(predictData);
    }

    @Override
    public double predict(Object features) {
        return 0;
    }
}
