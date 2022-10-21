package com.ibda.spark.regression;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.BinaryLogisticRegressionTrainingSummary;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.functions;

import java.util.Arrays;
import java.util.Map;

public class LogicRegression extends SparkRegression{

    public LogicRegression(String appName) {
        super(appName);
    }

    private ParamMap buildParams(String parent,Map<String, Object> params){
        if ((params == null || params.isEmpty())){
            return ParamMap.empty();
        }
        ParamMap result = new ParamMap();
        params.entrySet().stream().forEach(entry->{
            Param param = new Param(parent,entry.getKey(),null);
            result.put(param,entry.getValue());
        });
        return result;
    }

    @Override
    public PredictionModel fit(Dataset<Row> trainingData, ModelColumns modelCols, Map<String, Object> params) {
        //预处理
        Dataset<Row> training = preProcess(trainingData, modelCols);
        Dataset<Row> abbr = training.select(modelCols.labelCol,REGRESSION_FEATURES_VECTOR);
        LogisticRegression lr = new LogisticRegression()
                .setFeaturesCol(REGRESSION_FEATURES_VECTOR)
                .setLabelCol(modelCols.labelCol)
                .setPredictionCol(modelCols.predictCol)
                .setProbabilityCol(modelCols.probabilityCol);

        ParamMap paramMap = buildParams(lr.uid(),params);
        // Fit the model
        LogisticRegressionModel lrModel = lr.fit(abbr,paramMap);
        // Print the coefficients and intercept for logistic regression
        System.out.println("Coefficients: "
                + lrModel.coefficients() + " Intercept: " + lrModel.intercept());
        // Extract the summary from the returned LogisticRegressionModel instance trained in the earlier
        // example
        BinaryLogisticRegressionTrainingSummary trainingSummary = lrModel.binarySummary();

        System.out.println(
                String.format("accuracy:%1$.2f,precision:%2$s,recall:%3$s,f-score:%4$s,auc:%5$.2f",
                trainingSummary.accuracy(),
                Arrays.toString(trainingSummary.precisionByLabel()),
                Arrays.toString(trainingSummary.recallByLabel()),
                Arrays.toString(trainingSummary.fMeasureByLabel()),
                trainingSummary.areaUnderROC()));
        // Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
        // Obtain the loss per iteration.
        double[] objectiveHistory = trainingSummary.objectiveHistory();
        for (double lossPerIteration : objectiveHistory) {
            System.out.println(lossPerIteration);
        }
        Dataset<Row> predictions = trainingSummary.predictions();
        predictions.show();

        // Get the threshold corresponding to the maximum F-Measure and rerun LogisticRegression with
        // this selected threshold.
        Dataset<Row> fMeasure = trainingSummary.fMeasureByThreshold();
        double maxFMeasure = fMeasure.select(functions.max("F-Measure")).head().getDouble(0);
        double bestThreshold = fMeasure.where(fMeasure.col("F-Measure").equalTo(maxFMeasure))
                .select("threshold").head().getDouble(0);
        lrModel.setThreshold(bestThreshold);
        return lrModel;
    }

    @Override
    public RegressionSummary evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols, PredictionModel model) {
        return null;
    }

    @Override
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PredictionModel model) {
        return null;
    }
}
