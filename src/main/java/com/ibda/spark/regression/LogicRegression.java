package com.ibda.spark.regression;

import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import scala.collection.Iterator;

import java.util.Map;

public class LogicRegression extends SparkRegression{
    public static class LogicRegressionResult extends RegressionResult{

        /**
         * 从模型文件加载
         * @param path
         * @return
         */
        public static LogicRegressionResult loadFromModelFile(String path) {
            LogisticRegressionModel model = LogisticRegressionModel.load(path);
            return new LogicRegressionResult(model);
        }

        LogicRegressionResult(LogisticRegressionModel model) {
            super(model);
            //回归系数 TODO 只考虑了二元回归，需考虑多元回归
            if (model.numClasses() ==2){//二元回归
                double[] coefficients = new double[model.coefficients().size()+1];
                coefficients[0] = model.intercept();
                double[] array = model.coefficients().toArray();
                System.arraycopy(array,0,coefficients,1,array.length);
                coefficientsList.add(coefficients);
            }
            else{//多元回归
                //截距向量
                Vector interceptVector = model.interceptVector();
                double[] intercepts = interceptVector.toArray();
                //系数矩阵，每行为一套系数
                Matrix coefficientMatrix = model.coefficientMatrix();
                Iterator<Vector> rowIter = coefficientMatrix.rowIter();
                //将截距向量和系数矩阵拼接为完整的系数
                int i = 0;
               while (rowIter.hasNext()){
                   double[] row = rowIter.next().toArray();
                   double[] coefficients = new double[row.length+1];
                   coefficients[0] = intercepts[i];
                   System.arraycopy(row,0,coefficients,1,row.length);
                   coefficientsList.add(coefficients);
                   i++;
               }
            }
            if (model.hasSummary()){
                LogisticRegressionTrainingSummary summary = model.summary();
                this.predictions = summary.predictions();
                trainingMetrics.putAll(buildMetrics(summary));
            }
            else{ //根据

            }
        }



        @Override
        public Map<String, Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols) {
            Dataset<Row> testing = LogicRegression.preProcess(evaluatingData, modelCols);
            LogisticRegressionSummary summary = ((LogisticRegressionModel) model).evaluate(testing);
            Map<String, Object> metrics = this.buildMetrics(summary);
            metrics.put("predictions",summary.predictions());
            return metrics;
        }
    }
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
    public RegressionResult fit(Dataset<Row> trainingData, ModelColumns modelCols, Map<String, Object> params) {
        //预处理
        Dataset<Row> training = preProcess(trainingData, modelCols);
        //Dataset<Row> abbr = training.select(modelCols.labelCol, modelCols.featuresCol);
        LogisticRegression lr = new LogisticRegression()
                .setFeaturesCol(modelCols.featuresCol)
                .setLabelCol(modelCols.labelCol)
                .setPredictionCol(modelCols.predictCol)
                .setProbabilityCol(modelCols.probabilityCol);

        ParamMap paramMap = buildParams(lr.uid(),params);
        // Fit the model
        LogisticRegressionModel lrModel = lr.fit(training,paramMap);
        LogicRegressionResult result = new LogicRegressionResult(lrModel);
        return result;
    }


    @Override
    public Dataset<Row> predict(PredictionModel model, Dataset<Row> predictData, ModelColumns modelCols) {
        Dataset<Row> predicting = this.preProcess(predictData,modelCols);
        Dataset<Row> result = model.transform(predicting);
        return result;
    }

    @Override
    public double predict(Object features) {
        return 0;
    }
}
