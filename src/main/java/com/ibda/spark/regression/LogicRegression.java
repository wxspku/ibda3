package com.ibda.spark.regression;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.linalg.Matrix;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import scala.collection.Iterator;

import java.util.Map;

public class LogicRegression extends SparkRegression{
    /**
     * 逻辑回归结果
     */
    public static class LogicRegressionHyperModel extends RegressionHyperModel {

        /**
         * 从模型文件加载
         * @param path
         * @return
         */
        public static LogicRegressionHyperModel loadFromModelFile(String path) {
            LogisticRegressionModel model = LogisticRegressionModel.load(path);
            return new LogicRegressionHyperModel(model);
        }

        LogicRegressionHyperModel(LogisticRegressionModel model) {
            this(model,null);
        }

        LogicRegressionHyperModel(LogisticRegressionModel model, PipelineModel preProcessModel) {
            this(model, preProcessModel,null);
        }

        LogicRegressionHyperModel(LogisticRegressionModel model, PipelineModel preProcessModel, ModelColumns modelColumns){
            super(model,preProcessModel,modelColumns);
            //回归系数
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
            else{ //根据Evaluator计算

            }
        }

        @Override
        public Map<String, Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols,PipelineModel preProcessModel) {
            Dataset<Row> testing = modelCols.transform(evaluatingData,preProcessModel);
            LogisticRegressionSummary summary = ((LogisticRegressionModel) model).evaluate(testing);
            Map<String, Object> metrics = this.buildMetrics(summary);
            metrics.put("predictions",summary.predictions());
            return metrics;
        }

        @Override
        public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel) {
            Dataset<Row> predicting = modelCols.transform(predictData,preProcessModel);
            Dataset<Row> result = model.transform(predicting);
            return result;
        }
    }

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
