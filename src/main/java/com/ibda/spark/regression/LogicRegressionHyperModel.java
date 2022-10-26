package com.ibda.spark.regression;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Map;

/**
 * 逻辑回归结果
 */
public class LogicRegressionHyperModel extends RegressionHyperModel<LogisticRegressionModel> {

    /**
     * 从模型文件加载
     *
     * @param path
     * @return
     */
    public static LogicRegressionHyperModel loadFromModelFile(String path) {
        LogisticRegressionModel model = LogisticRegressionModel.load(path);
        return new LogicRegressionHyperModel(model);
    }

    LogicRegressionHyperModel(LogisticRegressionModel model) {
        this(model, null);
    }

    LogicRegressionHyperModel(LogisticRegressionModel model, PipelineModel preProcessModel) {
        this(model, preProcessModel, null);
    }

    LogicRegressionHyperModel(LogisticRegressionModel model, PipelineModel preProcessModel, ModelColumns modelColumns) {
        super(model, preProcessModel, modelColumns);
        initCoefficients(model);
    }

    @Override
    public Map<String, Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols, PipelineModel preProcessModel) {
        Dataset<Row> testing = modelCols.transform(evaluatingData, preProcessModel);
        LogisticRegressionSummary summary = ((LogisticRegressionModel) model).evaluate(testing);
        Map<String, Object> metrics = this.buildMetrics(summary);
        metrics.put("predictions", summary.predictions());
        return metrics;
    }

    @Override
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel) {
        Dataset<Row> predicting = modelCols.transform(predictData, preProcessModel);
        Dataset<Row> result = model.transform(predicting);
        return result;
    }
}
