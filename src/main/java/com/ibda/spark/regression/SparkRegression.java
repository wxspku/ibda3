package com.ibda.spark.regression;

import com.ibda.spark.statistics.BasicStatistics;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.*;

/**
 * Spark回归，支持训练、检测、预测
 */
public abstract class SparkRegression extends BasicStatistics {

    public SparkRegression(String appName) {
        super(appName);
    }

    /**
     * 训练回归模型，同时使用训练集数据训练数据预处理模型
     * @param trainingData
     * @param modelCols
     * @param params
     * @return
     */
    public RegressionHyperModel fit(Dataset<Row> trainingData, ModelColumns modelCols, Map<String, Object> params){
        PipelineModel preProcessModel = modelCols.fit(trainingData);
        return fit(trainingData, modelCols,preProcessModel,params);
    }

    protected ParamMap buildParams(String parent, Map<String, Object> params){
        if ((params == null || params.isEmpty())){
            return ParamMap.empty();
        }
        ParamMap paramMap = new ParamMap();
        params.entrySet().stream().forEach(entry->{
            Param param = new Param(parent,entry.getKey(),null);
            paramMap.put(param,entry.getValue());
        });
        return paramMap;
    }

    /**
     * @param trainingData      原始训练集
     * @param modelCols         模型分列设置
     * @param preProcessModel   数据预处理模型，需要先进行预训练，使用ModelColumns.fit方法进行训练
     * @param params    训练参数，根据回归类型及回归算法不同，参数名称也有所不同，具体参见spark文档
     * @return
     */
    public abstract RegressionHyperModel fit(Dataset<Row> trainingData, ModelColumns modelCols, PipelineModel preProcessModel, Map<String, Object> params);

    /**
     * 预测
     *
     * @param predictData
     * @param modelCols
     * @param preProcessModel
     * @param model
     * @return
     */
    public abstract Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel, PredictionModel model);

    /**
     * 预测单条数据
     * @param features
     * @return
     */
    public abstract double predict(final Object features);
}
