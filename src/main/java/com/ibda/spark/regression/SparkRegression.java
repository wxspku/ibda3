package com.ibda.spark.regression;

import com.ibda.spark.SparkAnalysis;
import org.apache.poi.ss.usermodel.Row;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.Dataset;

/**
 * Spark回归，支持训练、检测、预测
 */
public abstract class SparkRegression extends SparkAnalysis {

    public SparkRegression(String appName) {
        super(appName);
    }

    /**
     * 根据回归字段设置进行数据预处理，处理包括添加所需字段、类型字段编码，其他通用预处理(离群值、缺省值处理)暂不包括
     * @param predictData
     * @param modelCols
     * @return
     */
    public Dataset<Row> preTransform(Dataset<Row> predictData,ModelColumns modelCols) {
        return null;
    }
    //训练，返回模型，测试，返回性能数据，预测，实现数据结果
    public abstract Model fit(Dataset<Row> trainingData, ModelColumns modelCols,ParamMap params);

    //验证模型
    public abstract Model validate(Dataset<Row> validatingData,ModelColumns modelCols);

    //预测
    public abstract Dataset<Row> transform(Dataset<Row> predictData,ModelColumns modelCols);
}
