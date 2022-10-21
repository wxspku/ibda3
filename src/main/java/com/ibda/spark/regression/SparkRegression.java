package com.ibda.spark.regression;

import com.ibda.spark.SparkAnalysis;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Spark回归，支持训练、检测、预测
 */
public abstract class SparkRegression extends SparkAnalysis {

    protected static final String REGRESSION_FEATURES_VECTOR = "regression_features_vector";
    protected static final String VECTOR_SUFFIX = "_vector";

    public static class RegressionSummary {

    }

    public SparkRegression(String appName) {
        super(appName);
    }

    /**
     * 根据回归字段设置进行数据预处理，处理包括添加所需字段、类型字段编码，其他通用预处理(离群值、缺省值处理)暂不包括
     * spark自动添加预测相关列，无需外部添加
     * @param dataset
     * @param modelCols
     * @return
     */
    public Dataset<Row> preProcess(Dataset<Row> dataset, ModelColumns modelCols) {
        String[] columns = dataset.columns();
        if (ArrayUtils.contains(columns, REGRESSION_FEATURES_VECTOR)){
            return dataset;
        }
        //处理分类列
        Dataset<Row> encoded = dataset;
        String[] categoryFeatureVectors = null;
        String[] categoryFeatures = modelCols.categoryFeatures;
        String[] noneCategoryFeatures = modelCols.noneCategoryFeatures;
        if (categoryFeatures != null) {
            categoryFeatureVectors = new String[categoryFeatures.length];
            Arrays.stream(categoryFeatures).map(item -> item + VECTOR_SUFFIX)
                    .collect(Collectors.toList())
                    .toArray(categoryFeatureVectors);
            OneHotEncoder encoder = new OneHotEncoder()
                    .setInputCols(categoryFeatures)
                    .setOutputCols(categoryFeatureVectors);
            OneHotEncoderModel model = encoder.fit(dataset);
            encoded = model.transform(dataset);
        }
        //合并特征属性
        String[] features = new String[(categoryFeatures == null ? 0 : categoryFeatures.length) +
                (noneCategoryFeatures == null ? 0 : noneCategoryFeatures.length)];
        if (categoryFeatureVectors != null) {
            System.arraycopy(categoryFeatureVectors, 0, features, 0, categoryFeatureVectors.length);
        }
        if (noneCategoryFeatures != null) {
            System.arraycopy(noneCategoryFeatures, 0, features, categoryFeatureVectors.length, noneCategoryFeatures.length);
        }
        //把属性列合并为vector列
        Dataset<Row> result = assembleVector(encoded, features, REGRESSION_FEATURES_VECTOR);
        return result;
    }

    /**
     * @param trainingData
     * @param modelCols
     * @param params    训练参数，根据回归类型不同，参数名称也有所不同
     * @return
     */
    public abstract PredictionModel fit(Dataset<Row> trainingData, ModelColumns modelCols, Map<String, Object> params);


    /**
     * 评估模型
     *
     * @param evaluatingData
     * @param modelCols
     * @param model
     * @return
     */
    public abstract RegressionSummary evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols, PredictionModel model);

    /**
     * 预测
     *
     * @param predictData
     * @param modelCols
     * @param model
     * @return
     */
    public abstract Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PredictionModel model);
}
