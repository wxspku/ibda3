package com.ibda.spark.regression;

import cn.hutool.core.util.ReflectUtil;
import com.ibda.spark.statistics.BasicStatistics;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.shaded.org.apache.commons.beanutils.BeanUtils;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.feature.OneHotEncoder;
import org.apache.spark.ml.feature.OneHotEncoderModel;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Spark回归，支持训练、检测、预测
 */
public abstract class SparkRegression extends BasicStatistics {

    public static final String VECTOR_SUFFIX = "_vector";

    public abstract static class RegressionResult {
        protected static final String[] EXCLUDE_METHODS = new String[]{"getClass","toString","hashCode","wait","notify","notifyAll","asBinary"};
        RegressionResult(PredictionModel model){
            this.model = model;
        }
        /**
         * 训练好的回归模型
         */
        PredictionModel model = null;
        /**
         *带预测结果列的数据集
         */
        Dataset<Row> predictions = null;

        /**
         * 系数列表，List的每个元素为1套完整的系数，多元逻辑回归会返回多套系数
         */
        public List<double[]> coefficientsList = new ArrayList<>();

        /**
         * 训练性能指标，根据回归算法不同而有所区别，部分算法不生成指标结果，从summary读取，或者使用
         * Evaluator（MulticlassClassification）自行运算
         */
        public Map<String,Object> trainingMetrics = new LinkedHashMap<>();

        public PredictionModel getModel() {
            return model;
        }

        public Dataset<Row> getPredictions() {
            return predictions;
        }

        public List<double[]> getCoefficientsList() {
            return coefficientsList;
        }


        public Map<String, Object> getTrainingMetrics() {
            return new HashMap<>(trainingMetrics);
        }


        @Override
        public String toString() {
            return "RegressionResult{" +
                    "model=" + model +
                    ", coefficientsList=" + coefficientsList +
                    ", trainingMetrics=" + trainingMetrics +
                    '}';
        }

        /**
         * 从summary对象提取性能指标
         * @param summary
         * @return
         */
        protected Map<String,Object> buildMetrics(Object summary) {
            Map<String,Object> metrics = new LinkedHashMap<>();
            //通过属性获取性能指标
            try {
                metrics.putAll(BeanUtils.describe(summary));
            } catch (IllegalAccessException e) {
                e.printStackTrace();
            } catch (InvocationTargetException e) {
                e.printStackTrace();
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
            //性能指标
            Method[] publicMethods = ReflectUtil.getPublicMethods(summary.getClass());
            Arrays.stream(publicMethods).forEach(method->{
                if (method.getParameterCount()==0){
                    if (!ArrayUtils.contains(EXCLUDE_METHODS,method.getName())){
                        try {
                            Object performance = method.invoke(summary);
                            if (!(performance instanceof Dataset) &&
                                !(performance instanceof BinaryClassificationMetrics) &&
                                !(performance instanceof SparkSession)){
                                String name = method.getName();
                                if (performance instanceof MulticlassMetrics){
                                    name = "multiclassMetrics";
                                    Matrix confusionMatrix = ((MulticlassMetrics)performance).confusionMatrix();
                                    //混淆矩阵，行为实际值，列为预测值，转换为数组时是先列后行进行转换
                                    metrics.put("confusionMatrix",confusionMatrix);
                                }
                                metrics.put(name, performance);
                            }
                        } catch (IllegalAccessException e) {
                            e.printStackTrace();
                        } catch (InvocationTargetException e) {
                            e.printStackTrace();
                        }
                        catch (Exception e) {
                            e.printStackTrace();
                        }
                    }
                }
            });
            return metrics;
        }

        public void saveModel(String path)  throws IOException {
            if (model instanceof MLWritable){
                ((MLWritable)model).write().overwrite().save(path);
                return;
            }
            throw new RuntimeException("未实现的接口MLWritable.save(path)");
            //if (Model)
        }


        /**
         * 评估模型，返回评估的性能指标，其中predictions条目表示预测结果
         *
         * @param evaluatingData
         * @param modelCols
         * @return
         */
        public abstract Map<String,Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols);


    }

    public SparkRegression(String appName) {
        super(appName);
    }

    /**
     * TODO 可以转换为Pipeline，形成一个统一的转换模型，对训练集、测试集、预测集进行统一的转换，否则会导致错误
     * requirement failed: BLAS.dot(x: Vector, y:Vector) was given Vectors with non-matching sizes: x.size = 11, y.size = 12
     * https://www.codenong.com/58773455/
     * 根据回归字段设置进行数据预处理，处理包括添加所需字段、类型字段编码，其他通用预处理(离群值、缺省值处理)暂不包括
     * spark自动添加预测相关列，无需外部添加
     * @param dataset
     * @param modelCols
     * @return
     */
    public static Dataset<Row> preProcess(Dataset<Row> dataset, ModelColumns modelCols) {
        String[] columns = dataset.columns();
        if (ArrayUtils.contains(columns, modelCols.featuresCol)){
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
                    .setOutputCols(categoryFeatureVectors)
                    .setHandleInvalid("error");
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
        Dataset<Row> result = assembleVector(encoded, features, modelCols.featuresCol);
        return result.select(modelCols.labelCol, modelCols.featuresCol);
    }

    /**
     * @param trainingData
     * @param modelCols
     * @param params    训练参数，根据回归类型及回归算法不同，参数名称也有所不同，具体参见spark文档
     * @return
     */
    public abstract RegressionResult fit(Dataset<Row> trainingData, ModelColumns modelCols, Map<String, Object> params);




    /**
     * 预测
     *
     * @param model
     * @param predictData
     * @param modelCols
     * @return
     */
    public abstract Dataset<Row> predict(PredictionModel model, Dataset<Row> predictData, ModelColumns modelCols);

    /**
     * 预测单条数据
     * @param features
     * @return
     */
    public abstract double predict(final Object features);
}
