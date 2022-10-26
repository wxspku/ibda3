package com.ibda.spark.regression;

import cn.hutool.core.util.ReflectUtil;
import com.ibda.spark.statistics.BasicStatistics;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.shaded.org.apache.commons.beanutils.BeanUtils;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
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

/**
 * Spark回归，支持训练、检测、预测
 */
public abstract class SparkRegression extends BasicStatistics {

    public abstract static class RegressionHyperModel {

        public static final String PREDICTIONS_KEY = "predictions";

        protected static final String[] EXCLUDE_METHODS =
                new String[]{"getClass","toString","hashCode","wait","notify","notifyAll","asBinary"};

        public RegressionHyperModel(PredictionModel model, PipelineModel preProcessModel) {
            this(model,preProcessModel,null);
        }

        public RegressionHyperModel(PredictionModel model, PipelineModel preProcessModel,ModelColumns modelColumns) {
            this.model = model;
            this.preProcessModel = preProcessModel;
            this.modelColumns = modelColumns;
        }

        /**
         * 训练好的回归模型
         */
        PredictionModel model = null;

        /**
         * 数据预处理的PipelineModel，在训练回归模型时，同时使用训练数据训练预处理模型
         */
        PipelineModel preProcessModel = null;
        /**
         *带预测结果列的数据集
         */
        Dataset<Row> predictions = null;
        /**
         * 模型列设置
         */
        ModelColumns modelColumns = null;
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

        public PipelineModel getPreProcessModel() {
            return preProcessModel;
        }

        public ModelColumns getModelColumns() {
            return modelColumns;
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
            return "RegressionHyperModel{" +
                    "model=" + model +
                    ", preProcessModel=" + preProcessModel +
                    ", predictions=" + predictions +
                    ", modelColumns=" + modelColumns +
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

        /**
         *
         * @param path
         * @throws IOException
         */
        public void saveModel(String path)  throws IOException {
            if (model instanceof MLWritable){
                ((MLWritable)model).write().overwrite().save(path);
                return;
            }
            throw new RuntimeException("未实现的接口MLWritable.save(path)");
        }

        /**
         * 使用内部的Model、ModelColumns、PipelineModel评估模型，其中predictions条目表示预测结果
         * @param evaluatingData
         * @return
         */
        public Map<String,Object> evaluate(Dataset<Row> evaluatingData){
            return evaluate(evaluatingData,modelColumns,preProcessModel);
        }

        /**
         * 评估模型，返回评估的性能指标，其中predictions条目表示预测结果
         *
         * @param evaluatingData
         * @param modelCols
         * @return
         */
        public Map<String,Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols){
            return evaluate(evaluatingData,modelCols,preProcessModel);
        }

        /**
         * 评估模型，返回评估的性能指标，其中predictions条目表示预测结果
         * @param evaluatingData
         * @param modelCols
         * @param preProcessModel
         * @return
         */
        public abstract Map<String,Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols,PipelineModel preProcessModel);

        /**
         * 使用内部的预测模型，modelCols、preProcessModel预测数据集
         * @param predictData
         * @return
         */
        public Dataset<Row> predict(Dataset<Row> predictData) {
            return predict(predictData,modelColumns,preProcessModel);
        }

        /**
         * 使用内部的预测模型，preProcessModel、外部的modelCols预测数据集
         * @param predictData
         * @param modelCols
         * @return
         */
        public Dataset<Row> predict(Dataset<Row> predictData,  ModelColumns modelCols) {
            return predict(predictData,modelCols,preProcessModel);
        }

        /**
         * 使用内部的预测模型，外部的modelCols、preProcessModel预测数据集
         * @param predictData
         * @param modelCols
         * @param preProcessModel
         * @return
         */
        public abstract Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel);

    }

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
