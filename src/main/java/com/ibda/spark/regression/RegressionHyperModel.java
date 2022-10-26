package com.ibda.spark.regression;

import cn.hutool.core.util.ReflectUtil;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.shaded.org.apache.commons.beanutils.BeanUtils;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.LogisticRegressionSummary;
import org.apache.spark.ml.classification.LogisticRegressionTrainingSummary;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.util.HasTrainingSummary;
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

public  class RegressionHyperModel<T extends PredictionModel> {

    public static final String PREDICTIONS_KEY = "predictions";

    protected static final String[] EXCLUDE_METHODS =
            new String[]{"getClass", "toString", "hashCode", "wait", "notify", "notifyAll", "asBinary"};

    public  static RegressionHyperModel<PredictionModel> loadFromModelFile(String path) {
        //TODO 泛型问题
        PredictionModel model = LogisticRegressionModel.load(path);
        RegressionHyperModel<PredictionModel> hyperModel = new RegressionHyperModel<>(model);
        return hyperModel;
    }

    public RegressionHyperModel(T model) {
        this(model, null, null);
    }

    public RegressionHyperModel(T model, PipelineModel preProcessModel) {
        this(model, preProcessModel, null);
    }

    public RegressionHyperModel(T model, PipelineModel preProcessModel, ModelColumns modelColumns) {
        this.model = model;
        this.preProcessModel = preProcessModel;
        this.modelColumns = modelColumns;
    }



    /**
     * 训练好的回归模型
     */
    T model = null;

    /**
     * 数据预处理的PipelineModel，在训练回归模型时，同时使用训练数据训练预处理模型
     */
    PipelineModel preProcessModel = null;
    /**
     * 带预测结果列的数据集
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
    public Map<String, Object> trainingMetrics = new LinkedHashMap<>();

    public T getModel() {
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
     * @param path
     * @throws IOException
     */
    public void saveModel(String path) throws IOException {
        if (model instanceof MLWritable) {
            ((MLWritable) model).write().overwrite().save(path);
            return;
        }
        throw new RuntimeException("未实现的接口MLWritable.save(path)");
    }

    /**
     * 使用内部的Model、ModelColumns、PipelineModel评估模型，其中predictions条目表示预测结果
     *
     * @param evaluatingData
     * @return
     */
    public Map<String, Object> evaluate(Dataset<Row> evaluatingData) {
        return evaluate(evaluatingData, modelColumns, preProcessModel);
    }

    /**
     * 评估模型，返回评估的性能指标，其中predictions条目表示预测结果
     *
     * @param evaluatingData
     * @param modelCols
     * @return
     */
    public Map<String, Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols) {
        return evaluate(evaluatingData, modelCols, preProcessModel);
    }

    /**
     * 从summary对象提取性能指标
     *
     * @param summary
     * @return
     */
    protected Map<String, Object> buildMetrics(Object summary) {
        Map<String, Object> metrics = new LinkedHashMap<>();
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
        Arrays.stream(publicMethods).forEach(method -> {
            if (method.getParameterCount() == 0) {
                if (!ArrayUtils.contains(EXCLUDE_METHODS, method.getName())) {
                    try {
                        Object performance = method.invoke(summary);
                        if (!(performance instanceof Dataset) &&
                                !(performance instanceof BinaryClassificationMetrics) &&
                                !(performance instanceof SparkSession)) {
                            String name = method.getName();
                            if (performance instanceof MulticlassMetrics) {
                                name = "multiclassMetrics";
                                Matrix confusionMatrix = ((MulticlassMetrics) performance).confusionMatrix();
                                //混淆矩阵，行为实际值，列为预测值，转换为数组时是先列后行进行转换
                                metrics.put("confusionMatrix", confusionMatrix);
                            }
                            metrics.put(name, performance);
                        }
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }
            }
        });
        return metrics;
    }

    /**
     *
     * @param model
     */
    protected void initCoefficients(T model) {
        //回归系数
        if (ReflectUtil.invoke(model,"numClasses").equals(2)) {//二元回归
            double[] array = ((Vector)ReflectUtil.invoke(model,"coefficients()")).toArray();
            double[] coefficients = ArrayUtils.addFirst(array, ReflectUtil.invoke(model,"intercept"));
            coefficientsList.add(coefficients);
        } else {//多元回归
            //截距向量
            org.apache.spark.ml.linalg.Vector interceptVector = ReflectUtil.invoke(model,"interceptVector");
            double[] intercepts = interceptVector.toArray();
            //系数矩阵，每行作为一套系数OneVsOther
            org.apache.spark.ml.linalg.Matrix coefficientMatrix = ReflectUtil.invoke(model,"coefficientMatrix");
            scala.collection.Iterator<Vector> rowIter = coefficientMatrix.rowIter();
            //将截距向量和系数矩阵拼接为完整的系数
            int i = 0;
            while (rowIter.hasNext()) {
                double[] row = rowIter.next().toArray();
                double[] coefficients = ArrayUtils.addFirst(row, intercepts[i]);
                /*coefficients[0] = intercepts[i];
                System.arraycopy(row,0,coefficients,1,row.length);*/
                coefficientsList.add(coefficients);
                i++;
            }
        }
        if (model instanceof HasTrainingSummary){
            HasTrainingSummary summaryModel = (HasTrainingSummary) model;
            Object summary = summaryModel.summary();
            this.predictions = ReflectUtil.invoke(summary,"predictions");
            trainingMetrics.putAll(buildMetrics(summary));
        }
        else { //根据Evaluator计算

        }
    }

    /**
     * 评估模型，返回评估的性能指标，其中predictions条目表示预测结果
     *
     * @param evaluatingData
     * @param modelCols
     * @param preProcessModel
     * @return
     */
    public  Map<String, Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols, PipelineModel preProcessModel){
        Dataset<Row> testing = modelCols.transform(evaluatingData, preProcessModel);
        Object summary = ReflectUtil.invoke(model,"evaluate",testing);
        Map<String, Object> metrics = this.buildMetrics(summary);
        metrics.put("predictions", ReflectUtil.invoke(summary,"predictions"));
        return metrics;
    }

    /**
     * 使用内部的预测模型，modelCols、preProcessModel预测数据集
     *
     * @param predictData
     * @return
     */
    public Dataset<Row> predict(Dataset<Row> predictData) {
        return predict(predictData, modelColumns, preProcessModel);
    }

    /**
     * 使用内部的预测模型，preProcessModel、外部的modelCols预测数据集
     *
     * @param predictData
     * @param modelCols
     * @return
     */
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols) {
        return predict(predictData, modelCols, preProcessModel);
    }

    /**
     * 使用内部的预测模型，外部的modelCols、preProcessModel预测数据集
     *
     * @param predictData
     * @param modelCols
     * @param preProcessModel
     * @return
     */
    public  Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel) {
        Dataset<Row> predicting = modelCols.transform(predictData, preProcessModel);
        Dataset<Row> result = model.transform(predicting);
        return result;
    }

}
