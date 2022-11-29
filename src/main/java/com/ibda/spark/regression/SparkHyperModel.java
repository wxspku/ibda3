package com.ibda.spark.regression;

import cn.hutool.core.lang.Filter;
import cn.hutool.core.util.ReflectUtil;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.hadoop.shaded.org.apache.commons.beanutils.BeanUtils;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.classification.ClassificationModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.classification.OneVsRestModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.*;
import org.apache.spark.ml.util.HasTrainingSummary;
import org.apache.spark.ml.util.MLWritable;
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import java.io.IOException;
import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.*;

/**
 * 超模型，除机器学习的训练预测模型以外，还存储了其他参数
 *
 * @param <M>
 */
public class SparkHyperModel<M extends Model> implements Serializable {

    public static final String PREDICTIONS_KEY = "predictions";

    protected static final String[] EXCLUDE_METHODS =
            new String[]{"getClass", "toString", "hashCode", "wait", "notify", "notifyAll", "asBinary"};

    protected static final Class[] EXCLUDE_RETURN_TYPES = new Class[]{Dataset.class,
            BinaryClassificationMetrics.class,
            SparkSession.class};

    /**
     * @param path
     * @param clz
     * @param <M>
     * @return
     */
    public static <M extends Model> SparkHyperModel<M> loadFromModelFile(String path, Class<M> clz) {
        Method method = ReflectUtil.getMethodByName(clz, "load");
        if (method != null){
                M model = ReflectUtil.invokeStatic(method, path);
                SparkHyperModel<M> hyperModel = new SparkHyperModel<>(model);
                return hyperModel;
        }
        return null;


    }

    /**
     * 从评估性能数据获取带预测结果的测试记录集
     *
     * @param evaluateMetrics
     * @return
     */
    public static Dataset<Row> getEvaluatePredictions(Map<String, Object> evaluateMetrics) {
        return (Dataset<Row>) evaluateMetrics.get(PREDICTIONS_KEY);
    }

    /**
     * @param model
     */
    public SparkHyperModel(M model) {
        this(model, null, null);
    }

    /**
     * @param model
     * @param preProcessModel
     */
    public SparkHyperModel(M model, PipelineModel preProcessModel) {
        this(model, preProcessModel, null);
    }

    /**
     * @param model
     * @param preProcessModel
     * @param modelColumns
     */
    public SparkHyperModel(M model, PipelineModel preProcessModel, ModelColumns modelColumns) {
        this.model = model;
        this.preProcessModel = preProcessModel;
        this.modelColumns = modelColumns;
        initCoefficients();
        initMetrics();
    }


    /**
     * 训练好的机器学习模型，使用前需要将数据进行预处理，预处理模型参看preProcessModel
     */
    M model = null;

    /**
     * 数据预处理的PipelineModel，在训练回归模型时，可以同时使用训练数据训练预处理模型
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
     * 系数列表，List的每个元素为1套完整的系数，多元逻辑回归会返回多套系数，每套系数为OneVsOther的模型结果
     */
    public List<double[]> coefficientsList = new ArrayList<>();

    /**
     * 训练性能指标，根据回归算法不同而有所区别，部分算法不生成指标结果，从summary读取，或者使用
     * Evaluator（MulticlassClassification）自行运算
     */
    public Map<String, Object> trainingMetrics = new LinkedHashMap<>();

    public M getModel() {
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

    /**
     * 外部计算训练指标
     *
     * @param trainingMetrics
     */
    public void setTrainingMetrics(Map<String, Object> trainingMetrics) {
        this.trainingMetrics.putAll(trainingMetrics);
        this.predictions = (Dataset<Row>) trainingMetrics.get(PREDICTIONS_KEY);
    }

    @Override
    public String toString() {
        return "RegressionHyperModel{" +
                "model=" + model +
                ", preProcessModel=" + preProcessModel +
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
     * 评估模型，返回评估的性能指标，其中predictions条目表示预测结果
     *
     * @param evaluatingData
     * @param modelCols
     * @param preProcessModel
     * @return
     */
    public Map<String, Object> evaluate(Dataset<Row> evaluatingData, ModelColumns modelCols, PipelineModel preProcessModel) {
        Dataset<Row> testing = modelCols.transform(evaluatingData, preProcessModel);
        Method evaluateMethod = ReflectUtil.getMethodByName(model.getClass(), "evaluate");
        Map<String, Object> metrics = new LinkedHashMap<>();
        if (evaluateMethod != null) {
            Object summary = ReflectUtil.invoke(model, "evaluate", testing);
            Dataset<Row> predictions = ReflectUtil.invoke(summary, "predictions");
            metrics.put(PREDICTIONS_KEY, predictions);
            if (model instanceof RegressionModel ){
                metrics.putAll(getRegressionMetrics(modelCols,predictions));
            }
            metrics.putAll(this.buildMetrics(summary));

        } else {
            Dataset<Row> evaluated = model.transform(testing);
            metrics.put(PREDICTIONS_KEY, evaluated);
            //OneVsRestModel not a ClassificationModel
            if (model instanceof ClassificationModel || model instanceof OneVsRestModel) {
                MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator();
                evaluator.setLabelCol(modelCols.labelCol);
                evaluator.setPredictionCol(modelCols.predictCol);
                evaluator.setProbabilityCol(modelCols.probabilityCol);
                MulticlassMetrics classificationMetrics = evaluator.getMetrics(evaluated);
                metrics.putAll(this.buildMetrics(classificationMetrics));
            } else if (model instanceof RegressionModel || model instanceof IsotonicRegressionModel) {
                metrics.putAll(getRegressionMetrics(modelCols,evaluated));
            }
        }

        return metrics;
    }

    private  Map<String, Object> getRegressionMetrics(ModelColumns modelCols, Dataset<Row> evaluated) {
        Map<String, Object> metrics = new LinkedHashMap<>();
        RegressionEvaluator evaluator = new RegressionEvaluator();
        evaluator.setLabelCol(modelCols.labelCol);
        evaluator.setPredictionCol(modelCols.predictCol);
        RegressionMetrics regressionMetrics = evaluator.getMetrics(evaluated);
        metrics.putAll(this.buildMetrics(regressionMetrics));
        //SStot = Σ(观测值y-均值y)^2 , SSreg = Σ(预测值y-均值y)^2, SSerr = Σ(观测值y-预测值y)^2、SSy = Σy^2*w
        //r2 = 1 - SSerr/SStot、explainedVariance = SSreg / n、meanSquaredError = SSerr/n
        //devianceResiduals: 残差范围，[min, max]
        //非线性回归时，SStot ≠ SSreg + SSerr，且没有自由度的说法，或者说自由度为n
        String[] methodNames = new String[]{"SSy","SStot","SSreg","SSerr"};
        Arrays.stream(methodNames).forEach(methodName->{
            Method method= ReflectUtil.getMethodByName(RegressionMetrics.class,methodName);
            if (method != null){
                method.setAccessible(true);
                Object value = ReflectUtil.invoke(regressionMetrics,method);
                metrics.put(methodName,value);
            }
        });
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
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel) {
        Dataset<Row> predicting = modelCols.transform(predictData, preProcessModel);
        Dataset<Row> result = model.transform(predicting);
        return result;
    }

    /**
     * 预测单个数据
     *
     * @param features 经过转换后的向量形式的特征值
     * @return 预测的结果
     */
    public double predict(Vector features) {
        if (model instanceof PredictionModel) {
            PredictionModel predictionModel = ((PredictionModel) model);
            return predictionModel.predict(features);
        }
        throw new RuntimeException("Not a PredictionModel:" + model.getClass());
    }

    protected void initCoefficients() {
        //回归系数
        if (model instanceof LinearRegressionModel ||
                model instanceof GeneralizedLinearRegressionModel ||
                model instanceof AFTSurvivalRegressionModel ||
                model instanceof LogisticRegressionModel &&
                        ReflectUtil.invoke(model, "numClasses").equals(2)) {//线性回归或二元逻辑回归
            double[] array = ((Vector) ReflectUtil.invoke(model, "coefficients")).toArray();
            double[] coefficients = ArrayUtils.addFirst(array, ReflectUtil.invoke(model, "intercept"));
            coefficientsList.add(coefficients);
        } else if (model instanceof LogisticRegressionModel) {//多元回归
            //截距向量
            org.apache.spark.ml.linalg.Vector interceptVector = ReflectUtil.invoke(model, "interceptVector");
            double[] intercepts = interceptVector.toArray();
            //系数矩阵，每行作为一套系数OneVsOther
            org.apache.spark.ml.linalg.Matrix coefficientMatrix = ReflectUtil.invoke(model, "coefficientMatrix");
            scala.collection.Iterator<Vector> rowIter = coefficientMatrix.rowIter();
            //将截距向量和系数矩阵拼接为完整的系数
            int i = 0;
            while (rowIter.hasNext()) {
                double[] row = rowIter.next().toArray();
                double[] coefficients = ArrayUtils.addFirst(row, intercepts[i]);
                coefficientsList.add(coefficients);
                i++;
            }
        }
    }

    protected void initMetrics() {
        if (model instanceof HasTrainingSummary) {
            HasTrainingSummary summaryModel = (HasTrainingSummary) model;
            if (summaryModel.hasSummary()) {
                Object summary = summaryModel.summary();
                if (ReflectUtil.getMethodByName(summary.getClass(), "predictions") != null) {
                    this.predictions = ReflectUtil.invoke(summary, "predictions");
                    if (model instanceof RegressionModel){
                        trainingMetrics.putAll(getRegressionMetrics(modelColumns,predictions));
                    }

                }
                trainingMetrics.putAll(buildMetrics(summary));
            }
        } else { //根据Evaluator计算
            //throw new RuntimeException("Not implemented for none HasTrainingSummary model:" + model.getClass());
        }
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
        } catch (IllegalAccessException | InvocationTargetException | NoSuchMethodException e) {
            e.printStackTrace();
        }
        //通过方法获取性能指标
        // 线性回归模型根据参数部同，可能无法获取tValues、pValues数值
        // https://stackoverflow.com/questions/46696378/spark-linearregressionsummary-normal-summary
        List<Method> publicMethods = ReflectUtil.getPublicMethods(summary.getClass(), new Filter<Method>() {
            @Override
            public boolean accept(Method method) {
                //method.getName().equals("residuals") ||
                return  method.getParameterCount() == 0 &&
                        !ArrayUtils.contains(EXCLUDE_METHODS, method.getName()) &&
                        !ArrayUtils.contains(EXCLUDE_RETURN_TYPES, method.getReturnType());
            }
        });
        publicMethods.forEach(method -> {
            try {
                method.setAccessible(true);
                Object performance = method.invoke(summary);
                String name = method.getName();
                if (performance instanceof MulticlassMetrics) {
                    name = "multiclassMetrics";
                    Matrix confusionMatrix = ((MulticlassMetrics) performance).confusionMatrix();
                    //混淆矩阵，行为实际值，列为预测值，转换为数组时是先列后行进行转换
                    metrics.put("confusionMatrix", confusionMatrix);
                }
                else if (name.equals("residuals")){
                    Dataset<Row> residuals = (Dataset<Row>)performance;
                    Row[] minMax = (Row[])residuals.select("residuals").summary("min","max").take(2);
                    metrics.put("residuals_min",minMax[0].get(1));
                    metrics.put("residuals_max",minMax[1].get(1));
                }
                metrics.put(name, performance);

            } catch (IllegalAccessException | InvocationTargetException e) {
                e.printStackTrace();
            }
        });

        return metrics;
    }
}
