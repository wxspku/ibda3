package com.ibda.spark.regression;


import cn.hutool.core.util.ReflectUtil;
import com.ibda.spark.statistics.BasicStatistics;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.fpm.FPGrowth;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.BooleanParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.param.Params;
import org.apache.spark.ml.tuning.*;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/**
 * Spark机器学习通用类，支持训练、评估（使用训练方法的返回值）、预测
 * 泛型E代表 Estimator，用于训练
 * 泛型M代表 Model，用于评估和预测
 */
public class SparkML<E extends Estimator, M extends Model> extends BasicStatistics {
    /**
     * 抽象训练器，支持确定超参数值的训练、指定多个超参数值的调优训练
     *
     * @param <M>
     */
    abstract class HyperEstimator<M extends Model> {
        /**
         * 训练模板方法
         *
         * @param trainingData
         * @param modelCols
         * @param preProcessModel
         * @param params
         * @param paramGrid       name-value对，name为参数名，value为候选值组成的一个数组
         * @param tuningParams    除候选参数以外的评估超参数，比如评估类、k-folds
         * @return
         */
        public SparkHyperModel<M> fit(Dataset<Row> trainingData, ModelColumns modelCols,
                                      PipelineModel preProcessModel, Map<String, Object> params,
                                      Map<String, Object[]> paramGrid, Map<String, Object> tuningParams) {
            //预处理
            Dataset<Row> training = modelCols.transform(trainingData, preProcessModel);
            //泛型E的class
            E estimator = ReflectUtil.newInstance(eClass);
            //paramGrid和params重名时，以paramGrid为准，去除params中的同名参数
            if (paramGrid != null && !paramGrid.isEmpty()){
                paramGrid.keySet().stream().forEach(key->{
                    params.remove(key);
                });
            }
            populateParams(estimator, modelCols, params);
            ParamMap[] hyperGrid = buildParamGrid(estimator, paramGrid);
            M model = hyperFit(estimator, training, params, hyperGrid, tuningParams);
            SparkHyperModel<M> hyperModel = new SparkHyperModel<M>(model, preProcessModel, modelCols);
            if (hyperModel.getTrainingMetrics().isEmpty()) {
                //通过训练集评估训练结果指标
                Map<String, Object> trainingMetrics = hyperModel.evaluate(training, modelCols, preProcessModel);
                hyperModel.setTrainingMetrics(trainingMetrics);
            }
            return hyperModel;

        }

        /**
         * 根据参数训练模型，具体训练方法由实现类确定
         *
         * @param estimator
         * @param params
         * @param paramGrid
         * @param tuningParams
         * @return
         */
        abstract M hyperFit(Estimator estimator, Dataset<Row> training, Map<String, Object> params, ParamMap[] paramGrid, Map<String, Object> tuningParams);
    }


    private Class<E> eClass = null;

    /**
     * @param eClass
     */
    public SparkML(Class<E> eClass) {
        this(null, eClass);
    }

    /**
     * @param appName
     * @param eClass
     */
    public SparkML(String appName, Class<E> eClass) {
        super(appName);
        this.eClass = eClass;
    }


    /**
     * 训练回归模型，同时使用训练集数据训练数据预处理模型
     *
     * @param trainingData
     * @param modelCols
     * @param params
     * @return
     */
    public SparkHyperModel<M> fit(Dataset<Row> trainingData, ModelColumns modelCols, Map<String, Object> params) {
        PipelineModel preProcessModel = (eClass.equals(FPGrowth.class))?null:modelCols.fit(trainingData);
        return fit(trainingData, modelCols, preProcessModel, params);
    }

    /**
     * @param trainingData    原始训练集
     * @param modelCols       模型分列设置
     * @param preProcessModel 数据预处理模型，需要先进行预训练，使用ModelColumns.fit方法进行训练
     * @param params          训练参数，根据回归类型及回归算法不同，参数名称也有所不同，具体参见spark文档
     * @return
     */
    public SparkHyperModel<M> fit(Dataset<Row> trainingData, ModelColumns modelCols,
                                  PipelineModel preProcessModel, Map<String, Object> params) {
        HyperEstimator hyperEstimator = new HyperEstimator() {

            @Override
            Model hyperFit(Estimator estimator, Dataset training, Map params, ParamMap[] paramGrid, Map tuningParams) {
                M model = (M) estimator.fit(training);
                return model;
            }
        };
        return hyperEstimator.fit(trainingData, modelCols, preProcessModel, params, null, null);
    }


    /**
     * 预测数据集
     *
     * @param predictData
     * @param modelCols
     * @param preProcessModel
     * @param model
     * @return
     */
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel, Model model) {
        SparkHyperModel<M> hyperModel = new SparkHyperModel(model, preProcessModel, modelCols);
        return hyperModel.predict(predictData);
    }

    /**
     * 预测单条数据
     *
     * @param model    预测模型
     * @param features 经过预处理后的特征数据
     * @return
     */
    public double predict(PredictionModel model, final Vector features) {
        return model.predict(features);
    }

    /**
     * k-折交叉验证方式训练模型，将训练数据划分为numFolds份，针对每个参数组合，执行numFolds轮训练和评估，numFolds轮评估的均值即为该参数组合的最终评估值
     * 每轮训练选取其中numFolds-1份数据进行模型训练，剩余1份数据用于模型评估。选择性能最优的模型输出
     * 假设参数组合数为m，则训练轮次：m * numFolds，因此性能较低，且非常耗费资源
     *
     * @param trainingData
     * @param modelCols
     * @param preProcessModel
     * @param params          已明确，无需评估的参数
     * @param evaluatorClass  评估类，用于评估模型
     * @param numFolds        交叉验证的折数，至少3以上，惯例数为10，或选取K≈log(n),且n/K>3d，d为特征数，也可以设置其他数，设为0时，由系统自动选取
     * @param paramGrid       候选参数集Map，key为参数名，value为该参数的多个评估值，bool类型的参数,value为空
     * @return
     */
    public SparkHyperModel<M> fitByCrossValidator(Dataset<Row> trainingData, ModelColumns modelCols,
                                                  PipelineModel preProcessModel, Map<String, Object> params,
                                                  Map<String, Object[]> paramGrid, Class<? extends Evaluator> evaluatorClass, int numFolds) {
        HyperEstimator hyperEstimator = new HyperEstimator() {
            @Override
            Model hyperFit(Estimator estimator, Dataset training, Map params, ParamMap[] paramGrid, Map tuningParams) {
                Evaluator evaluator = buildEvaluator(tuningParams, params);
                CrossValidator crossValidator = new CrossValidator()
                        .setEstimator(estimator)
                        .setEvaluator(evaluator)
                        .setEstimatorParamMaps(paramGrid)
                        .setNumFolds((int) tuningParams.get("numFolds"))  // Use 3+ in practice
                        .setParallelism(8);

                // Run train cross validation, and choose the best set of parameters.
                CrossValidatorModel crossValidatorModel = crossValidator.fit(training);
                M model = (M) crossValidatorModel.bestModel();
                return model;
            }
        };
        Map<String, Object> tuningParams = new HashMap<>();
        tuningParams.put("evaluatorClass", evaluatorClass);
        tuningParams.put("numFolds", numFolds == 0 ? 10 : numFolds);
        return hyperEstimator.fit(trainingData, modelCols, preProcessModel, params, paramGrid, tuningParams);
    }



    /**
     * 简单训练集划分方式验证训练模型，将训练集按照指定比例7:3进行固定划分，70%用于训练，30%用于评估
     * 针对每个参数组合，只执行一次训练和评估
     *
     * @param trainingData
     * @param modelCols
     * @param preProcessModel
     * @param params
     * @param paramGrid
     * @param evaluatorClass  评估类，用于评估模型
     * @param trainRatio      训练集中用于训练的数据比例
     * @return
     */
    public SparkHyperModel<M> fitByTrainSplitValidator(Dataset<Row> trainingData, ModelColumns modelCols,
                                                       PipelineModel preProcessModel, Map<String, Object> params,
                                                       Map<String, Object[]> paramGrid, Class<? extends Evaluator> evaluatorClass, double trainRatio) {
        HyperEstimator hyperEstimator = new HyperEstimator() {
            @Override
            Model hyperFit(Estimator estimator, Dataset training, Map params, ParamMap[] paramGrid, Map tuningParams) {
                Evaluator evaluator = buildEvaluator(tuningParams, params);
                TrainValidationSplit trainValidationSplit = new TrainValidationSplit()
                        .setEstimator(estimator)
                        .setEvaluator(evaluator)
                        .setEstimatorParamMaps(paramGrid)
                        .setTrainRatio((double) tuningParams.get("trainRatio"))  // 70% for training and the remaining 20% for validation
                        .setParallelism(8);  // Evaluate up to 2 parameter settings in parallel

                // Run train validation split, and choose the best set of parameters.
                TrainValidationSplitModel splitModel = trainValidationSplit.fit(training);
                M model = (M) splitModel.bestModel();
                return model;
            }
        };
        Map<String, Object> tuningParams = new HashMap<>();
        tuningParams.put("evaluatorClass", evaluatorClass);
        tuningParams.put("trainRatio", (trainRatio >= 1 || trainRatio <= 0) ? 0.7d : trainRatio);
        return hyperEstimator.fit(trainingData, modelCols, preProcessModel, params, paramGrid, tuningParams);

    }

    /**
     * @param modelCols
     * @param params
     * @return
     */
    protected static ParamMap populateParams(Params mlParams, ModelColumns modelCols, Map<String, Object> params) {
        //合并参数
        if (modelCols != null){
            params.put("featuresCol", modelCols.featuresCol);
            params.put("labelCol", modelCols.labelCol);
            params.put("predictionCol", modelCols.predictCol);
            params.put("probabilityCol", modelCols.probabilityCol);
            if (modelCols.weightCol != null) {
                params.put("weightCol", modelCols.weightCol);
            }
        }
        ParamMap paramMap = ParamMap.empty();
        if ((params != null && !params.isEmpty())) {
            params.entrySet().stream().forEach(entry -> {
                Param param = new Param(mlParams.uid(), entry.getKey(), null);
                try {
                    if (mlParams.isDefined(param)) {
                        mlParams.set(param, entry.getValue());
                    }
                } catch (Exception ex) {
                    //不需要的参数，忽略
                }
                paramMap.put(param, entry.getValue());
            });
        }
        return paramMap;
    }

    /**
     * 将参数Map转换为ParamMap数组
     *
     * @param mlParams
     * @param paramGrid
     * @return
     */
    protected static ParamMap[] buildParamGrid(Params mlParams, Map<String, Object[]> paramGrid) {
        if (paramGrid == null || paramGrid.isEmpty()) {
            return null;
        }
        ParamGridBuilder builder = new ParamGridBuilder();

        paramGrid.entrySet().stream().forEach(entry -> {
            if (entry.getValue() == null || entry.getValue().length == 0) {
                BooleanParam param = new BooleanParam(mlParams.uid(), entry.getKey(), null);
                builder.addGrid(param);
            } else {
                populateParamGrid(builder, mlParams.uid(), entry, entry.getValue()[0].getClass());
            }

        });
        return builder.build();
    }

    private static <T> void populateParamGrid(ParamGridBuilder builder, String parent, Map.Entry<String, Object[]> entry, Class<T> clazz) {
        Param<T> param = new Param<>(parent, entry.getKey(), null);
        Object[] entryValue = entry.getValue();
        int len = entryValue.length;
        /*scala.collection.mutable.Stack<T> stack = new scala.collection.mutable.Stack<>();
        Arrays.stream(entryValue).forEach(value -> {
            stack.push((T) value);
        });
        builder.addGrid(param, stack);*/
    }

    private static Evaluator buildEvaluator(Map tuningParams, Map params) {
        Class evaluatorClass = (Class) tuningParams.get("evaluatorClass");
        Evaluator evaluator = (Evaluator) ReflectUtil.newInstance(evaluatorClass);
        populateParams(evaluator,null, params);
        return evaluator;
    }
}
