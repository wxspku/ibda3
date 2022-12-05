package com.ibda.spark.regression;


import cn.hutool.core.util.ReflectUtil;
import com.ibda.spark.statistics.BasicStatistics;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.BooleanParam;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.ml.tuning.ParamGridBuilder;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.util.*;

/**
 * Spark机器学习通用类，支持训练、评估（使用训练方法的返回值）、预测
 * 泛型E代表 Estimator，用于训练
 * 泛型M代表 Model，用于评估和预测
 */
public class SparkML<E extends Estimator,M extends Model> extends BasicStatistics {

    private Class<E> eClass = null;

    /**
     *
     * @param eClass
     */
    public SparkML(Class<E> eClass){
        this(null,eClass);
    }
    /**
     *
     * @param appName
     * @param eClass
     */
    public SparkML(String appName, Class<E> eClass) {
        super(appName);
        this.eClass = eClass;
    }


    /**
     * 训练回归模型，同时使用训练集数据训练数据预处理模型
     * @param trainingData
     * @param modelCols
     * @param params
     * @return
     */
    public SparkHyperModel<M> fit(Dataset<Row> trainingData, ModelColumns modelCols, Map<String, Object> params){
        PipelineModel preProcessModel = modelCols.fit(trainingData);
        return fit(trainingData, modelCols,preProcessModel,params);
    }

    /**
     * @param trainingData      原始训练集
     * @param modelCols         模型分列设置
     * @param preProcessModel   数据预处理模型，需要先进行预训练，使用ModelColumns.fit方法进行训练
     * @param params    训练参数，根据回归类型及回归算法不同，参数名称也有所不同，具体参见spark文档
     * @return
     */
    public SparkHyperModel<M> fit(Dataset<Row> trainingData, ModelColumns modelCols,
                                  PipelineModel preProcessModel, Map<String, Object> params){
        //预处理
        Dataset<Row> training = modelCols.transform(trainingData,preProcessModel);
        //泛型E的class
        E estimator = (E)ReflectUtil.newInstance(eClass);
        ParamMap paramMap = populateEstimator(estimator, modelCols, params);
        M model = (M)estimator.fit(training,paramMap);
        SparkHyperModel<M> hyperModel = new SparkHyperModel<M>(model,preProcessModel,modelCols);
        if (hyperModel.getTrainingMetrics().isEmpty()){
            //通过训练集评估训练结果指标
            Map<String, Object> trainingMetrics = hyperModel.evaluate(training, modelCols, preProcessModel);
            hyperModel.setTrainingMetrics(trainingMetrics);
        }
        return hyperModel;

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
    public Dataset<Row> predict(Dataset<Row> predictData, ModelColumns modelCols, PipelineModel preProcessModel, Model model){
        SparkHyperModel<M> hyperModel = new SparkHyperModel( model, preProcessModel, modelCols);
        return hyperModel.predict(predictData);
    }

    /**
     * 预测单条数据
     *
     * @param model     预测模型
     * @param features  经过预处理后的特征数据
     * @return
     */
    public double predict(PredictionModel model, final Vector features){
        return model.predict(features);
    }

    /**
     * k-折交叉验证方式训练模型，将训练数据划分为numFolds份，针对每个参数组合，执行numFolds轮训练和评估，numFolds轮评估的均值即为该参数组合的最终评估值
     * 每轮训练选取其中numFolds-1份数据进行模型训练，剩余1份数据用于模型评估。选择性能最优的模型输出
     * 假设参数组合数为m，则训练轮次：m * numFolds，因此性能较低，且非常耗费资源
     * @param trainingData
     * @param modelCols
     * @param preProcessModel
     * @param params     已明确，无需评估的参数
     * @param evaluatorClass     评估类，用于评估模型
     * @param numFolds   交叉验证的折数，惯例数为10，或选取K≈log(n),且n/K>3d，d为特征数，也可以设置其他数，设为0时，由系统自动选取
     * @param paramGrid  候选参数集Map，key为参数名，value为该参数的多个评估值，bool类型的参数,value为空
     * @return
     */
    public SparkHyperModel<M> fitByCrossValidator(Dataset<Row> trainingData, ModelColumns modelCols,
                                                  PipelineModel preProcessModel, Map<String, Object> params,
                                                  Class evaluatorClass, int numFolds, Map<String, Object[]> paramGrid){
        return null;
    }

    /**
     * 简单训练集划分方式验证训练模型，将训练集按照指定比例7:3进行固定划分，70%用于训练，30%用于评估
     * 针对每个参数组合，只执行一次训练和评估
     * @param trainingData
     * @param modelCols
     * @param preProcessModel
     * @param params
     * @param evaluatorClass     评估类，用于评估模型
     * @param paramGrid
     * @return
     */
    public SparkHyperModel<M> fitByTrainSplitValidator(Dataset<Row> trainingData, ModelColumns modelCols,
                                                       PipelineModel preProcessModel, Map<String, Object> params,
                                                       Class evaluatorClass, Map<String, Object[]> paramGrid){
        return null;
    }

    /**
     *
     * @param modelCols
     * @param params
     * @return
     */
    protected ParamMap populateEstimator(E estimator, ModelColumns modelCols, Map<String, Object> params) {
        //合并参数
        params.put("featuresCol", modelCols.featuresCol);
        params.put("labelCol", modelCols.labelCol);
        params.put("predictionCol", modelCols.predictCol);
        params.put("probabilityCol", modelCols.probabilityCol);
        if (modelCols.weightCol != null){
            params.put("weightCol", modelCols.weightCol);
        }

        ParamMap paramMap = ParamMap.empty();
        if ((params != null && !params.isEmpty())){
            String parent = estimator.uid();
            params.entrySet().stream().forEach(entry -> {
                Param param = new Param(parent, entry.getKey(), null);
                try{
                    if (estimator.isDefined(param)) {
                        estimator.set(param, entry.getValue());
                    }
                }
                catch (Exception ex){
                    //不需要的参数，忽略
                }
                paramMap.put(param, entry.getValue());
            });
        }
        return paramMap;
    }

    protected ParamMap[] buildParamGrid(E estimator,Map<String, Object[]> paramGrid){
        ParamGridBuilder builder = new ParamGridBuilder();
        String parent = estimator.uid();
        paramGrid.entrySet().stream().forEach(entry -> {
            if (entry.getValue() == null){
                BooleanParam param = new BooleanParam(parent, entry.getKey(), null);
                builder.addGrid(param);
            }
            else{
                /**
                 * ParamGridBuilder	addGrid(DoubleParam param, double[] values)
                 * Adds a double param with multiple values.
                 * ParamGridBuilder	addGrid(FloatParam param, float[] values)
                 * Adds a float param with multiple values.
                 * ParamGridBuilder	addGrid(IntParam param, int[] values)
                 * Adds an int param with multiple values.
                 * ParamGridBuilder	addGrid(LongParam param, long[] values)
                 */
                Param param = new Param(parent, entry.getKey(), null);
                /*scala.collection.mutable.TreeSet treeSet = new scala.collection.mutable.TreeSet();
                ListSet listSet = new ListSet();
                listSet.
                builder.addGrid(param, listSet);*/
            }

        });
        return builder.build();
    }
}
