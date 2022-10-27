package com.ibda.spark.regression;

import cn.hutool.core.util.ReflectUtil;
import com.ibda.spark.statistics.BasicStatistics;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.param.Param;
import org.apache.spark.ml.param.ParamMap;
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
        //合并参数
        params.put("featuresCol",modelCols.featuresCol);
        params.put("labelCol",modelCols.labelCol);
        params.put("predictionCol",modelCols.predictCol);
        params.put("probabilityCol",modelCols.probabilityCol);
        //泛型E的class
        E estimator = (E)ReflectUtil.newInstance(eClass);
        ParamMap paramMap = buildParams(estimator.uid(),params);
        M model = (M)estimator.fit(training,paramMap);
        SparkHyperModel<M> hyperModel = new SparkHyperModel<M>(model,preProcessModel,modelCols);
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
     * 构建训练用的参数集
     * @param parent
     * @param params
     * @return
     */
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
}
