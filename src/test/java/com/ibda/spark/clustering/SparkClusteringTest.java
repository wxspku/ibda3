package com.ibda.spark.clustering;

import com.ibda.spark.SparkMLTest;
import com.ibda.spark.regression.ModelColumns;
import com.ibda.spark.regression.SparkHyperModel;
import com.ibda.spark.statistics.DescriptiveStatistics;
import com.ibda.util.AnalysisConst;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.IsotonicRegressionModel;
import org.apache.spark.sql.Column;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.apache.spark.sql.functions;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public abstract class SparkClusteringTest<E extends Estimator, M extends Model> extends SparkMLTest {

    @Override
    protected void loadTest01Data() {
        modelColumns = ModelColumns.MODEL_COLUMNS_DEFAULT;
        trainingSplit = false;
        loadDataSet(FilePathUtil.getAbsolutePath("data/mllib/iris_libsvm.txt", true), "libsvm");

    }

    @Override
    protected void loadTest02Data() {
        //region,tenure,age,marital,address,income,ed,employ,retire,gender,reside,tollfree,equip,callcard,wireless,
        // longmon,tollmon,equipmon,cardmon,wiremon,multline,voice,pager,internet,callid,callwait,forward,confer,ebill,
        // zlnlong,zlntoll,zlnequi,zlncard,zlnwire,lninc,zmultlin,zvoice,zpager,zinterne,zcallid,zcallwai,zforward,zconfer,zebill,
        // custcat,churn,cluster_no,cdistance
        modelColumns = new ModelColumns(
                new String[]{"zlnlong","zlntoll","zlnequi","zlncard","zlnwire","zmultlin","zvoice","zpager","zinterne",
                        "zcallid","zcallwai","zforward","zconfer","zebill"},
                null,
                null,
                "cluster_no");
        trainingSplit = false;
        loadDataSet(FilePathUtil.getAbsolutePath("data/telco_extra_cluster.csv", false), "csv");

    }

    protected void imputeTrainingDataset(){
        if (modelColumns.getNoneCategoryFeatures() == null){
            return;
        }
        //修改数据，将zln...字段的空值设置为-10,对应原始值的0
        Map<String,Column> columnMap = new HashMap<>();
        Arrays.stream(modelColumns.getNoneCategoryFeatures()).forEach(
                feature->{if (feature.contains("zln")){
                    columnMap.put(feature,functions.expr(String.format("case when %1$s is null then -5 else %1$s end",feature)));
                }
         });
        Dataset ds = datasets[0].withColumns(columnMap);
        //ds.write().option("header",true).csv("output/telco_extra_cluster.csv");
        datasets[0] = ds;
    }

    @Override
    public void test01LearningEvaluatingPredicting() throws IOException {
        //训练
        System.out.println("训练聚类模型：" + estimatorClass.getSimpleName() + "/" + modelClass.getSimpleName());
        SparkHyperModel<M> hyperModel = sparkLearning.fit(datasets[0], modelColumns, pipelineModel, trainingParams);
        System.out.println("训练模型结果及性能\n:" + hyperModel);
        if (hyperModel.getPredictions() != null) {
            hyperModel.getPredictions().show();
        }
        //评估,聚类结果的分类编号没有实际意义，需要和现有分类进行对应
        Map<String, Object> metrics = hyperModel.evaluate(datasets[0]);
        System.out.println("评估模型性能\n:" + metrics);
        Dataset<Row> tested = SparkHyperModel.getEvaluatePredictions(metrics);
        tested.show();

        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/" + modelClass.getSimpleName() + ".model", true);
        System.out.println("保存及加载模型：" + modelPath);
        hyperModel.saveModel(modelPath);

        SparkHyperModel<M> loadedModel = SparkHyperModel.loadFromModelFile(modelPath, modelClass);
        Map<String, Object> metrics2 = loadedModel.evaluate(datasets[0], modelColumns, pipelineModel);
        System.out.println("评估存储模型性能\n:" + metrics2);
    }
}
