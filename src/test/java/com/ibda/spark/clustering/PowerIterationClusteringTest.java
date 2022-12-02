package com.ibda.spark.clustering;


import com.ibda.spark.regression.SparkHyperModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.PowerIterationClustering;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.evaluation.ClusteringMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

public class PowerIterationClusteringTest extends SparkClusteringTest<KMeans, KMeansModel>{

    @Override
    public void initTrainingParams() {

    }

    @Override
    protected void loadTest01Data() {
        super.loadTest01Data();
        trainingParams.put("k",3);
    }

    @Override
    protected void loadTest02Data() {
        super.loadTest02Data();
        trainingParams.put("k",6);
    }

    @Override
    public void test01LearningEvaluatingPredicting() throws IOException {
        //聚类参数
        //Column features must be of type equal to one of the following types: [int, bigint]
        PowerIterationClustering pic = new PowerIterationClustering()
            .setK((Integer)trainingParams.get("k"))
            .setMaxIter(100)
            .setInitMode("degree") // This can be either "random" to use a random vector as vertex properties, or "degree" to use a normalized sum of similarities with other vertices
            .setSrcCol(modelColumns.getFeaturesCol())
            .setDstCol(modelColumns.getPredictCol());
        //聚类
        Dataset<Row> result = pic.assignClusters(datasets[0]);
        result.show(false);
        //评估
        Map<String, Object> metrics = SparkHyperModel.getClusteringMetrics(modelColumns,result,null);
        System.out.println(metrics);
    }

    @Override
    public void test02MachineLearning() throws IOException {
        super.test02MachineLearning();
    }
}
