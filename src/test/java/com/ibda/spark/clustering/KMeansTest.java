package com.ibda.spark.clustering;


import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.clustering.KMeansModel;
import org.junit.Test;

import java.io.IOException;

public class KMeansTest extends SparkClusteringTest<KMeans, KMeansModel> {
    @Override
    protected void loadTest01Data() {
        super.loadTest01Data();
        trainingParams.put("k",3);
        trainingParams.put("distanceMeasure","euclidean"); // 'euclidean' and 'cosine'
    }

    @Override
    protected void loadTest02Data() {
        super.loadTest02Data();
        trainingParams.put("k",6);
        trainingParams.put("distanceMeasure","euclidean"); // 'euclidean' and 'cosine'
    }
    @Override
    public void initTrainingParams() {
        trainingParams.put("seed",System.currentTimeMillis());
        trainingParams.put("maxIter",100);
        trainingParams.put("tol",1E-10);
        trainingParams.put("initMode","k-means||"); //random "k-means||"
    }

    @Override
    public void test02MachineLearning() throws IOException {
        super.test02MachineLearning();
    }
}
