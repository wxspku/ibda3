package com.ibda.spark.clustering;

import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;

import java.io.IOException;

public class BisectingKMeansTest extends SparkClusteringTest<BisectingKMeans, BisectingKMeansModel>{
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
        trainingParams.put("minDivisibleClusterSize",0.05d);
    }

    @Override
    public void test02MachineLearning() throws IOException {
        super.test02MachineLearning();
    }
}
