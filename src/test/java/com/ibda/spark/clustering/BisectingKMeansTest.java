package com.ibda.spark.clustering;

import org.apache.spark.ml.clustering.BisectingKMeans;
import org.apache.spark.ml.clustering.BisectingKMeansModel;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;

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

    @Override
    public void testValidationSplitTuning() throws IOException {
        loadTest02Data();
        super.testValidationSplitTuning();
    }

    @Override
    public void testCrossValidationTuning() throws IOException {
        loadTest02Data();
        super.testCrossValidationTuning();
    }

    protected void initTuningGrid() {
        tuningParamGrid.put("k",new Integer[]{4,6,8});
        tuningParamGrid.put("distanceMeasure",new String[]{"euclidean" , "cosine"});
    }
}
