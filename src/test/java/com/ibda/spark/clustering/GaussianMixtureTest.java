package com.ibda.spark.clustering;

import org.apache.spark.ml.clustering.GaussianMixture;
import org.apache.spark.ml.clustering.GaussianMixtureModel;

import java.io.IOException;

public class GaussianMixtureTest extends SparkClusteringTest<GaussianMixture, GaussianMixtureModel>{

    @Override
    protected void loadTest01Data() {
        super.loadTest01Data();
        trainingParams.put("k",3);
        trainingParams.put("distanceMeasure","euclidean"); // 'euclidean' and 'cosine'
    }

    @Override
    protected void loadTest02Data() {
        super.loadTest02Data();
        trainingParams.put("k",4);
        trainingParams.put("distanceMeasure","euclidean"); // 'euclidean' and 'cosine'
    }

    @Override
    public void initTrainingParams() {
        trainingParams.put("seed",System.currentTimeMillis());
        trainingParams.put("maxIter",100);
        trainingParams.put("tol",1E-10);
    }

    @Override
    public void test01LearningEvaluatingPredicting() throws IOException {
        super.test01LearningEvaluatingPredicting();
    }

    @Override
    public void test02MachineLearning() throws IOException {
        super.test02MachineLearning();
    }
}
