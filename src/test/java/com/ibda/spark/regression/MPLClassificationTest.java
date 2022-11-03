package com.ibda.spark.regression;

import org.apache.spark.ml.classification.MultilayerPerceptronClassificationModel;
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier;

import java.io.IOException;

/**
 * 神经网络
 */
public class MPLClassificationTest extends SparkMLTest<MultilayerPerceptronClassifier, MultilayerPerceptronClassificationModel> {
    @Override
    public void initTrainingParams() {

        // specify layers for the neural network:
        // input layer of size 4 (features), two intermediate of size 5 and 4
        // and output of size 3 (classes)
        int[] layers = new int[] {5, 10,15, 20, 15,10,5, 2};

        trainingParams.put("layers",layers); //default 50
        trainingParams.put("blockSize", 128);
        trainingParams.put("maxIter",100);
        trainingParams.put("seed",System.currentTimeMillis());

        trainingParams.put("tol", 1.0E-8); //default 1
        trainingParams.put("solver", "l-bfgs"); //gd 、l-bfgs
        trainingParams.put("stepSize",0.001d); //default 0.1
    }

    @Override
    public void test02LearningMultinomialRegression() throws IOException {
        trainingParams.put("layers",new int[] {4, 10,15, 20, 15,10,5, 3}); //default 50
        super.test02LearningMultinomialRegression();
    }
}
