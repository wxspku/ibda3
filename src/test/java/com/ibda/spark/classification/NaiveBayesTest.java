package com.ibda.spark.classification;

import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;

public class NaiveBayesTest extends SparkClassificationTest<NaiveBayes, NaiveBayesModel> {


    @Override
    public void initTrainingParams() {
        trainingParams.put("modelType", "gaussian"); // "multinomial", "complement", "bernoulli", "gaussian"  default = multinomial
        trainingParams.put("smoothing", 0.5d);
    }
}
