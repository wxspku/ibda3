package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.classification.NaiveBayes;
import org.apache.spark.ml.classification.NaiveBayesModel;

public class NaiveBayesTest extends SparkMLTest<NaiveBayes, NaiveBayesModel> {


    @Override
    public void initTrainingParams(){
        trainingParams.put("modelType","gaussian"); // "multinomial", "complement", "bernoulli", "gaussian"  default = multinomial
        trainingParams.put("smoothing",0.5d);
    }
}
