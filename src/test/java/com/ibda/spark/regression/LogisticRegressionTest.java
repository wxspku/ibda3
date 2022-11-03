package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.sql.Column;

import java.util.Arrays;

public class LogisticRegressionTest extends SparkMLTest<LogisticRegression, LogisticRegressionModel> {
    @Override
    public void prepareData() {
        loadBinomialData();
        initTrainingParams();
    }

    protected void loadBinomialData() {
        modelColumns = new ModelColumns(
                new String[]{"Age"},
                new String[]{"Income", "Credit_cards", "Education", "Car_loans"},
                "Credit_rating");
        loadDataSet(FilePathUtil.getAbsolutePath("data/credit_decision_tree.csv", false),"csv");
    }

    protected void loadMultinomialData() {
        //agecat,gender,marital,active,bfast
        modelColumns = new ModelColumns(
                null,
                new String[]{"agecat","gender","marital","active"},
                "preferbfast");
        loadDataSet(FilePathUtil.getAbsolutePath("data/cereal_multinomial.csv", false),"csv");


    }
    @Override
    protected void initTrainingParams(){
        trainingParams.put("maxIter", 100);
        trainingParams.put("tol", 1E-8);
        trainingParams.put("threshold", 0.5);
        trainingParams.put("regParam", 0.1);
        trainingParams.put("elasticNetParam", 0.8);
    }
}
