package com.ibda.spark.classification;

import com.ibda.spark.regression.ModelColumns;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;

public class LogisticRegressionTest extends SparkClassificationTest<LogisticRegression, LogisticRegressionModel> {


    protected void loadTest01Data() {
        modelColumns = new ModelColumns(
                new String[]{"Age"},
                new String[]{"Income", "Credit_cards", "Education", "Car_loans"},
                "Credit_rating");
        loadDataSet(FilePathUtil.getAbsolutePath("data/credit_decision_tree.csv", false), "csv");
    }

    protected void loadTest02Data() {
        //agecat,gender,marital,active,bfast
        /*modelColumns = new ModelColumns(
                null,
                new String[]{"agecat","gender","marital","active"},
                "preferbfast");
        loadDataSet(FilePathUtil.getAbsolutePath("data/cereal_multinomial.csv", false),"csv");*/
        modelColumns = ModelColumns.MODEL_COLUMNS_DEFAULT;
        loadDataSet(FilePathUtil.getAbsolutePath("data/mllib/sample_multiclass_classification_data.txt", true),
                "libsvm");

    }

    @Override
    public void initTrainingParams() {
        trainingParams.put("maxIter", 100);
        trainingParams.put("tol", 1E-8);
        trainingParams.put("threshold", 0.5);
        trainingParams.put("regParam", 0.1);
        trainingParams.put("elasticNetParam", 0.8);
    }
}
