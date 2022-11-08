package com.ibda.spark.classification;

import org.apache.spark.ml.classification.LinearSVC;
import org.apache.spark.ml.classification.OneVsRest;
import org.apache.spark.ml.classification.OneVsRestModel;

public class OneVsRestTest extends SparkClassificationTest<OneVsRest, OneVsRestModel> {
    /*@Override
    public void prepareData() {
        modelColumns = ModelColumns.MODEL_COLUMNS_DEFAULT;
        loadDataSet(FilePathUtil.getAbsolutePath("data/mllib/sample_multiclass_classification_data.txt", true),"libsvm");
        LinearSVC classifier = new LinearSVC()
                .setMaxIter(10)
                .setTol(1E-8)
                .setFitIntercept(true);
        trainingParams.put("classifier",classifier); //default 50
    }*/
    @Override
    public void initTrainingParams() {
        LinearSVC classifier = new LinearSVC()
                .setMaxIter(100)
                .setTol(1E-8)
                .setFitIntercept(true);
        trainingParams.put("classifier", classifier); //default 50
    }
}
