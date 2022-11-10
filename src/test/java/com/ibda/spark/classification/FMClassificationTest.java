package com.ibda.spark.classification;

import com.ibda.spark.regression.ModelColumns;
import com.ibda.spark.regression.SparkML;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.classification.FMClassificationModel;
import org.apache.spark.ml.classification.FMClassifier;
import org.junit.Before;

public class FMClassificationTest extends SparkClassificationTest<FMClassifier, FMClassificationModel> {

    @Before
    @Override
    public void prepareData() {
        this.scaleByMinMax = true;
        super.prepareData();
    }
    @Override
    public void initTrainingParams() {
        //trainingParams.put("stepSize", 0.001d); //default 0.1
        //trainingParams.put("regParam", 0.001d);
        trainingParams.put("tol", 1.0E-8);
        trainingParams.put("maxIter", 500);
        //trainingParams.put("fitLinear", true);
        //trainingParams.put("factorSize",2);
    }


}
