package com.ibda.spark.regression;

import com.ibda.spark.SparkMLTest;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;

import java.io.IOException;

public abstract class SparkRegressionTest<E extends Estimator, M extends Model> extends SparkMLTest<E, M> {
    @Override
    public void prepareData() {
        loadTest01Data();
        initTrainingParams();
    }

    @Override
    public void test02MachineLearning() throws IOException {
        //GBTClassificationModel、LinearSVCModel只支持二分类
        this.loadTest02Data();
        test01LearningEvaluatingPredicting();
    }

    @Override
    protected void loadTest01Data() {
        modelColumns = new ModelColumns(
                new String[]{"price", "engine_s", "horsepow", "wheelbas", "width", "length", "curb_wgt", "fuel_cap", "mpg"},
                new String[]{"type"},//new String[]{"type","manufact"},
                null,//new String[]{"manufact"},
                "lnsales");
        loadDataSet(FilePathUtil.getAbsolutePath("data/car_sales_linear.csv", false), "csv");
    }

    @Override
    protected void loadTest02Data() {
        //car,age,gender,inccat,ed,marital
        modelColumns = new ModelColumns(
                new String[]{"age", "inccat", "ed", "marital"},
                new String[]{"gender"},
                new String[]{"gender"},
                "car");
        loadDataSet(FilePathUtil.getAbsolutePath("data/car_decision_tree.csv", false), "csv");
    }


}
