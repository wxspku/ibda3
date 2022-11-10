package com.ibda.spark.regression;

import com.ibda.spark.SparkMLTest;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;

import java.io.IOException;

/**
 * ExplainedVariance = SSR / N :  regression. explainedVariance = $\sum_i (\hat{y_i} - \bar{y})^2^ / n$
 * SSE = MSE * K
 *
 * r2 = ExplainedVariance * N /(ExplainedVariance * N + MSE * K)
 *
 *  regression score. explainedVariance = 1 - variance(y - \hat{y}) / variance(y)
 * @param <E>
 * @param <M>
 */
public abstract class SparkRegressionTest<E extends Estimator, M extends Model> extends SparkMLTest<E, M> {


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
