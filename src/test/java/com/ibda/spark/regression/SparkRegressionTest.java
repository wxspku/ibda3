package com.ibda.spark.regression;

import com.ibda.spark.SparkMLTest;
import com.ibda.spark.util.SparkUtil;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.junit.Test;

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
                new String[]{"zprice","zwheelba"},//"price","wheelbas", "engine_s", "horsepow", "wheelbas", "width", "length", "curb_wgt", "fuel_cap", "mpg"
                new String[]{"type","manufact"},//new String[]{"type","manufact","model"},
                new String[]{"manufact"},//,"model"
                "sales");//lnsales
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

    protected void loadTest03Data() {
        //car,age,gender,inccat,ed,marital
        modelColumns = new ModelColumns(
                new String[]{"X1_transaction_date","X2_house_age","X3_distance_to_the_nearest_MRT_station",
                        "X4_number_of_convenience_stores", "X5_latitude","X6_longitude"},//
                null,
                null,
                "Y_house_price_of_unit_area");
        loadDataSet(FilePathUtil.getAbsolutePath("data/Real_estate_valuation.xlsx", false), SparkUtil.EXCEL_FORMAT);
    }

    @Test
    public void test03MachineLearning() throws IOException {
        //GBTClassificationModel、LinearSVCModel只支持二分类
        this.loadTest03Data();
        test01LearningEvaluatingPredicting();
    }

    /*@Override
    public void testValidationSplitTuning() throws IOException {
        testTuning(false, RegressionEvaluator.class);
    }*/

    @Override
    public void testValidationSplitTuning() throws IOException {
        this.loadTest03Data();
        testTuning(false, RegressionEvaluator.class);
    }

    /*@Override
    public void testCrossValidationTuning() throws IOException {
        testTuning(true,RegressionEvaluator.class);
    }
*/
    @Override
    public void testCrossValidationTuning() throws IOException {
        this.loadTest03Data();
        testTuning(true,RegressionEvaluator.class);
    }

    @Override
    protected void initTuningGrid() {

    }

}
