package com.ibda.spark.classification;

import com.ibda.spark.SparkMLTest;
import com.ibda.spark.regression.ModelColumns;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.classification.*;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.junit.Test;

import java.io.IOException;


public abstract class SparkClassificationTest<E extends Estimator, M extends Model> extends SparkMLTest<E, M> {

    //加载test01的缺省测试数据，二分类
    protected void loadTest01Data() {
        modelColumns = new ModelColumns(
                new String[]{"Age", "Income", "Credit_cards", "Education", "Car_loans"},
                null,
                "Credit_rating");
        loadDataSet(FilePathUtil.getAbsolutePath("data/credit_decision_tree.csv", false), "csv");
    }

    //加载test02的缺省测试数据，多分类
    protected void loadTest02Data() {
        //agecat,gender,marital,active,bfast，缺省加载，不做OneHotEncoder处理，逻辑回归需做处理
        modelColumns = new ModelColumns(
                new String[]{"agecat", "gender", "marital", "active"},
                null,
                "preferbfast");
        loadDataSet(FilePathUtil.getAbsolutePath("data/cereal_multinomial.csv", false), "csv");
    }

    @Test
    public void test02MachineLearning() throws IOException {
        //GBTClassificationModel、LinearSVCModel、FMClassificationModel只支持二分类
        if (ClassificationModel.class.isAssignableFrom(modelClass) &&
                !modelClass.equals(GBTClassificationModel.class) &&
                !modelClass.equals(LinearSVCModel.class) &&
                !modelClass.equals(FMClassificationModel.class)||
                modelClass.equals(OneVsRestModel.class)) {
            loadTest02Data();
            test01LearningEvaluatingPredicting();
        } else {
            System.err.println("Not a Multinomial Regression-----------------");
        }
    }

    @Override
    public void testValidationSplitTuning() throws IOException {
        testTuning(false, MulticlassClassificationEvaluator.class);
    }

    @Override
    public void testCrossValidationTuning() throws IOException {
        testTuning(true,MulticlassClassificationEvaluator.class);
    }

    protected void initTuningGrid() {

    }
}
