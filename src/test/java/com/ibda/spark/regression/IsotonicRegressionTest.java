package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.regression.IsotonicRegression;
import org.apache.spark.ml.regression.IsotonicRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.junit.Test;

import java.io.IOException;

/**
 * 保序回归，实际上是一个分段线性回归，目前只支持一元回归
 */
public class IsotonicRegressionTest extends SparkRegressionTest<IsotonicRegression, IsotonicRegressionModel> {
    @Override
    protected void loadTest01Data() {
        //"sex", "age", "label", "censor"
        modelColumns = ModelColumns.MODEL_COLUMNS_DEFAULT;
        loadDataSet(FilePathUtil.getAbsolutePath("data/mllib/sample_isotonic_regression_libsvm_data.txt", true), "libsvm");
    }

    @Override
    public void initTrainingParams() {

    }

    @Test
    public void test01LearningEvaluatingPredicting() throws IOException {
        super.test01LearningEvaluatingPredicting();
        // Trains an isotonic regression model.
        /*IsotonicRegression ir = new IsotonicRegression();
        IsotonicRegressionModel model = ir.fit(datasets[0]);

        System.out.println("Boundaries in increasing order: " + model.boundaries() + "\n");
        System.out.println("Predictions associated with the boundaries: " + model.predictions() + "\n");

        // Makes predictions.
        Dataset<Row> transform = model.transform(datasets[1]);
        transform.show();*/

    }

    @Override
    public void test02MachineLearning() throws IOException {
        
    }
}
