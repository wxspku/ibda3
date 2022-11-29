package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.regression.AFTSurvivalRegression;
import org.apache.spark.ml.regression.AFTSurvivalRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;

import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.regression.AFTSurvivalRegression;
import org.apache.spark.ml.regression.AFTSurvivalRegressionModel;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

public class AFTRegressionTest extends SparkRegressionTest<AFTSurvivalRegression, AFTSurvivalRegressionModel>{

    @Override
    protected void loadTest01Data() {
        //"sex", "age", "label", "censor"
        modelColumns = new ModelColumns(
                new String[]{"age"},
                new String[]{"sex"},//new String[]{"type","manufact"},
                null,//new String[]{"manufact"},
                "label");
        modelColumns.setAdditionCols(new String[]{"censor"});
        loadDataSet(FilePathUtil.getAbsolutePath("data/aft_sample.csv", false), "csv");
    }

    @Override
    public void initTrainingParams() {
        double[] quantileProbabilities = new double[]{0.3, 0.6};
        //trainingParams.put("quantileProbabilities",quantileProbabilities);
        trainingParams.put("quantilesCol","quantiles");
        trainingParams.put("censorCol","censor");
    }

    @Override
    public void test02MachineLearning() throws IOException {
        System.out.println("Not implemented");
    }
}
