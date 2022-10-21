package com.ibda.spark.regression;

import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.HashMap;
import java.util.Map;

public class LogicRegressionTest {
    LogicRegression regression = new LogicRegression(null);
    Dataset<Row> source = null;
    //age,ed,employ,address,income,debtinc,creddebt,othdebt,default
    ModelColumns modelColumns = new ModelColumns(
            new String[]{"age", "employ", "address", "income", "debtinc", "creddebt", "othdebt"},
            new String[]{"ed"},
            "default",
            "predicted",
            "probability");

    @Before
    public void setUp() throws Exception {
        source = regression.loadData(FilePathUtil.getAbsolutePath("data/bankloan_logistic.csv", false));
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void preProcess() {
        Dataset<Row> processed = regression.preProcess(source, modelColumns);
        processed.show();
    }

    @Test
    public void fit() {
        /*ParamMap paramMap = new ParamMap()
                .put(lr.maxIter().w(20))  // Specify 1 Param.
                .put(lr.maxIter(), 30)  // This overwrites the original maxIter.
                .put(lr.regParam().w(0.1), lr.threshold().w(0.55));  // Specify multiple Params.*/
        Dataset<Row> filter = source.filter("default is not null");
        System.out.println("记录数：" + filter.count());
        Map<String, Object> param = new HashMap<String, Object>();
        param.put("maxIter", 100);
        param.put("tol", 1E-10);
        param.put("threshold", 0.35);

        PredictionModel predictionModel = regression.fit(filter, modelColumns, param);
        System.out.println(predictionModel);
    }

    @Test
    public void evaluate() {
    }

    @Test
    public void predict() {
    }
}