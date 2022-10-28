package com.ibda.spark.regression;

import com.ibda.spark.util.SparkUtil;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.ml.regression.LinearRegressionTrainingSummary;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class LinearRegressionTest {
    SparkML<LinearRegression,LinearRegressionModel> linear = new SparkML<>(LinearRegression.class);
    Dataset<Row> carSales = null;
    PipelineModel pipelineModel = null;
    //age,ed,employ,address,income,debtinc,creddebt,othdebt,default
    ModelColumns modelColumns = new ModelColumns(
            new String[]{"price", "engine_s", "horsepow", "wheelbas", "width", "length", "curb_wgt","fuel_cap","mpg"},
            new String[]{"manufact"},//new String[]{"type","manufact"},
            new String[]{"manufact"},//new String[]{"manufact"},
            "lnsales");

    @Before
    public void setUp() throws Exception {
        carSales = linear.loadData(FilePathUtil.getAbsolutePath("data/car_sales_linear.csv", false));
        pipelineModel = modelColumns.fit(carSales);
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testLinearRegression() throws Exception {
        // Load training data.
        SparkSession spark = SparkUtil.buildSparkSession("ILoveLinearRegression");
        Dataset<Row> training = spark.read().format("libsvm")
                .load("data/mllib/sample_linear_regression_data.txt");
        training.show();
        LinearRegression lr = new LinearRegression()
                .setMaxIter(10)
                .setRegParam(0.3)
                .setElasticNetParam(0.8);
        System.out.println(lr.defaultParamMap());
        // Fit the model.
        LinearRegressionModel lrModel = lr.fit(training);

        // Print the coefficients and intercept for linear regression.
        System.out.println("Coefficients: "
                + lrModel.coefficients() + " Intercept: " + lrModel.intercept());

        // Summarize the model over the training set and print out some metrics.
        LinearRegressionTrainingSummary trainingSummary = lrModel.summary();
        System.out.println("numIterations: " + trainingSummary.totalIterations());
        System.out.println("objectiveHistory: " + Vectors.dense(trainingSummary.objectiveHistory()));
        trainingSummary.residuals().show();
        System.out.println("RMSE: " + trainingSummary.rootMeanSquaredError());
        System.out.println("r2: " + trainingSummary.r2());
        System.out.println("pValues: " + Arrays.toString(trainingSummary.pValues()));
        System.out.println("tValues: " + Arrays.toString(trainingSummary.tValues()));
    }

    @Test
    public void testCarSalesRegression() throws Exception {
        Map<String, Object> param = new HashMap<String, Object>();
        param.put("maxIter", 100);
        param.put("tol", 1E-8);
        param.put("regParam", 0.2);
        param.put("elasticNetParam", 0.7);
        /*ParamMap paramMap = new ParamMap()
                .put(lr.maxIter().w(20))  // Specify 1 Param.
                .put(lr.maxIter(), 30)  // This overwrites the original maxIter.
                .put(lr.regParram().w(0.1), lr.threshold().w(0.55));  // Specify multiple Params.*/
        System.out.println("测试多元线性回归分析,car_sales数据集------------------");
        Dataset<Row> trainAndTest = carSales.filter("lnsales is not null");
        //划分训练集、测试集
        Dataset<Row>[] datasets = trainAndTest.randomSplit(new double[]{0.6d, 0.4d});
        Dataset<Row> training = datasets[0];
        Dataset<Row> testing = datasets[1];
        System.out.println(String.format("记录总数：%1$s,训练集大小：%2$s,测试集大小：%3$s",trainAndTest.count(),training.count(),testing.count()));

        //训练
        System.out.println("训练多元线性回归模型------------------");
        SparkHyperModel<LinearRegressionModel> linearModel = linear.fit(training, modelColumns,pipelineModel, param);
        System.out.println("训练模型结果及性能\n:" + linearModel);
        //评估
        Map<String, Object> metrics = linearModel.evaluate(testing);
        System.out.println("评估模型性能\n:" + metrics);
        Dataset<Row> tested = SparkHyperModel.getEvaluatePredictions(metrics);
        tested.show();
        //预测
        Dataset<Row> predicting = testing;//carSales.filter("lnsales is null");
        System.out.println("预测数据集:" + predicting.count());
        predicting.show();
        Dataset<Row> predicted = linearModel.predict(predicting);
        System.out.println("预测结果集:" + predicted.count());
        predicted.show();

        //预测单个数据
        Row[] rows = (Row[])predicted.select("regression_features_vector").head(20);
        Arrays.stream(rows).forEach(row->{
            GenericRowWithSchema gRow = (GenericRowWithSchema)row;
            Vector data = (Vector)gRow.values()[0];
            double label = linear.predict(linearModel.getModel(),data);
            System.out.println(data.toString() + ":" + label);
        });


        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/car_sales_linear.model", true);
        System.out.println("保存及加载模型："  + modelPath);
        linearModel.saveModel(modelPath);

        SparkHyperModel<LinearRegressionModel> result3= SparkHyperModel.loadFromModelFile(modelPath, LinearRegressionModel.class);
        Map<String, Object> metrics2 = result3.evaluate(testing, modelColumns,pipelineModel );
        System.out.println("评估存储模型性能\n:" + metrics2);

        System.out.println("使用存储模型进行预测\n:");
        Dataset<Row> predicted2 = linear.predict(predicting,modelColumns,pipelineModel,result3.getModel());
        predicted2.show();
    }
}