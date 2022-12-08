package com.ibda.spark.regression;

import com.ibda.spark.util.SparkUtil;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.MinMaxScaler;
import org.apache.spark.ml.feature.MinMaxScalerModel;
import org.apache.spark.ml.regression.FMRegressionModel;
import org.apache.spark.ml.regression.FMRegressor;
import org.apache.spark.mllib.evaluation.RegressionMetrics;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.junit.Before;
import org.junit.Test;

public class FMRegressionTest extends SparkRegressionTest<FMRegressor, FMRegressionModel> {

    @Before
    public void prepareData() {
        this.scaleByMinMax = true;
        super.prepareData();
    }

    @Override
    public void initTrainingParams() {
        trainingParams.put("maxIter", 500);
        trainingParams.put("tol", 1E-8);
        //trainingParams.put("factorSize", 8);
        /*trainingParams.put("stepSize",0.001);
        trainingParams.put("regParam", 0.1);
        trainingParams.put("elasticNetParam", 0.8);*/
    }

    @Override
    protected void loadTest02Data() {
        //car,age,gender,inccat,ed,marital
        modelColumns = new ModelColumns(
                new String[]{"age"},
                new String[]{"gender","inccat", "ed", "marital"},
                new String[]{"gender"},
                "car");
        loadDataSet(FilePathUtil.getAbsolutePath("data/car_decision_tree.csv", false), "csv");
    }

    @Override
    protected void initTuningGrid() {
        /*tuningParamGrid.put("solver",new String[]{"gd", "adamW"});
        tuningParamGrid.put("factorSize", new Integer[]{4,8,16});
        tuningParamGrid.put("regParam",new Double[]{0d,0.05d,0.1d,0.2d});*/
        tuningParamGrid.put("miniBatchFraction",new Double[]{0.70d,0.75d,0.8d});
        tuningParamGrid.put("stepSize",new Double[]{1.5d,2d,2.5d});
    }
    //@Test
    public void testFMRegression() {
        SparkSession spark = SparkUtil.buildSparkSession("FMRegressionTest");
        Dataset<Row> data = spark.read().format("libsvm").load("data/mllib/sample_libsvm_data.txt");

        // Scale features.
        MinMaxScalerModel featureScaler = new MinMaxScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .fit(data);

        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[] {0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a FM model.
        FMRegressor fm = new FMRegressor()
                .setLabelCol("label")
                .setFeaturesCol("scaledFeatures")
                .setStepSize(0.001);

        // Create a Pipeline.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {featureScaler, fm});

        // Train model.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("prediction", "label", "features").show(5);
        predictions.show();

        // Select (prediction, true label) and compute test error.
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);


        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        FMRegressionModel fmModel = (FMRegressionModel)(model.stages()[1]);
        System.out.println("Factors: " + fmModel.factors());
        System.out.println("Linear: " + fmModel.linear());
        System.out.println("Intercept: " + fmModel.intercept());

        RegressionMetrics metrics = evaluator.getMetrics(predictions);
        System.out.println(metrics);

    }
}
