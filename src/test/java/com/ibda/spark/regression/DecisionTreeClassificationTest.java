package com.ibda.spark.regression;

import com.ibda.spark.util.SparkUtil;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.DecisionTreeRegressionModel;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.junit.Test;

import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

public class DecisionTreeClassificationTest extends SparkMLTest<DecisionTreeClassifier,DecisionTreeClassificationModel>{

    protected void initTrainingParams(){
        trainingParams.put("impurity", "gini");
        trainingParams.put("minInfoGain", 0.0d);
        trainingParams.put("maxDepth", 10); //default 5
        trainingParams.put("maxBins", 64);  //default 32 必须是2的整数幂
        trainingParams.put("minInstancesPerNode", 10); //default 1
    }


    //@Test
    public void decisionTreeClassificationDemo(){
        // Load the data stored in LIBSVM format as a DataFrame.
        SparkSession spark = SparkUtil.buildSparkSession(this.getClass().getSimpleName());
        Dataset<Row> data = spark
                .read()
                .format("libsvm")
                .load("data/mllib/sample_libsvm_data.txt");

        // Index labels, adding metadata to the label column.
        // Fit on whole dataset to include all labels in index.
        StringIndexerModel labelIndexer = new StringIndexer()
                .setInputCol("label")
                .setOutputCol("indexedLabel")
                .fit(data);

        // Automatically identify categorical features, and index them.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
                .fit(data);

        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("indexedLabel")
                .setFeaturesCol("indexedFeatures");

        // Convert indexed labels back to original labels.
        IndexToString labelConverter = new IndexToString()
                .setInputCol("prediction")
                .setOutputCol("predictedLabel")
                .setLabels(labelIndexer.labelsArray()[0]);

        // Chain indexers and tree in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{labelIndexer, featureIndexer, dt, labelConverter});

        // Train model. This also runs the indexers.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("predictedLabel", "label", "features").show(5);

        // Select (prediction, true label) and compute test error.
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("indexedLabel")
                .setPredictionCol("prediction")
                .setMetricName("accuracy");
        double accuracy = evaluator.evaluate(predictions);
        System.out.println("Test Error = " + (1.0 - accuracy));

        DecisionTreeClassificationModel treeModel =
                (DecisionTreeClassificationModel) (model.stages()[2]);
        System.out.println("Learned classification tree model:\n" + treeModel.toDebugString());
    }

    //@Test
    public void decisionTreeRegressionDemo(){
        SparkSession spark = SparkUtil.buildSparkSession(this.getClass().getSimpleName());
        // Load the data stored in LIBSVM format as a DataFrame.
        Dataset<Row> data = spark.read().format("libsvm")
                .load("data/mllib/sample_libsvm_data.txt");

        // Automatically identify categorical features, and index them.
        // Set maxCategories so features with > 4 distinct values are treated as continuous.
        VectorIndexerModel featureIndexer = new VectorIndexer()
                .setInputCol("features")
                .setOutputCol("indexedFeatures")
                .setMaxCategories(4)
                .fit(data);

        // Split the data into training and test sets (30% held out for testing).
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Train a DecisionTree model.
        DecisionTreeRegressor dt = new DecisionTreeRegressor()
                .setFeaturesCol("indexedFeatures");

        // Chain indexer and tree in a Pipeline.
        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{featureIndexer, dt});

        // Train model. This also runs the indexer.
        PipelineModel model = pipeline.fit(trainingData);

        // Make predictions.
        Dataset<Row> predictions = model.transform(testData);

        // Select example rows to display.
        predictions.select("label", "features").show(5);

        // Select (prediction, true label) and compute test error.
        RegressionEvaluator evaluator = new RegressionEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("rmse");
        double rmse = evaluator.evaluate(predictions);
        System.out.println("Root Mean Squared Error (RMSE) on test data = " + rmse);

        DecisionTreeRegressionModel treeModel =
                (DecisionTreeRegressionModel) (model.stages()[1]);
        System.out.println("Learned regression tree model:\n" + treeModel.toDebugString());
    }

    //@Test
    public void testCreditRatingClassification() throws IOException {
        SparkML<DecisionTreeClassifier, DecisionTreeClassificationModel> decisionTree = new SparkML<>(null,DecisionTreeClassifier.class);

        //Credit_rating,Age,Income,Credit_cards,Education,Education
        ModelColumns modelColumns = new ModelColumns(
                new String[]{"Age", "Income", "Credit_cards", "Education", "Education"},
                null,
                "Credit_rating");

        Dataset<Row>  creditRating = decisionTree.loadData(FilePathUtil.getAbsolutePath("data/credit_decision_tree.csv", false));
        PipelineModel pipelineModel = modelColumns.fit(creditRating);
        Map<String, Object> param = new HashMap<String, Object>();
        param.put("impurity", "gini");
        param.put("minInfoGain", 0.0d);
        param.put("maxDepth", 10); //default 5
        param.put("maxBins", 64);  //default 32 必须是2的整数幂
        param.put("minInstancesPerNode", 10); //default 1
        //过滤训练及测试集
        System.out.println("测试决策树分析,credit_rating数据集------------------");
        Dataset<Row> trainAndTest = creditRating.filter("Credit_rating is not null");
        //划分训练集、测试集
        Dataset<Row>[] datasets = trainAndTest.randomSplit(new double[]{0.7d, 0.3d});
        Dataset<Row> training = datasets[0];
        Dataset<Row> testing = datasets[1];
        System.out.println(String.format("记录总数：%1$s,训练集大小：%2$s,测试集大小：%3$s",trainAndTest.count(),training.count(),testing.count()));

        //训练
        System.out.println("训练决策树分类模型------------------");
        SparkHyperModel<DecisionTreeClassificationModel> treeModel = decisionTree.fit(training, modelColumns,pipelineModel, param);
        System.out.println("训练模型结果及性能\n:" + treeModel);
        System.out.println(treeModel.getModel().toDebugString());
        //评估
        Map<String, Object> metrics = treeModel.evaluate(testing);
        System.out.println("评估模型性能\n:" + metrics);
        Dataset<Row> tested = SparkHyperModel.getEvaluatePredictions(metrics);
        tested.show();
        //预测
        Dataset<Row> predicting = creditRating.filter("Credit_rating is null");
        System.out.println("预测数据集:" + predicting.count());
        predicting.show();
        Dataset<Row> predicted = treeModel.predict(predicting);
        System.out.println("预测结果集:" + predicted.count());
        predicted.show();

        //预测单个数据
        Row[] rows = (Row[])predicted.select("regression_features_vector").head(20);
        Arrays.stream(rows).forEach(row->{
            GenericRowWithSchema gRow = (GenericRowWithSchema)row;
            Vector data = (Vector)gRow.values()[0];
            double label = decisionTree.predict(treeModel.getModel(),data);
            System.out.println(data.toString() + ":" + label);
        });


        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/credit_rating_decision_tree.model", true);
        System.out.println("保存及加载模型："  + modelPath);
        treeModel.saveModel(modelPath);

        SparkHyperModel<DecisionTreeClassificationModel> result3= SparkHyperModel.loadFromModelFile(modelPath, DecisionTreeClassificationModel.class);
        Map<String, Object> metrics2 = result3.evaluate(testing, modelColumns,pipelineModel );
        System.out.println("评估存储模型性能\n:" + metrics2);

        System.out.println("使用存储模型进行预测\n:");
        Dataset<Row> predicted2 = decisionTree.predict(predicting,modelColumns,pipelineModel,result3.getModel());
        predicted2.show();
    }

    //@Test
    public void testCarScoreRegression() throws IOException {
        //car,age,gender,inccat,ed,marital
        SparkML<DecisionTreeRegressor, DecisionTreeRegressionModel> treeRegression = new SparkML<>(null,DecisionTreeRegressor.class);

        //car,age,gender,inccat,ed,marital
        ModelColumns modelColumns = new ModelColumns(
                new String[]{"age", "inccat","gender", "ed", "marital"},
                null,
                new String[]{"gender"},
                "car");

        Dataset<Row>  carSales = treeRegression.loadData(FilePathUtil.getAbsolutePath("data/car_decision_tree.csv", false));
        PipelineModel pipelineModel = modelColumns.fit(carSales);
        Map<String, Object> param = new HashMap<String, Object>();
        param.put("impurity", "variance");
        param.put("minInfoGain", 0.0d);
        param.put("maxDepth", 10); //default 5
        param.put("maxBins", 64);  //default 32 必须是2的整数幂
        param.put("minInstancesPerNode", 10); //default 1
        //过滤训练及测试集
        System.out.println("测试决策树回归,car数据集------------------");
        Dataset<Row> trainAndTest = carSales.filter("car is not null");
        //划分训练集、测试集
        Dataset<Row>[] datasets = trainAndTest.randomSplit(new double[]{0.7d, 0.3d});
        Dataset<Row> training = datasets[0];
        Dataset<Row> testing = datasets[1];
        System.out.println(String.format("记录总数：%1$s,训练集大小：%2$s,测试集大小：%3$s",trainAndTest.count(),training.count(),testing.count()));

        //训练
        System.out.println("训练决策树回归模型------------------");
        SparkHyperModel<DecisionTreeRegressionModel> treeModel = treeRegression.fit(training, modelColumns,pipelineModel, param);
        System.out.println("训练模型结果及性能\n:" + treeModel);
        System.out.println(treeModel.getModel().toDebugString());
        //评估
        Map<String, Object> metrics = treeModel.evaluate(testing);
        System.out.println("评估回归模型性能\n:" + metrics);
        Dataset<Row> tested = SparkHyperModel.getEvaluatePredictions(metrics);
        tested.show();
        //预测
        Dataset<Row> predicting = carSales.filter("car is null");
        System.out.println("预测数据集:" + predicting.count());
        predicting.show();
        Dataset<Row> predicted = treeModel.predict(predicting);
        System.out.println("预测结果集:" + predicted.count());
        predicted.show();

        //预测单个数据
        Row[] rows = (Row[])predicted.select("regression_features_vector").head(20);
        Arrays.stream(rows).forEach(row->{
            GenericRowWithSchema gRow = (GenericRowWithSchema)row;
            Vector data = (Vector)gRow.values()[0];
            double label = treeRegression.predict(treeModel.getModel(),data);
            System.out.println(data.toString() + ":" + label);
        });


        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/car_tree_regression.model", true);
        System.out.println("保存及加载模型："  + modelPath);
        treeModel.saveModel(modelPath);

        SparkHyperModel<DecisionTreeRegressionModel> result3= SparkHyperModel.loadFromModelFile(modelPath, DecisionTreeRegressionModel.class);
        Map<String, Object> metrics2 = result3.evaluate(testing, modelColumns,pipelineModel );
        System.out.println("评估存储模型性能\n:" + metrics2);

        System.out.println("使用存储模型进行预测\n:");
        Dataset<Row> predicted2 = treeRegression.predict(predicting,modelColumns,pipelineModel,result3.getModel());
        predicted2.show();
    }
}
