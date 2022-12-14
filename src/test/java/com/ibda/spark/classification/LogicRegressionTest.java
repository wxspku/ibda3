package com.ibda.spark.classification;

import cn.hutool.core.util.ReflectUtil;
import com.ibda.spark.regression.ModelColumns;
import com.ibda.spark.regression.SparkHyperModel;
import com.ibda.spark.regression.SparkML;
import com.ibda.util.AnalysisConst;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.lang.reflect.Field;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static com.ibda.spark.regression.ModelColumns.MODEL_COLUMNS_DEFAULT;

public class LogicRegressionTest {
    SparkML<LogisticRegression, LogisticRegressionModel> logistic = new SparkML<>(null, LogisticRegression.class);
    Dataset<Row> bankloans = null;
    PipelineModel pipelineModel = null;
    //age,ed,employ,address,income,debtinc,creddebt,othdebt,default
    ModelColumns modelColumns = new ModelColumns(
            new String[]{"age", "employ", "address", "income", "debtinc", "creddebt", "othdebt"},
            new String[]{"ed"},
            "default");

    @Before
    public void setUp() throws Exception {
        bankloans = logistic.loadData(FilePathUtil.getAbsolutePath("data/bankloan_logistic.csv", false));
        pipelineModel = modelColumns.fit(bankloans);
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void testGetEClass() {
        Field field = ReflectUtil.getField(getClass(), "logistic");
        Type type = field.getGenericType();
        ParameterizedType parameterizedType = (ParameterizedType) type;
        Type actualType = parameterizedType.getActualTypeArguments()[0];
        System.out.println(actualType);
    }

    @Test
    public void preProcess() {
        Dataset<Row> processed = modelColumns.transform(bankloans, pipelineModel);
        DenseMatrix matrix = logistic.getCorrelationMatrix(processed,
                modelColumns.getFeaturesCol(),
                AnalysisConst.CorrelationMethod.pearson);
        System.out.println(matrix);
        processed.show();
    }

    @Test
    public void fit() throws IOException {
        Map<String, Object> param = new HashMap<String, Object>();
        param.put("maxIter", 100);
        param.put("tol", 1E-8);
        param.put("threshold", 0.5);
        /*ParamMap paramMap = new ParamMap()
                .put(lr.maxIter().w(20))  // Specify 1 Param.
                .put(lr.maxIter(), 30)  // This overwrites the original maxIter.
                .put(lr.regParam().w(0.1), lr.threshold().w(0.55));  // Specify multiple Params.*/
        System.out.println("??????????????????????????????,bankloan?????????------------------");
        Dataset<Row> trainAndTest = bankloans.filter("default is not null");
        //???????????????????????????
        Dataset<Row>[] datasets = trainAndTest.randomSplit(new double[]{0.8d, 0.2d});
        Dataset<Row> training = datasets[0];
        Dataset<Row> testing = datasets[1];
        System.out.println(String.format("???????????????%1$s,??????????????????%2$s,??????????????????%3$s", trainAndTest.count(), training.count(), testing.count()));

        //??????
        System.out.println("????????????????????????????????????????????????????????????------------------");
        SparkHyperModel<LogisticRegressionModel> logisticHyperModel = logistic.fit(training, modelColumns, param);
        System.out.println("???????????????????????????\n:" + logisticHyperModel);
        //??????
        Map<String, Object> metrics = logisticHyperModel.evaluate(testing);
        System.out.println("??????????????????\n:" + metrics);
        Dataset<Row> tested = SparkHyperModel.getEvaluatePredictions(metrics);
        tested.show();
        //??????
        Dataset<Row> predicting = bankloans.filter("default is null");
        System.out.println("???????????????:" + predicting.count());
        predicting.show();
        Dataset<Row> predicted = logisticHyperModel.predict(predicting);
        System.out.println("???????????????:" + predicted.count());
        predicted.show();

        //??????????????????
        Row[] rows = (Row[]) predicted.select("regression_features_vector").head(20);
        Arrays.stream(rows).forEach(row -> {
            GenericRowWithSchema gRow = (GenericRowWithSchema) row;
            DenseVector data = (DenseVector) gRow.values()[0];
            double label = logistic.predict(logisticHyperModel.getModel(), data);
            System.out.println(data.toString() + ":" + label);
        });


        //????????????
        System.out.println("??????????????????---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/bankloan_logistic.model", true);
        System.out.println("????????????????????????" + modelPath);
        logisticHyperModel.saveModel(modelPath);

        SparkHyperModel<LogisticRegressionModel> result3 = SparkHyperModel.loadFromModelFile(modelPath, LogisticRegressionModel.class);
        Map<String, Object> metrics2 = result3.evaluate(testing, modelColumns, pipelineModel);
        System.out.println("????????????????????????\n:" + metrics2);

        System.out.println("??????????????????????????????\n:");
        Dataset<Row> predicted2 = logistic.predict(predicting, modelColumns, pipelineModel, result3.getModel());
        predicted2.show();

        //libsvm?????????
        System.out.println("??????????????????????????????,libsvm?????????------------------");
        Dataset<Row> libsvm = logistic.loadData(FilePathUtil.getAbsolutePath("data/mllib/sample_multiclass_classification_data.txt", true),
                "libsvm");
        Dataset<Row>[] libsvmDatasets = libsvm.randomSplit(new double[]{0.8d, 0.2d});
        Dataset<Row> libsvm_training = libsvmDatasets[0];
        Dataset<Row> libsvm_testing = libsvmDatasets[1];
        System.out.println(String.format("???????????????%1$s,??????????????????%2$s,??????????????????%3$s", libsvm.count(), libsvm_training.count(), libsvm_testing.count()));

        SparkHyperModel result2 = logistic.fit(libsvm_training, MODEL_COLUMNS_DEFAULT, param);
        System.out.println("???????????????????????????????????????" + result2);
        result2.getPredictions().show();


        Map<String, Object> libsvm_metrics = result2.evaluate(libsvm_testing);
        System.out.println("????????????\n:" + libsvm_metrics);

        //TrainValidationSplit??????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        //??????????????????ParamGridBuilder()???????????????????????????Evaluator??????
        /*TrainValidationSplit split = new TrainValidationSplit();
        split.setTrainRatio(0.8);
        TrainValidationSplitModel splitModel = split.fit(trainAndTest);
        Dataset<Row> transform = splitModel.transform(trainAndTest);
        System.out.println(transform);*/
    }

    @Test
    public void evaluate() {
    }

    @Test
    public void predict() {
    }
}