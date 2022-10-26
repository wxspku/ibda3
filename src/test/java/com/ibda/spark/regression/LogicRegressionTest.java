package com.ibda.spark.regression;

import com.ibda.util.AnalysisConst;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.linalg.DenseMatrix;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import static com.ibda.spark.regression.ModelColumns.MODEL_COLUMNS_DEFAULT;

public class LogicRegressionTest {
    LogicRegression regression = new LogicRegression(null);
    Dataset<Row> bankloans = null;
    PipelineModel pipelineModel = null;
    //age,ed,employ,address,income,debtinc,creddebt,othdebt,default
    ModelColumns modelColumns = new ModelColumns(
            new String[]{"age", "employ", "address", "income", "debtinc", "creddebt", "othdebt"},
            new String[]{"ed"},
            "default");

    @Before
    public void setUp() throws Exception {
        bankloans = regression.loadData(FilePathUtil.getAbsolutePath("data/bankloan_logistic.csv", false));
        pipelineModel = modelColumns.fit(bankloans);
    }

    @After
    public void tearDown() throws Exception {
    }

    @Test
    public void preProcess() {
        Dataset<Row> processed = modelColumns.transform(bankloans,pipelineModel);
        DenseMatrix matrix = regression.getCorrelationMatrix(processed,
                modelColumns.featuresCol,
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
        System.out.println("测试二元逻辑回归分析,bankloan数据集------------------");
        Dataset<Row> trainAndTest = bankloans.filter("default is not null");
        //划分训练集、测试集
        Dataset<Row>[] datasets = trainAndTest.randomSplit(new double[]{0.7d, 0.3d});
        Dataset<Row> training = datasets[0];
        Dataset<Row> testing = datasets[1];
        System.out.println(String.format("记录总数：%1$s,训练集大小：%2$s,测试集大小：%3$s",trainAndTest.count(),training.count(),testing.count()));

        //训练
        System.out.println("训练二元逻辑回归模型，同时训练预处理模型------------------");
        RegressionHyperModel result = regression.fit(training, modelColumns, param);
        System.out.println("训练模型结果及性能\n:" + result);
        //评估
        Map<String, Object> metrics = result.evaluate(testing);
        System.out.println("评估模型性能\n:" + metrics);
        Dataset<Row> tested = (Dataset<Row>)metrics.get(RegressionHyperModel.PREDICTIONS_KEY);
        tested.show();
        //预测
        Dataset<Row> predicting = bankloans.filter("default is null");
        System.out.println("预测数据集:" + predicting.count());
        predicting.show();
        Dataset<Row> predicted = result.predict(predicting);
        System.out.println("预测结果集:" + predicted.count());
        predicted.show();
        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/bankloan_logistic.model", true);
        System.out.println("保存及加载模型："  + modelPath);
        result.saveModel(modelPath);

        RegressionHyperModel result3= LogicRegressionHyperModel.loadFromModelFile(modelPath);
        Map<String, Object> metrics2 = result3.evaluate(testing, modelColumns,pipelineModel );
        System.out.println("评估存储模型性能\n:" + metrics2);

        System.out.println("使用存储模型进行预测\n:");
        Dataset<Row> predicted2 = regression.predict(predicting,modelColumns,pipelineModel,result3.getModel());
        predicted2.show();

        //libsvm数据集
        System.out.println("测试多元逻辑回归分析,libsvm数据集------------------");
        Dataset<Row> libsvm = regression.loadData(FilePathUtil.getAbsolutePath("data/mllib/sample_multiclass_classification_data.txt",true),
                "libsvm");
        Dataset<Row>[] libsvmDatasets = libsvm.randomSplit(new double[]{0.8d, 0.2d});
        Dataset<Row> libsvm_training = libsvmDatasets[0];
        Dataset<Row> libsvm_testing = libsvmDatasets[1];
        System.out.println(String.format("记录总数：%1$s,训练集大小：%2$s,测试集大小：%3$s",libsvm.count(),libsvm_training.count(),libsvm_testing.count()));

        RegressionHyperModel result2 = regression.fit(libsvm_training,MODEL_COLUMNS_DEFAULT, param);
        System.out.println("训练多元逻辑回归模型完成：" + result2);
        result2.getPredictions().show();


        Map<String, Object> libsvm_metrics = result2.evaluate(libsvm_testing);
        System.out.println("测试性能\n:" + libsvm_metrics);

        //TrainValidationSplit超参数调优，用于在训练模型时对候选参数进行逐一训练和评估，最终选择最优的模型
        //候选参数根据ParamGridBuilder()指定，模型评估通过Evaluator指定
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