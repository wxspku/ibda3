package com.ibda.spark;

import com.ibda.spark.regression.ModelColumns;
import com.ibda.spark.regression.SparkHyperModel;
import com.ibda.spark.regression.SparkML;
import com.ibda.spark.statistics.DescriptiveStatistics;
import com.ibda.util.AnalysisConst;
import com.ibda.util.FilePathUtil;
import org.apache.spark.ml.Estimator;
import org.apache.spark.ml.Model;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.regression.IsotonicRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.catalyst.expressions.GenericRowWithSchema;
import org.junit.After;
import org.junit.Before;
import org.junit.FixMethodOrder;
import org.junit.Test;
import org.junit.runners.MethodSorters;

import java.io.IOException;
import java.lang.reflect.ParameterizedType;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

@FixMethodOrder(MethodSorters.NAME_ASCENDING)
public abstract class SparkMLTest<E extends Estimator, M extends Model> {
    //特征是否进行最大值
    protected boolean scaleByMinMax = false;

    //是否划分训练集和测试集
    protected boolean trainingSplit = true;
    /**
     * 训练类
     */
    protected Class<E> estimatorClass = (Class<E>) ((ParameterizedType) this.getClass().getGenericSuperclass()).getActualTypeArguments()[0];
    /**
     * 模型类
     */
    protected Class<M> modelClass = (Class<M>) ((ParameterizedType) this.getClass().getGenericSuperclass()).getActualTypeArguments()[1];
    protected SparkML<E, M> sparkLearning = null;
    protected SparkSession spark = null;
    protected PipelineModel pipelineModel = null;
    protected ModelColumns modelColumns = null;
    //数组：训练集、测试集、预测集
    protected Dataset<Row>[] datasets = new Dataset[3];
    protected Map<String, Object> trainingParams = new HashMap<String, Object>();

    //处理缺省值
    protected void imputeTrainingDataset(){

    }

    protected void loadDataSet(String datafile, String format) {
        System.out.println("加载数据集:" + datafile);
        Dataset allData = sparkLearning.loadData(datafile, format);
        splitDataset(allData);
    }

    protected void splitDataset(Dataset originalData) {
        Dataset<Row> trainAndTest = originalData.filter(modelColumns.getLabelCol() + " is not null");
        //划分训练集、测试集
        double[] splitting = trainingSplit ? new double[]{0.7d, 0.3d} : new double[]{1d, 0.0d};
        Dataset<Row>[] trainAndTests = trainAndTest.randomSplit(splitting, System.currentTimeMillis());
        datasets[0] = trainAndTests[0];
        datasets[1] = trainAndTests[1];
        //预测集
        datasets[2] = originalData.filter(modelColumns.getLabelCol() + " is null");
        imputeTrainingDataset();
        System.out.println(String.format("记录总数：%1$s,训练集大小：%2$s,测试集大小：%3$s,预测集大小：%4$s", originalData.count(), datasets[0].count(), datasets[1].count(), datasets[2].count()));
        pipelineModel = modelColumns.fit(datasets[0], scaleByMinMax);
    }

    /**
     * 准备数据、设置训练参数
     */
    @Before
    public void prepareData() {
        sparkLearning = new SparkML<>(estimatorClass);
        spark = sparkLearning.getSpark();
        loadTest01Data();
        initTrainingParams();
    }

    @After
    public void destroy() throws Throwable {
        sparkLearning.finalize();
    }

    @Test
    public void test01LearningEvaluatingPredicting() throws IOException {
        //训练
        System.out.println("训练模型：" + estimatorClass.getSimpleName() + "/" + modelClass.getSimpleName());
        SparkHyperModel<M> hyperModel = sparkLearning.fit(datasets[0], modelColumns, pipelineModel, trainingParams);
        System.out.println("训练模型结果及性能\n:" + hyperModel);
        if (hyperModel.getPredictions() != null) {
            hyperModel.getPredictions().show(100);
        }
        //评估
        Map<String, Object> metrics = hyperModel.evaluate(datasets[1]);
        System.out.println("评估模型性能\n:" + metrics);
        Dataset<Row> tested = SparkHyperModel.getEvaluatePredictions(metrics);
        tested.show();
        //预测
        Dataset<Row> predicting = datasets[2];//carSales.filter("lnsales is null");
        System.out.println("预测数据集:" + predicting.count());
        predicting.show();
        Dataset<Row> predicted = hyperModel.predict(predicting);
        System.out.println("预测结果集:" + predicted.count());
        predicted.show();

        //预测单个数据
        if (hyperModel.getModel() instanceof PredictionModel) {
            Row[] rows = (Row[]) predicted.select(modelColumns.getFeaturesCol()).head(20);
            Arrays.stream(rows).forEach(row -> {
                GenericRowWithSchema gRow = (GenericRowWithSchema) row;
                Vector data = (Vector) gRow.values()[0];
                double label = sparkLearning.predict((PredictionModel) hyperModel.getModel(), data);
                System.out.println(data.toString() + ":" + label);
            });
        } else if (hyperModel.getModel() instanceof IsotonicRegressionModel) {
            IsotonicRegressionModel model = ((IsotonicRegressionModel) hyperModel.getModel());
            System.out.println("Boundaries in increasing order: " + model.boundaries() + "\n");
            System.out.println("Predictions associated with the boundaries: " + model.predictions() + "\n");
            DescriptiveStatistics descriptiveStatistics = sparkLearning.getDescriptiveStatistics(datasets[0], "features", null,
                    new AnalysisConst.DescriptiveTarget[]{AnalysisConst.DescriptiveTarget.min, AnalysisConst.DescriptiveTarget.max});
            for (int i = 0; i <= 20; i++) {
                Random random = new Random(System.currentTimeMillis() + i);
                double x = descriptiveStatistics.getMin()[0] + (descriptiveStatistics.getMax()[0] - descriptiveStatistics.getMin()[0]) * random.nextDouble();
                double y = model.predict(x);
                System.out.println("x=" + x + ": y=" + y);
            }
        }
        //模型读写
        System.out.println("测试读写模型---------------");
        String modelPath = FilePathUtil.getAbsolutePath("output/" + modelClass.getSimpleName() + ".model", true);
        System.out.println("保存及加载模型：" + modelPath);
        hyperModel.saveModel(modelPath);

        SparkHyperModel<M> loadedModel = SparkHyperModel.loadFromModelFile(modelPath, modelClass);
        Map<String, Object> metrics2 = loadedModel.evaluate(datasets[1], modelColumns, pipelineModel);
        System.out.println("评估存储模型性能\n:" + metrics2);

        System.out.println("使用存储模型进行预测\n:");
        Dataset<Row> predicted2 = loadedModel.predict(datasets[2], modelColumns, pipelineModel);
        predicted2.show();
    }

    @Test
    public void test02MachineLearning() throws IOException {
        loadTest02Data();
        test01LearningEvaluatingPredicting();
    }

    //加载test01的测试数据
    protected void loadTest01Data() {

    }

    protected void loadTest02Data() {

    }

    public abstract void initTrainingParams();
}
